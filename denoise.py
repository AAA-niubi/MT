#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

model_dir = os.path.join(os.path.dirname(__file__), "models")
if model_dir not in sys.path:
    sys.path.append(model_dir)

from TimeDART_v2 import Model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    use_gpu = True
    use_multi_gpu = False
    gpu = 0
    devices = "0"
    device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"

    input_path = os.path.join("data", "sample_QH", "self-supervused_data", "test", "test_samples_TS5.npy")
    means_path = os.path.join("data", "sample_QH", "self-supervused_data", "test", "test_means_TS5.npy")
    stds_path  = os.path.join("data", "sample_QH", "self-supervused_data", "test", "test_stds_TS5.npy")
    indices_path = os.path.join("data", "sample_QH", "self-supervused_data", "test", "test_indices_TS5.npy")
    output_path = "./denoised_results_test1"

    trimmed_data_dir = os.path.join("DATA", "sample_QH", "trimmed_data")
    dataset_type = "test"
    normalization_method = "auto"
    enable_denormalization = True
    save_denormalized_result = True

    required_time_steps = 450
    required_channels = 5

    patch_len = 16
    stride = 16
    expected_patch_seq_len = int((required_time_steps - patch_len) / stride) + 1

    task_name = "pretrain"
    input_len = 450
    pred_len = 450
    d_model = 256
    n_heads = 4
    d_ff = 512
    dropout = 0.2
    head_dropout = 0.2
    time_steps = 2000
    scheduler = "cosine"
    mask_ratio = 0.3
    e_layers = 3
    d_layers = 3
    backbone = "Transformer"
    llm_path = None

    batch_size = 96
    model_ckpt = "./checkpoints/best_model.pth"

    confidence_threshold = 0.8
    enable_confidence_filtering = True

    stage1_only = False

    noise_weight = 0.3
    recon_weight = 0.7
    adaptive_weights = True

    enable_spike_suppression = True
    spike_threshold = 2.0
    spike_window_size = 7
    spike_suppression_strength = 1
    min_spike_width = 1
    max_spike_width = 10

    enable_smoothing_constraints = True
    inner_smooth_weight = 3.0
    boundary_align_weight = 2.0
    anchor_extend_length = 3
    smoothing_loss_weight = 1.5

    enable_iterative_denoising = False
    denoise_iterations = 1
    iteration_confidence_decay = 0.95
    iteration_spike_threshold_decay = 0.9
    enable_iteration_visualization = False
    convergence_threshold = 0.00
    enable_early_stopping = True

    use_fixed_noise_regions = True
    noise_region_expansion = 0

    enable_timefreq_fusion = True

    use_timefreq_mask_file = True
    timefreq_mask_path = "noise_masks_from_bright_bands.npy"

    enable_edge_pair_analysis = False
    linearity_threshold = 0.95
    inflection_threshold = 2
    slope_change_threshold = 0.3
    area_time_ratio_threshold = 0.5
    edge_morph_min_run_remove = 3
    edge_morph_max_gap_fill = 5
    edge_morph_min_run_keep = 1
    edge_morph_min_keep_ratio = 0.5

    enable_square_wave_detection = False
    square_wave_min_length = 10
    square_wave_shift_coverage_threshold = 0.95
    band_data_dir = os.path.join("vmd_outlierfree_ultrasmooth_results", "data")
    band_upper_path = os.path.join("vmd_outlierfree_ultrasmooth_results", "data", "upper_band.npy")
    band_lower_path = os.path.join("vmd_outlierfree_ultrasmooth_results", "data", "lower_band.npy")
    band_midline_path = os.path.join("vmd_outlierfree_ultrasmooth_results", "data", "ultrasmooth_lowfreq_data.npy")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args: Config) -> Model:
    model = Model(args).to(args.device)
    if not os.path.exists(args.model_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_ckpt}")
    checkpoint = torch.load(args.model_ckpt, map_location=args.device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except KeyError as e:
        raise KeyError(f"Missing key in checkpoint: {e}") from e
    if "best_test_loss" not in checkpoint:
        raise ValueError("Checkpoint was not saved by the current training script; please retrain.")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_and_fix_input(signal: np.ndarray, args: Config):
    if signal.ndim == 2:
        signal = signal[..., np.newaxis]
    elif signal.ndim == 1:
        signal = signal[np.newaxis, ..., np.newaxis]
    elif signal.ndim != 3:
        raise ValueError(f"Expected 1-D/2-D/3-D input, got {signal.ndim}-D.")

    if signal.shape[1] != args.required_time_steps:
        raise ValueError(
            f"Time-step mismatch: model requires {args.required_time_steps}, got {signal.shape[1]}."
        )
    if signal.shape[2] != args.required_channels:
        raise ValueError(
            f"Channel mismatch: model requires {args.required_channels}, got {signal.shape[2]}."
        )

    sample_indices = np.arange(signal.shape[0])
    return signal, sample_indices


# ---------------------------------------------------------------------------
# Stage 1 – noise classification
# ---------------------------------------------------------------------------

def stage1_noise_classification(model: Model, input_signal: np.ndarray, args: Config):
    input_tensor = torch.from_numpy(input_signal).float()
    dataloader = DataLoader(
        TensorDataset(input_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 4,
        pin_memory=args.device == "cuda",
    )

    all_noise_probs, all_noise_masks, all_cls_outputs = [], [], []

    with torch.no_grad():
        for (batch_data,) in dataloader:
            batch_noisy = batch_data.to(args.device)
            batch_size, input_len, num_features = batch_noisy.size()

            x_ci = model.channel_independence(batch_noisy)
            x_patch = model.patch(x_ci)
            emb = model.enc_embedding(x_patch)
            emb_bias = model.add_sos_token_and_drop_last(emb)
            emb_bias = model.positional_encoding(emb_bias)

            if model.args.backbone == "Qwen2.5-0.5B":
                enc_out = model.encoder(inputs_embeds=emb_bias, output_hidden_states=True)
                cls_feat = enc_out.hidden_states[-1]
            else:
                cls_feat = model.encoder(emb_bias, is_mask=True)

            cls_patch = cls_feat.reshape(batch_size, num_features, -1, model.d_model)
            cls_out = model.projection(cls_patch)

            def _align_to_input(t: torch.Tensor, target_len: int) -> torch.Tensor:
                if t.dim() == 4:
                    b, c, s, p = t.shape
                    t = t.reshape(b, c, -1)
                if t.size(-1) != target_len:
                    t = torch.nn.functional.interpolate(
                        t.transpose(1, 2), size=target_len, mode='linear', align_corners=False
                    ).transpose(1, 2)
                if t.dim() == 3 and t.size(1) != target_len:
                    t = torch.nn.functional.interpolate(
                        t.transpose(1, 2), size=target_len, mode='linear', align_corners=False
                    ).transpose(1, 2)
                return t.view(batch_size, target_len, num_features)

            cls_out = _align_to_input(cls_out, input_len)

            noise_prob = torch.sigmoid(torch.abs(cls_out - batch_noisy))
            noise_mask = (noise_prob > args.confidence_threshold).float()

            all_noise_probs.append(noise_prob.cpu().numpy())
            all_noise_masks.append(noise_mask.cpu().numpy())
            all_cls_outputs.append(cls_out.cpu().numpy())

    return (
        np.concatenate(all_noise_probs, axis=0),
        np.concatenate(all_noise_masks, axis=0),
        np.concatenate(all_cls_outputs, axis=0),
    )


# ---------------------------------------------------------------------------
# Stage 2 – selective denoising
# ---------------------------------------------------------------------------

def stage2_denoising(model: Model, input_signal: np.ndarray, noise_masks: np.ndarray, args: Config) -> np.ndarray:
    input_tensor = torch.from_numpy(input_signal).float()
    if isinstance(noise_masks, np.ndarray):
        mask_tensor = torch.from_numpy(noise_masks.astype(np.float32)).bool()
    else:
        mask_tensor = torch.from_numpy(np.array(noise_masks, dtype=np.float32)).bool()

    dataloader = DataLoader(
        TensorDataset(input_tensor, mask_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if sys.platform == "win32" else 4,
        pin_memory=args.device == "cuda",
    )

    denoised_results = []

    with torch.no_grad():
        for batch_data, batch_noise_mask in dataloader:
            batch_noisy = batch_data.to(args.device)
            batch_noise_mask = batch_noise_mask.to(args.device)

            try:
                batch_denoised = model(batch_noisy)
                denoised_results.append(batch_denoised.cpu().numpy())

            except RuntimeError as e:
                if "must match the size" not in str(e):
                    raise
                batch_size, seq_len, num_features = batch_noisy.shape
                patch_len, stride = args.patch_len, args.stride
                expected_patches = int((seq_len - patch_len) / stride) + 1
                reconstructed_len = (expected_patches - 1) * stride + patch_len

                if reconstructed_len == seq_len:
                    raise

                if reconstructed_len < seq_len:
                    adjusted = batch_noisy[:, :reconstructed_len, :]
                else:
                    padding = torch.zeros(
                        batch_size, reconstructed_len - seq_len, num_features,
                        device=batch_noisy.device,
                    )
                    adjusted = torch.cat([batch_noisy, padding], dim=1)

                batch_denoised = model(adjusted)

                if batch_denoised.shape[1] != seq_len:
                    if batch_denoised.shape[1] > seq_len:
                        batch_denoised = batch_denoised[:, :seq_len, :]
                    else:
                        batch_denoised = torch.nn.functional.interpolate(
                            batch_denoised.transpose(1, 2),
                            size=seq_len,
                            mode='linear',
                            align_corners=False,
                        ).transpose(1, 2)

                denoised_results.append(batch_denoised.cpu().numpy())

    return np.concatenate(denoised_results, axis=0)


# ---------------------------------------------------------------------------
# Post-processing – spike suppression
# ---------------------------------------------------------------------------

def spike_suppression_post_process(
    signal: np.ndarray,
    noise_masks: np.ndarray,
    spike_threshold: float = 2.5,
    window_size: int = 5,
    suppression_strength: float = 1.0,
    min_spike_width: int = 1,
    max_spike_width: int = 10,
):
    signal = signal.copy()
    batch_size, time_steps, num_features = signal.shape
    total_spikes = 0

    for b in tqdm(range(batch_size), desc="spike suppression"):
        for ch in range(num_features):
            channel_signal = signal[b, :, ch]
            channel_noise_mask = noise_masks[b, :, ch]

            if not np.any(channel_noise_mask):
                continue

            noise_regions = []
            in_noise, start = False, 0
            for i, v in enumerate(channel_noise_mask):
                if v and not in_noise:
                    start, in_noise = i, True
                elif not v and in_noise:
                    noise_regions.append((start, i - 1))
                    in_noise = False
            if in_noise:
                noise_regions.append((start, time_steps - 1))

            for noise_start, noise_end in noise_regions:
                ext_s = max(0, noise_start - window_size)
                ext_e = min(time_steps - 1, noise_end + window_size)
                ext_sig = channel_signal[ext_s:ext_e + 1]
                ext_len = len(ext_sig)

                local_means = np.array([
                    np.mean(ext_sig[max(0, i - window_size // 2):min(ext_len, i + window_size // 2 + 1)])
                    for i in range(ext_len)
                ])
                local_stds = np.array([
                    np.std(ext_sig[max(0, i - window_size // 2):min(ext_len, i + window_size // 2 + 1)]) + 1e-8
                    for i in range(ext_len)
                ])

                rel_s = noise_start - ext_s
                rel_e = noise_end - ext_s
                region_sig = ext_sig[rel_s:rel_e + 1]
                region_means = local_means[rel_s:rel_e + 1]
                region_stds = local_stds[rel_s:rel_e + 1]

                spike_candidates = np.abs(region_sig - region_means) > (spike_threshold * region_stds)

                spike_regions = []
                in_spike, spike_start = False, 0
                for i, v in enumerate(spike_candidates):
                    if v and not in_spike:
                        spike_start, in_spike = i, True
                    elif not v and in_spike:
                        spike_regions.append((spike_start, i - 1))
                        in_spike = False
                if in_spike:
                    spike_regions.append((spike_start, len(spike_candidates) - 1))

                for ss, se in spike_regions:
                    width = se - ss + 1
                    if not (min_spike_width <= width <= max_spike_width):
                        continue
                    total_spikes += 1
                    abs_ss = noise_start + ss
                    abs_se = noise_start + se

                    for i in range(abs_ss, abs_se + 1):
                        ref_s = max(ext_s, i - window_size // 2)
                        ref_e = min(ext_e + 1, i + window_size // 2 + 1)
                        win_idx = [j for j in range(ref_s, ref_e) if not (abs_ss <= j <= abs_se)]
                        if not win_idx:
                            continue
                        median_val = np.median(channel_signal[win_idx])
                        if abs_ss > 0 and abs_se < time_steps - 1:
                            t = (i - abs_ss) / max(abs_se - abs_ss, 1)
                            linear_val = channel_signal[abs_ss - 1] * (1 - t) + channel_signal[abs_se + 1] * t
                        else:
                            linear_val = median_val
                        signal[b, i, ch] = (median_val + linear_val) / 2

    return signal, total_spikes


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def fuse_noise_masks(model_masks: np.ndarray, timefreq_masks: np.ndarray):
    fused = np.logical_and(model_masks, timefreq_masks)
    total = model_masks.size
    stats = {
        'model_noise_ratio':    float(np.sum(model_masks) / total),
        'timefreq_noise_ratio': float(np.sum(timefreq_masks) / total),
        'fused_noise_ratio':    float(np.sum(fused) / total),
        'model_noise_points':   int(np.sum(model_masks)),
        'timefreq_noise_points':int(np.sum(timefreq_masks)),
        'fused_noise_points':   int(np.sum(fused)),
        'reduction_ratio':      1.0 - float(np.sum(fused) / max(np.sum(model_masks), 1)),
    }
    return fused, stats


def expand_noise_regions(noise_masks: np.ndarray, expansion_pixels: int = 0) -> np.ndarray:
    if expansion_pixels <= 0:
        return noise_masks
    expanded = noise_masks.copy()
    batch_size, time_steps, num_channels = noise_masks.shape
    for b in range(batch_size):
        for ch in range(num_channels):
            mask = noise_masks[b, :, ch]
            regions = []
            in_n, start = False, 0
            for t, v in enumerate(mask):
                if v and not in_n:
                    start, in_n = t, True
                elif not v and in_n:
                    regions.append((start, t - 1))
                    in_n = False
            if in_n:
                regions.append((start, time_steps - 1))
            for rs, re in regions:
                expanded[b, max(0, rs - expansion_pixels):min(time_steps, re + expansion_pixels + 1), ch] = True
    return expanded


def save_noise_masks(noise_masks: np.ndarray, output_path: str, fusion_stats: Optional[Dict] = None) -> str:
    os.makedirs(output_path, exist_ok=True)
    mask_path = os.path.join(output_path, "noise_masks.npy")
    np.save(mask_path, noise_masks.astype(bool))
    if fusion_stats is not None:
        with open(os.path.join(output_path, "fusion_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(fusion_stats, f, indent=2, ensure_ascii=False)
    return mask_path


# ---------------------------------------------------------------------------
# Square-wave detection
# ---------------------------------------------------------------------------

def detect_square_wave_regions(
    signal: np.ndarray,
    noise_mask: np.ndarray,
    band_upper: np.ndarray,
    band_lower: np.ndarray,
    min_length: int = 5,
    shift_coverage_threshold: float = 0.95,
):
    time_steps = len(signal)
    square_wave_mask = np.zeros(time_steps, dtype=bool)
    square_wave_info = []

    noise_indices = np.where(noise_mask)[0]
    if len(noise_indices) == 0:
        return square_wave_mask, square_wave_info

    noise_regions = []
    start = noise_indices[0]
    for i in range(1, len(noise_indices)):
        if noise_indices[i] != noise_indices[i - 1] + 1:
            noise_regions.append((start, noise_indices[i - 1]))
            start = noise_indices[i]
    noise_regions.append((start, noise_indices[-1]))

    for region_start, region_end in noise_regions:
        if region_end - region_start + 1 < min_length:
            continue

        reg_sig = signal[region_start:region_end + 1]
        reg_up  = band_upper[region_start:region_end + 1]
        reg_lo  = band_lower[region_start:region_end + 1]

        outlier_mask = (reg_sig > reg_up) | (reg_sig < reg_lo)
        num_outliers = int(np.sum(outlier_mask))
        if num_outliers == 0:
            continue

        outlier_vals = reg_sig[outlier_mask]
        midline = (reg_up + reg_lo) / 2.0
        shift_up = float(np.median(midline[outlier_mask] - outlier_vals))
        shift_dn = -shift_up

        shifted_up = reg_sig + shift_up
        shifted_dn = reg_sig + shift_dn
        in_band_up = (shifted_up >= reg_lo) & (shifted_up <= reg_up)
        in_band_dn = (shifted_dn >= reg_lo) & (shifted_dn <= reg_up)
        cov_up = float(np.sum(in_band_up[outlier_mask]) / num_outliers)
        cov_dn = float(np.sum(in_band_dn[outlier_mask]) / num_outliers)

        best_cov   = max(cov_up, cov_dn)
        best_shift = shift_up if cov_up >= cov_dn else shift_dn
        in_band_best = in_band_up if cov_up >= cov_dn else in_band_dn

        if best_cov < shift_coverage_threshold:
            continue

        outlier_idx_in_region = np.where(outlier_mask)[0]
        covered_idx = outlier_idx_in_region[in_band_best[outlier_mask]]

        if len(covered_idx) > 0:
            actual_start = region_start + covered_idx[0]
            actual_end   = region_start + covered_idx[-1]
        else:
            actual_start, actual_end = region_start, region_end

        square_wave_mask[actual_start:actual_end + 1] = True
        square_wave_info.append({
            'start':           int(actual_start),
            'end':             int(actual_end),
            'original_start':  int(region_start),
            'original_end':    int(region_end),
            'length':          int(actual_end - actual_start + 1),
            'original_length': int(region_end - region_start + 1),
            'shift_amount':    float(best_shift),
            'shift_direction': 'up' if best_shift > 0 else 'down',
            'coverage':        float(best_cov),
            'num_outliers':    num_outliers,
            'num_covered_outliers': int(len(covered_idx)),
        })

    return square_wave_mask, square_wave_info


def refine_masks_with_square_wave_detection(input_signal: np.ndarray, noise_masks: np.ndarray, args: Config):
    if not args.enable_square_wave_detection:
        return noise_masks, {'enabled': False}

    try:
        band_upper = np.load(args.band_upper_path)
        band_lower = np.load(args.band_lower_path)
        band_mid   = np.load(args.band_midline_path)
    except Exception as e:
        return noise_masks, {'enabled': False, 'error': str(e)}

    if band_upper.shape[0] == 1 and band_upper.shape[2] == input_signal.shape[2]:
        total_t = band_upper.shape[1]
        expected_t = input_signal.shape[0] * input_signal.shape[1]
        if total_t != expected_t:
            return noise_masks, {'enabled': False, 'error': 'Band time-step mismatch.'}
        band_upper = band_upper.reshape(input_signal.shape)
        band_lower = band_lower.reshape(input_signal.shape)
        band_mid   = band_mid.reshape(input_signal.shape)
    elif band_upper.shape != input_signal.shape:
        return noise_masks, {'enabled': False, 'error': 'Band shape mismatch.'}

    refined = noise_masks.copy()
    batch_size, time_steps, num_features = input_signal.shape
    stats = {
        'enabled': True,
        'total_noise_regions': 0,
        'square_wave_regions_detected': 0,
        'masks_refined': 0,
        'channels_processed': 0,
    }

    for b in tqdm(range(batch_size), desc="square-wave detection"):
        for ch in range(num_features):
            ch_sig  = input_signal[b, :, ch]
            ch_mask = noise_masks[b, :, ch]
            if not np.any(ch_mask):
                continue

            ni = np.where(ch_mask)[0]
            n_regions = 1 + sum(ni[i] != ni[i - 1] + 1 for i in range(1, len(ni)))
            stats['total_noise_regions'] += n_regions

            sw_mask, sw_info = detect_square_wave_regions(
                ch_sig, ch_mask,
                band_upper[b, :, ch], band_lower[b, :, ch],
                min_length=args.square_wave_min_length,
                shift_coverage_threshold=args.square_wave_shift_coverage_threshold,
            )

            for info in sw_info:
                refined[b, info['original_start']:info['original_end'] + 1, ch] = False
                refined[b, info['start']:info['end'] + 1, ch] = True
                stats['masks_refined'] += 1

            if sw_info:
                stats['square_wave_regions_detected'] += len(sw_info)
                stats['channels_processed'] += 1

    stats['original_noise_points'] = int(np.sum(noise_masks))
    stats['refined_noise_points']  = int(np.sum(refined))
    stats['mask_change_ratio'] = float(
        (stats['refined_noise_points'] - stats['original_noise_points'])
        / max(stats['original_noise_points'], 1)
    )
    return refined, stats


# ---------------------------------------------------------------------------
# Iterative denoising
# ---------------------------------------------------------------------------

def iterative_denoising_pipeline(model: Model, input_signal: np.ndarray, args: Config):
    current_signal = input_signal.copy()
    iteration_history: List[Dict] = []
    convergence_info = {
        'converged': False,
        'convergence_iteration': -1,
        'final_change': 0.0,
        'iteration_changes': [],
    }

    orig_conf   = args.confidence_threshold
    orig_spike  = args.spike_threshold
    fixed_masks = None
    init_probs  = None
    init_cls    = None

    for it in range(args.denoise_iterations):
        curr_conf  = orig_conf  * (args.iteration_confidence_decay  ** it) if not args.use_fixed_noise_regions else orig_conf
        curr_spike = orig_spike * (args.iteration_spike_threshold_decay ** it)
        args.confidence_threshold = curr_conf
        args.spike_threshold      = curr_spike

        if it == 0 or not args.use_fixed_noise_regions:
            probs, masks, cls_out = stage1_noise_classification(model, current_signal, args)
            if it == 0 and args.use_fixed_noise_regions:
                fixed_masks = expand_noise_regions(masks, args.noise_region_expansion) if args.noise_region_expansion > 0 else masks.copy()
                init_probs, init_cls = probs.copy(), cls_out.copy()
        else:
            masks, probs, cls_out = fixed_masks, init_probs, init_cls

        denoised = stage2_denoising(model, current_signal, masks, args)

        if args.enable_spike_suppression:
            processed, spike_count = spike_suppression_post_process(
                denoised, masks,
                spike_threshold=curr_spike,
                window_size=args.spike_window_size,
                suppression_strength=args.spike_suppression_strength,
                min_spike_width=args.min_spike_width,
                max_spike_width=args.max_spike_width,
            )
        else:
            processed, spike_count = denoised, 0

        change = float(np.mean(np.abs(processed - current_signal))) if it > 0 else 0.0
        convergence_info['iteration_changes'].append(change)

        info = {
            'iteration':           it + 1,
            'confidence_threshold': curr_conf,
            'spike_threshold':      curr_spike,
            'noise_ratio':          float(np.mean(masks)),
            'spike_count':          spike_count,
            'signal_change':        change,
            'converged':            False,
            'used_fixed_regions':   args.use_fixed_noise_regions and it > 0,
        }

        if args.enable_early_stopping and it > 0 and change < args.convergence_threshold:
            info['converged'] = True
            convergence_info.update({'converged': True, 'convergence_iteration': it + 1, 'final_change': change})
            iteration_history.append(info)
            break

        iteration_history.append(info)
        current_signal = processed.copy()

    args.confidence_threshold = orig_conf
    args.spike_threshold      = orig_spike

    if not convergence_info['converged']:
        convergence_info['final_change'] = convergence_info['iteration_changes'][-1] if convergence_info['iteration_changes'] else 0.0

    return current_signal, iteration_history, convergence_info


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_denoised_result(result: np.ndarray, args: Config, iteration_history=None, convergence_info=None) -> None:
    os.makedirs(args.output_path, exist_ok=True)

    suffix = ""
    if args.enable_iterative_denoising and iteration_history:
        suffix = f"_iter{len(iteration_history)}"
        if convergence_info and convergence_info.get('converged'):
            suffix += "_converged"

    std_path = os.path.join(args.output_path, f"denoised_standardized_{args.required_time_steps}steps{suffix}.npy")
    np.save(std_path, result.astype(np.float32))

    if iteration_history:
        hist_path = os.path.join(args.output_path, f"iteration_history_{args.required_time_steps}steps.json")
        payload = {
            'iteration_history': _to_serializable(iteration_history),
            'convergence_info':  _to_serializable(convergence_info),
            'config': {
                'denoise_iterations':              int(args.denoise_iterations),
                'initial_confidence_threshold':    float(args.confidence_threshold),
                'iteration_confidence_decay':      float(args.iteration_confidence_decay),
                'iteration_spike_threshold_decay': float(args.iteration_spike_threshold_decay),
                'convergence_threshold':           float(args.convergence_threshold),
                'enable_early_stopping':           bool(args.enable_early_stopping),
                'use_fixed_noise_regions':         bool(args.use_fixed_noise_regions),
                'noise_region_expansion':          int(args.noise_region_expansion),
            },
        }
        with open(hist_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.enable_denormalization and args.save_denormalized_result:
        try:
            denorm_result, used_method = denormalize_data(
                result, args.trimmed_data_dir, args.dataset_type, args.normalization_method
            )
            denorm_path = os.path.join(
                args.output_path,
                f"denoised_denormalized_{used_method}_{args.required_time_steps}steps{suffix}.npy",
            )
            np.save(denorm_path, denorm_result.astype(np.float32))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = Config()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    input_signal = np.load(args.input_path)

    input_fixed, sample_indices = validate_and_fix_input(input_signal, args)
    model = load_model(args)

    iteration_history = convergence_info = None

    if args.enable_iterative_denoising and not args.stage1_only:
        final_result, iteration_history, convergence_info = iterative_denoising_pipeline(model, input_fixed, args)
        noise_probs, noise_masks, cls_outputs = stage1_noise_classification(model, input_fixed, args)
    else:
        noise_probs, noise_masks, cls_outputs = stage1_noise_classification(model, input_fixed, args)

        if not args.stage1_only:
            denoised = stage2_denoising(model, input_fixed, noise_masks, args)

            if args.enable_spike_suppression:
                final_result, _ = spike_suppression_post_process(
                    denoised, noise_masks,
                    spike_threshold=args.spike_threshold,
                    window_size=args.spike_window_size,
                    suppression_strength=args.spike_suppression_strength,
                    min_spike_width=args.min_spike_width,
                    max_spike_width=args.max_spike_width,
                )
            else:
                final_result = denoised
        else:
            final_result = None

    # Restore sample order
    if np.array_equal(sample_indices, np.arange(len(sample_indices))):
        ord_masks = noise_masks
        ord_probs = noise_probs
        ord_cls   = cls_outputs
        ord_result = final_result
    else:
        restore = np.argsort(sample_indices)
        ord_masks  = noise_masks[restore]
        ord_probs  = noise_probs[restore]
        ord_cls    = cls_outputs[restore]
        ord_result = final_result[restore] if final_result is not None else None

    # Time-frequency fusion
    fusion_stats = None
    if args.enable_timefreq_fusion:
        if getattr(args, 'use_timefreq_mask_file', True) and not os.path.exists(getattr(args, 'timefreq_mask_path', '')):
            raise FileNotFoundError(f"External time-frequency mask not found: {args.timefreq_mask_path}")

        tf_masks = None
        try:
            if os.path.exists(args.timefreq_mask_path):
                ext = np.load(args.timefreq_mask_path)
                ext = (ext > 0) if ext.dtype != np.bool_ else ext
                if ext.shape != ord_masks.shape:
                    raise ValueError(f"External mask shape mismatch: {ext.shape} vs {ord_masks.shape}")
                tf_masks = ext.astype(bool)
        except Exception:
            tf_masks = np.zeros_like(ord_masks, dtype=bool)

        ord_masks, fusion_stats = fuse_noise_masks(ord_masks, tf_masks)

        if args.enable_edge_pair_analysis:
            ord_masks, _ = apply_noise_edge_pair_analysis(
                input_fixed, ord_masks,
                min_run_remove=args.edge_morph_min_run_remove,
                max_gap_fill=args.edge_morph_max_gap_fill,
                min_run_keep=args.edge_morph_min_run_keep,
                linearity_threshold=args.linearity_threshold,
                inflection_threshold=args.inflection_threshold,
                slope_change_threshold=args.slope_change_threshold,
                area_time_ratio_threshold=args.area_time_ratio_threshold,
            )

    # Square-wave refinement
    if args.enable_square_wave_detection:
        ord_masks, sw_stats = refine_masks_with_square_wave_detection(input_fixed, ord_masks, args)
        if sw_stats.get('enabled'):
            os.makedirs(args.output_path, exist_ok=True)
            with open(os.path.join(args.output_path, "square_wave_detection_stats.json"), 'w', encoding='utf-8') as f:
                json.dump(_to_serializable(sw_stats), f, indent=2, ensure_ascii=False)

    mask_path = save_noise_masks(ord_masks, args.output_path, fusion_stats)

    if args.stage1_only:
        print(f"Stage 1 complete. Noise mask saved to: {mask_path}")
        return

    save_denoised_result(ord_result, args, iteration_history, convergence_info)

    suffix = ""
    if args.enable_iterative_denoising and iteration_history:
        suffix = f"_iter{len(iteration_history)}"
        if convergence_info and convergence_info.get('converged'):
            suffix += "_converged"
    result_file = f"denoised_standardized_{args.required_time_steps}steps{suffix}.npy"

    print(f"Noise mask  : {mask_path}")
    print(f"Denoised    : {os.path.join(args.output_path, result_file)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)