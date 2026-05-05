#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RestoreHighWaveRegions')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    high_wave_mask_path      = os.path.join("denoised_results_test1", "high_regions_mask.npy")
    fused_signal_path        = os.path.join("signal_fusion_results1", "fused_signal_direct_replace.npy")
    original_signal_txt_path = os.path.join("data", "extracted_five_channels_timeline.txt")

    band_data_dir   = os.path.join("vmd_outlierfree_ultrasmooth_results", "data")
    upper_band_path = os.path.join("vmd_outlierfree_ultrasmooth_results", "data", "upper_band.npy")
    lower_band_path = os.path.join("vmd_outlierfree_ultrasmooth_results", "data", "lower_band.npy")

    output_dir            = "signal_fusion_results1"
    output_npy_filename   = "fused_signal_with_restored_high_waves.npy"
    output_txt_filename   = "fused_signal_with_restored_high_waves.txt"
    output_stats_filename = "high_wave_restoration_stats.json"

    square_wave_mask_output_path   = os.path.join("denoised_results_test1", "square_wave_mask.npy")
    updated_noise_mask_output_path = os.path.join("denoised_results_test1", "noise_masks_updated_with_square_waves.npy")
    noise_mask_input_path          = os.path.join("denoised_results_test1", "noise_masks_updated_by_high_regions.npy")

    points_per_sample = 450
    num_channels      = 5

    enable_square_wave_detection   = True
    square_wave_coverage_threshold = 0.1
    square_wave_min_length         = 1
    smooth_transition_width        = 0
    band_margin_ratio              = 0.1

    enable_spike_suppression = False
    spike_threshold_factor   = 3
    spike_smooth_window      = 3

    channel_names = ['EX', 'EY', 'HX', 'HY', 'HZ']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_high_wave_mask(config: Config) -> np.ndarray:
    if not os.path.exists(config.high_wave_mask_path):
        raise FileNotFoundError(f"High-wave mask not found: {config.high_wave_mask_path}")
    mask = np.load(config.high_wave_mask_path).astype(bool)
    logger.info("High-wave mask loaded | shape=%s | ratio=%.2f%%", mask.shape, np.mean(mask) * 100)
    return mask


def load_fused_signal(config: Config) -> np.ndarray:
    if not os.path.exists(config.fused_signal_path):
        raise FileNotFoundError(f"Fused signal not found: {config.fused_signal_path}")
    sig = np.load(config.fused_signal_path)
    logger.info("Fused signal loaded | shape=%s", sig.shape)
    return sig


def load_original_signal(config: Config) -> np.ndarray:
    if not os.path.exists(config.original_signal_txt_path):
        raise FileNotFoundError(f"Original signal TXT not found: {config.original_signal_txt_path}")
    raw = np.loadtxt(config.original_signal_txt_path)
    if raw.ndim != 2 or raw.shape[1] != config.num_channels:
        raise ValueError(f"Expected {config.num_channels} columns, got shape {raw.shape}.")
    T = raw.shape[0]
    if T % config.points_per_sample != 0:
        raise ValueError(f"Total time steps {T} not divisible by points_per_sample={config.points_per_sample}.")
    sig = raw.reshape(T // config.points_per_sample, config.points_per_sample, config.num_channels)
    logger.info("Original signal loaded | shape=%s", sig.shape)
    return sig


def load_band_data(config: Config):
    for p in (config.upper_band_path, config.lower_band_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Band file not found: {p}")
    upper = np.load(config.upper_band_path)
    lower = np.load(config.lower_band_path)
    logger.info("Band data loaded | upper=%s lower=%s", upper.shape, lower.shape)
    return upper, lower


# ---------------------------------------------------------------------------
# Region utilities
# ---------------------------------------------------------------------------

def find_continuous_regions(mask_1d: np.ndarray):
    if not np.any(mask_1d):
        return []
    regions, in_region, start = [], False, 0
    for i, v in enumerate(mask_1d):
        if v and not in_region:
            start, in_region = i, True
        elif not v and in_region:
            regions.append((start, i - 1))
            in_region = False
    if in_region:
        regions.append((start, len(mask_1d) - 1))
    return regions


# ---------------------------------------------------------------------------
# Square-wave detection & restoration
# ---------------------------------------------------------------------------

def detect_square_wave(signal_region, upper_band_region, lower_band_region, config: Config):
    signal_baseline = np.median(signal_region)
    band_center     = (upper_band_region + lower_band_region) / 2
    shift           = signal_baseline - np.median(band_center)
    within_band     = (signal_region >= lower_band_region + shift) & (signal_region <= upper_band_region + shift)
    coverage        = float(np.mean(within_band))
    return coverage >= config.square_wave_coverage_threshold, coverage, shift


def suppress_spikes_in_region(signal_region: np.ndarray, config: Config) -> np.ndarray:
    if not config.enable_spike_suppression or len(signal_region) < config.spike_smooth_window:
        return signal_region
    out = signal_region.copy()
    hw  = config.spike_smooth_window // 2
    for i in range(hw, len(signal_region) - hw):
        local = np.concatenate([signal_region[i - hw:i], signal_region[i + 1:i + 1 + hw]])
        lm, ls = np.mean(local), np.std(local)
        if ls > 1e-6 and abs(signal_region[i] - lm) > config.spike_threshold_factor * ls:
            out[i] = lm
    return out


def shift_region_to_band(signal_region, upper_band_region, lower_band_region, config: Config):
    relative    = signal_region - np.mean(signal_region)
    band_center = (upper_band_region + lower_band_region) / 2
    band_width  = upper_band_region - lower_band_region
    amplitude   = np.max(relative) - np.min(relative)
    avg_bw      = np.mean(band_width)

    if amplitude > avg_bw * (1 - 2 * config.band_margin_ratio):
        relative = relative * (avg_bw * (1 - 2 * config.band_margin_ratio) / amplitude)

    shifted = relative + band_center
    margin  = band_width * config.band_margin_ratio
    for i in range(len(shifted)):
        shifted[i] = np.clip(shifted[i], lower_band_region[i] + margin[i], upper_band_region[i] - margin[i])

    return suppress_spikes_in_region(shifted, config)


def restore_high_wave_regions(fused_signal, original_signal, high_wave_mask, upper_band, lower_band, config: Config):
    if fused_signal.shape != original_signal.shape:
        raise ValueError(f"Shape mismatch: fused={fused_signal.shape} original={original_signal.shape}")
    if fused_signal.shape != high_wave_mask.shape:
        raise ValueError(f"Shape mismatch: fused={fused_signal.shape} mask={high_wave_mask.shape}")

    if upper_band.shape[0] == 1:
        N, T, C = fused_signal.shape
        if upper_band.shape[1] != N * T:
            raise ValueError(f"Band time steps {upper_band.shape[1]} != {N * T}.")
        upper_band = upper_band.reshape(N, T, C)
        lower_band = lower_band.reshape(N, T, C)

    restored = fused_signal.copy()
    restored[high_wave_mask] = original_signal[high_wave_mask]

    num_samples, time_steps, num_channels = fused_signal.shape
    square_wave_mask    = np.zeros_like(high_wave_mask, dtype=bool)
    square_wave_count   = 0
    square_wave_points  = 0
    square_wave_details = []

    if config.enable_square_wave_detection:
        for s in range(num_samples):
            for ch in range(num_channels):
                ch_mask = high_wave_mask[s, :, ch]
                if not np.any(ch_mask):
                    continue
                for rs, re in find_continuous_regions(ch_mask):
                    if re - rs + 1 < config.square_wave_min_length:
                        continue
                    sig_r = restored[s, rs:re + 1, ch]
                    up_r  = upper_band[s, rs:re + 1, ch]
                    lo_r  = lower_band[s, rs:re + 1, ch]
                    is_sq, cov, shift = detect_square_wave(sig_r, up_r, lo_r, config)
                    if not is_sq:
                        continue

                    restored[s, rs:re + 1, ch] = shift_region_to_band(sig_r, up_r, lo_r, config)

                    tw = config.smooth_transition_width
                    if tw > 0:
                        if rs > 0:
                            ls_  = max(0, rs - tw)
                            lw   = rs - ls_
                            w    = 0.5 * (1 - np.cos(np.pi * np.arange(lw) / lw))
                            for i, wi in enumerate(w):
                                idx = ls_ + i
                                restored[s, idx, ch] = (1 - wi) * original_signal[s, idx, ch] + wi * restored[s, rs, ch]
                        if re < time_steps - 1:
                            re_end = min(time_steps, re + 1 + tw)
                            rw     = re_end - re - 1
                            w      = 0.5 * (1 + np.cos(np.pi * np.arange(rw) / rw))
                            for i, wi in enumerate(w):
                                idx = re + 1 + i
                                if idx < time_steps:
                                    restored[s, idx, ch] = wi * restored[s, re, ch] + (1 - wi) * original_signal[s, idx, ch]

                    square_wave_mask[s, rs:re + 1, ch] = True
                    square_wave_count  += 1
                    square_wave_points += re - rs + 1
                    square_wave_details.append({
                        'sample': s, 'channel': ch,
                        'start': int(rs), 'end': int(re),
                        'length': int(re - rs + 1),
                        'coverage': float(cov), 'shift': float(shift),
                    })

    high_wave_pts = int(np.sum(high_wave_mask))
    total_pts     = fused_signal.size
    changed       = np.abs(restored - fused_signal)

    stats = {
        'total_points':                  total_pts,
        'high_wave_points':              high_wave_pts,
        'high_wave_ratio':               float(high_wave_pts / total_pts),
        'square_wave_detection_enabled': config.enable_square_wave_detection,
        'square_wave_count':             square_wave_count,
        'square_wave_points':            square_wave_points,
        'non_square_wave_points':        high_wave_pts - square_wave_points,
        'changed_points':                int(np.sum(changed > 1e-10)),
        'avg_change_in_high_wave':       float(np.mean(changed[high_wave_mask])) if high_wave_pts > 0 else 0.0,
        'max_change':                    float(np.max(changed)),
        'restored_signal_range':         [float(np.min(restored)),        float(np.max(restored))],
        'fused_signal_range':            [float(np.min(fused_signal)),    float(np.max(fused_signal))],
        'original_signal_range':         [float(np.min(original_signal)), float(np.max(original_signal))],
        'square_wave_details':           square_wave_details,
    }

    logger.info(
        "Restoration done | high_wave=%d (%.2f%%) | square_wave=%d | changed=%d",
        high_wave_pts, high_wave_pts / total_pts * 100,
        square_wave_points, stats['changed_points'],
    )
    return restored, stats, square_wave_mask


# ---------------------------------------------------------------------------
# Serialization & persistence
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, dict):        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [_to_serializable(v) for v in obj]
    return obj


def save_results(restored_signal: np.ndarray, stats: dict, config: Config):
    os.makedirs(config.output_dir, exist_ok=True)

    npy_path = os.path.join(config.output_dir, config.output_npy_filename)
    np.save(npy_path, restored_signal.astype(np.float32))

    txt_path = os.path.join(config.output_dir, config.output_txt_filename)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for sample in restored_signal:
            np.savetxt(f, sample, fmt="%.6f", delimiter=" ")

    stats_path = os.path.join(config.output_dir, config.output_stats_filename)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(stats), f, indent=2, ensure_ascii=False)

    logger.info("Results saved | npy=%s txt=%s stats=%s", npy_path, txt_path, stats_path)
    return npy_path, txt_path, stats_path


def save_square_wave_masks(square_wave_mask: np.ndarray, config: Config):
    os.makedirs(os.path.dirname(config.square_wave_mask_output_path), exist_ok=True)
    np.save(config.square_wave_mask_output_path, square_wave_mask.astype(bool))
    logger.info("Square-wave mask saved: %s | points=%d (%.2f%%)",
                config.square_wave_mask_output_path,
                int(np.sum(square_wave_mask)), float(np.mean(square_wave_mask)) * 100)

    if not os.path.exists(config.noise_mask_input_path):
        logger.warning("Noise mask not found (%s); skipping update.", config.noise_mask_input_path)
        return config.square_wave_mask_output_path, None

    orig_mask = np.load(config.noise_mask_input_path).astype(bool)
    if orig_mask.shape != square_wave_mask.shape:
        logger.error("Shape mismatch orig=%s sw=%s; skipping update.", orig_mask.shape, square_wave_mask.shape)
        return config.square_wave_mask_output_path, None

    num_samples, time_steps, num_channels = square_wave_mask.shape
    old_sw_mask = np.zeros_like(square_wave_mask, dtype=bool)

    for s in range(num_samples):
        for ch in range(num_channels):
            precise = square_wave_mask[s, :, ch]
            orig    = orig_mask[s, :, ch]
            if not np.any(precise):
                continue
            for rs, re in find_continuous_regions(precise):
                ss, se = max(0, rs - 20), min(time_steps - 1, re + 20)
                sub = orig[ss:se + 1]
                if not np.any(sub):
                    continue
                for lrs, lre in find_continuous_regions(sub):
                    gs, ge = ss + lrs, ss + lre
                    tmp = np.zeros(time_steps, dtype=bool)
                    tmp[gs:ge + 1] = True
                    if np.any(tmp & precise):
                        old_sw_mask[s, gs:ge + 1, ch] = True

    updated = orig_mask.copy()
    updated[old_sw_mask]      = False
    updated[square_wave_mask] = True
    np.save(config.updated_noise_mask_output_path, updated.astype(bool))

    logger.info(
        "Noise mask updated | before=%d (%.2f%%) after=%d (%.2f%%) | saved: %s",
        int(np.sum(orig_mask)), float(np.mean(orig_mask)) * 100,
        int(np.sum(updated)),   float(np.mean(updated))   * 100,
        config.updated_noise_mask_output_path,
    )
    return config.square_wave_mask_output_path, config.updated_noise_mask_output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = Config()

    try:
        high_wave_mask  = load_high_wave_mask(config)
        fused_signal    = load_fused_signal(config)
        original_signal = load_original_signal(config)
        upper_band, lower_band = load_band_data(config)

        restored, stats, square_wave_mask = restore_high_wave_regions(
            fused_signal, original_signal, high_wave_mask, upper_band, lower_band, config
        )

        npy_path, _, _     = save_results(restored, stats, config)
        sw_mask_path, upd  = save_square_wave_masks(square_wave_mask, config)

        logger.info("Pipeline complete | npy=%s | sw_mask=%s", npy_path, sw_mask_path)
        if upd:
            logger.info("Updated noise mask: %s", upd)

    except Exception as exc:
        logger.error("Pipeline failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise


if __name__ == '__main__':
    main()