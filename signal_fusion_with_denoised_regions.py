#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import json
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SignalFusion')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class SignalFusionConfig:
    original_signal_txt_path = os.path.join("data", "extracted_five_channels_timeline.txt")
    denoised_signal_path     = os.path.join("denoised_results_test1", "denoised_standardized_450steps.npy")

    original_means_path   = os.path.join("DATA", "sample_QH", "self-supervused_data", "test", "test_means_TS5.npy")
    original_stds_path    = os.path.join("DATA", "sample_QH", "self-supervused_data", "test", "test_stds_TS5.npy")
    original_indices_path = os.path.join("DATA", "sample_QH", "self-supervused_data", "test", "test_indices_TS5.npy")

    noise_statistics_path = os.path.join("noise_regions_inference_test1", "noise_statistics.json")
    noise_masks_path      = os.path.join("denoised_results_test1", "noise_masks_updated_with_square_waves.npy")

    output_dir              = "./signal_fusion_results1"
    save_intermediate_results = False
    enable_visualization    = False
    save_simple_txt         = True

    confidence_threshold = 0.8
    fusion_method        = "direct_replace"
    smooth_window_size   = 10
    blend_weight         = 0.8

    num_samples_to_visualize        = 3
    channels_to_visualize           = [0, 1, 2, 3, 4]
    channel_names                   = ['EX', 'EY', 'HX', 'HY', 'HZ']
    figsize                         = (20, 12)
    enable_noise_region_visualization = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_original_txt(config: SignalFusionConfig) -> np.ndarray:
    if not os.path.exists(config.original_signal_txt_path):
        raise FileNotFoundError(f"Original TXT not found: {config.original_signal_txt_path}")
    raw = np.loadtxt(config.original_signal_txt_path)
    expected = 440 * 450
    if raw.shape[0] != expected:
        raise ValueError(f"Expected {expected} time steps, got {raw.shape[0]}.")
    sig = raw.reshape(440, 450, 5)
    logger.info("Original signal loaded | shape=%s | range=[%.4f, %.4f]", sig.shape, np.min(sig), np.max(sig))
    return sig


def load_denoised_signal_data(config: SignalFusionConfig) -> np.ndarray:
    if not os.path.exists(config.denoised_signal_path):
        raise FileNotFoundError(f"Denoised signal not found: {config.denoised_signal_path}")
    sig = np.load(config.denoised_signal_path)
    if sig.shape != (440, 450, 5):
        raise ValueError(f"Expected shape (440, 450, 5), got {sig.shape}.")
    logger.info("Denoised signal loaded | shape=%s", sig.shape)
    return sig


def load_denormalization_params(config: SignalFusionConfig):
    for p in (config.original_means_path, config.original_stds_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Denorm parameter file not found: {p}")
    means   = np.load(config.original_means_path)
    stds    = np.load(config.original_stds_path)
    indices = np.load(config.original_indices_path) if os.path.exists(config.original_indices_path) else None
    logger.info("Denorm params loaded | means=%s stds=%s", means.shape, stds.shape)
    return means, stds, indices


def load_noise_region_info(config: SignalFusionConfig):
    if not os.path.exists(config.noise_statistics_path):
        logger.warning("Noise statistics file not found: %s", config.noise_statistics_path)
        return None
    with open(config.noise_statistics_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    logger.info("Noise statistics loaded | overall ratio=%.1f%%",
                stats['overall']['overall_noise_ratio'] * 100)
    return stats


def load_model_noise_masks(original_signal: np.ndarray, config: SignalFusionConfig) -> np.ndarray:
    if not os.path.exists(config.noise_masks_path):
        raise FileNotFoundError(f"Noise mask not found: {config.noise_masks_path}")
    masks = np.load(config.noise_masks_path)
    if masks.shape != original_signal.shape:
        raise ValueError(f"Mask shape {masks.shape} != signal shape {original_signal.shape}.")
    logger.info("Noise masks loaded | noise ratio=%.1f%%", np.mean(masks) * 100)
    return masks.astype(bool)


# ---------------------------------------------------------------------------
# Denormalization & fusion
# ---------------------------------------------------------------------------

def denormalize_signals(denoised_signal: np.ndarray, means: np.ndarray, stds: np.ndarray,
                        config: SignalFusionConfig) -> np.ndarray:
    if means is None or stds is None:
        raise ValueError("Denorm parameters (means/stds) must not be None.")
    N, T, C = denoised_signal.shape
    if means.ndim == 2:
        means = means[:, np.newaxis, :]
        stds  = stds[:, np.newaxis, :]
    if means.shape != (N, 1, C) or stds.shape != (N, 1, C):
        raise ValueError(
            f"Expected means/stds shape ({N}, 1, {C}), got means={means.shape} stds={stds.shape}."
        )
    result = denoised_signal * stds + means
    logger.info("Denormalization done | range=[%.4f, %.4f]", np.min(result), np.max(result))
    return result


def fuse_signals(original_signal: np.ndarray, denoised_denorm: np.ndarray,
                 noise_masks: np.ndarray, config: SignalFusionConfig) -> np.ndarray:
    fused = original_signal.copy()
    fused[noise_masks] = denoised_denorm[noise_masks]
    logger.info(
        "Fusion done | noise ratio=%.1f%% | replaced=%d | avg change=%.4f | range=[%.4f, %.4f]",
        np.mean(noise_masks) * 100, int(np.sum(noise_masks)),
        float(np.mean(np.abs(fused - original_signal))),
        np.min(fused), np.max(fused),
    )
    return fused


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_fusion_results(fused_signal, original_signal, denoised_denorm, noise_masks, config: SignalFusionConfig):
    os.makedirs(config.output_dir, exist_ok=True)

    np.save(os.path.join(config.output_dir, f"fused_signal_{config.fusion_method}.npy"),
            fused_signal.astype(np.float32))
    np.save(os.path.join(config.output_dir, "original_signal_reshaped.npy"),
            original_signal.astype(np.float32))
    np.save(os.path.join(config.output_dir, "denoised_signal_denorm.npy"),
            denoised_denorm.astype(np.float32))

    if config.save_simple_txt:
        for name, arr in [("fused_signal_simple.txt", fused_signal),
                           ("original_signal_reshaped.txt", original_signal)]:
            with open(os.path.join(config.output_dir, name), 'w', encoding='utf-8') as f:
                for sample in arr:
                    np.savetxt(f, sample, fmt="%.6f", delimiter=" ")

    stats = {
        "fusion_method":          config.fusion_method,
        "denormalization_source": "user_specified_means_stds",
        "noise_ratio":            float(np.mean(noise_masks)),
        "signal_ranges": {
            "original":       [float(np.min(original_signal)),  float(np.max(original_signal))],
            "denoised_denorm":[float(np.min(denoised_denorm)),  float(np.max(denoised_denorm))],
            "fused":          [float(np.min(fused_signal)),      float(np.max(fused_signal))],
        },
    }
    with open(os.path.join(config.output_dir, "fusion_statistics.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", config.output_dir)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_initial_noise_regions(original_signal: np.ndarray, noise_masks: np.ndarray,
                                    config: SignalFusionConfig) -> None:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    vis_dir = os.path.join(config.output_dir, "initial_noise_regions_visualization")
    os.makedirs(vis_dir, exist_ok=True)

    batch_size, time_steps, num_features = original_signal.shape
    samples_per_figure = 10

    for ch in config.channels_to_visualize:
        ch_name = config.channel_names[ch]
        num_figs = int(np.ceil(batch_size / samples_per_figure))

        for fi in range(num_figs):
            s0 = fi * samples_per_figure
            s1 = min(s0 + samples_per_figure, batch_size)
            n  = s1 - s0

            cont_sig  = original_signal[s0:s1, :, ch].ravel()
            cont_mask = noise_masks[s0:s1, :, ch].ravel().astype(bool)
            t_ax      = np.arange(len(cont_sig))

            fig, axes = plt.subplots(2, 1, figsize=(30, 10), sharex=True)

            axes[0].plot(t_ax, cont_sig, 'b-', linewidth=0.6, alpha=0.8, label='Original Signal')
            noise_idx = np.where(cont_mask)[0]
            if len(noise_idx) > 0:
                regions, start_i = [], noise_idx[0]
                for k in range(1, len(noise_idx)):
                    if noise_idx[k] != noise_idx[k - 1] + 1:
                        regions.append((start_i, noise_idx[k - 1]))
                        start_i = noise_idx[k]
                regions.append((start_i, noise_idx[-1]))
                for ri, (rs, re) in enumerate(regions):
                    axes[0].axvspan(rs, re, alpha=0.3, color='red',
                                    label='Noise Region' if ri == 0 else '')

            for i in range(1, n):
                bp = i * time_steps
                axes[0].axvline(bp, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                                label='Sample Boundary' if i == 1 else '')
                axes[1].axvline(bp, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

            axes[0].set_ylabel(f'{ch_name}\nAmplitude', fontsize=12)
            axes[0].set_title(f'Channel {ch_name} — Noise Regions (samples {s0}–{s1 - 1})',
                              fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc='upper right', fontsize=10)

            axes[1].fill_between(t_ax, 0, cont_mask.astype(int), color='red', alpha=0.6, label='Noise Mask')
            axes[1].set_ylabel('Noise Mask', fontsize=12)
            axes[1].set_xlabel('Continuous Time Steps', fontsize=12)
            axes[1].set_ylim(-0.1, 1.1)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='upper right', fontsize=10)

            info = (f"Samples {s0}–{s1 - 1} ({n} samples)\n"
                    f"Noise: {int(np.sum(cont_mask))}/{len(cont_mask)} ({np.mean(cont_mask):.2%})")
            axes[1].text(0.02, 0.98, info, transform=axes[1].transAxes, fontsize=10,
                         va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir,
                                     f'noise_regions_{ch_name}_s{s0}-{s1 - 1}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    for ch in config.channels_to_visualize:
        ch_name = config.channel_names[ch]
        fig, ax = plt.subplots(figsize=(20, 12))
        im = ax.imshow(noise_masks[:, :, ch].astype(float), aspect='auto',
                       cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Sample Index', fontsize=12)
        ax.set_title(f'Channel {ch_name} — Noise Distribution Heatmap', fontsize=14, fontweight='bold')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Noise Mask (0=Clean, 1=Noise)', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'noise_heatmap_{ch_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ch_ratios = [float(np.mean(noise_masks[:, :, c])) for c in range(num_features)]
    axes[0, 0].bar(config.channel_names, ch_ratios, color='coral', alpha=0.7)
    axes[0, 0].set_ylabel('Noise Ratio', fontsize=11)
    axes[0, 0].set_title('Noise Ratio by Channel', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, r in enumerate(ch_ratios):
        axes[0, 0].text(i, r, f'{r:.2%}', ha='center', va='bottom', fontsize=9)

    sample_ratios = np.mean(noise_masks, axis=(1, 2))
    axes[0, 1].plot(sample_ratios, 'o-', color='steelblue', markersize=3, linewidth=0.8, alpha=0.7)
    axes[0, 1].axhline(np.mean(sample_ratios), color='red', linestyle='--', linewidth=1.5,
                       label=f'Mean: {np.mean(sample_ratios):.2%}')
    axes[0, 1].set_xlabel('Sample Index', fontsize=11)
    axes[0, 1].set_ylabel('Noise Ratio', fontsize=11)
    axes[0, 1].set_title('Noise Ratio by Sample', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    all_lengths = []
    for s in range(batch_size):
        for c in range(num_features):
            idx = np.where(noise_masks[s, :, c])[0]
            if len(idx) == 0:
                continue
            start_i = idx[0]
            for k in range(1, len(idx)):
                if idx[k] != idx[k - 1] + 1:
                    all_lengths.append(int(idx[k - 1] - start_i + 1))
                    start_i = idx[k]
            all_lengths.append(int(idx[-1] - start_i + 1))

    if all_lengths:
        axes[1, 0].hist(all_lengths, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(all_lengths), color='red', linestyle='--', linewidth=1.5,
                           label=f'Mean: {np.mean(all_lengths):.1f}')
        axes[1, 0].set_xlabel('Region Length (time steps)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Noise Region Length Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No noise regions found', ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontsize=12)

    axes[1, 1].axis('off')
    total_pts = batch_size * time_steps * num_features
    txt = ("Overall Noise Statistics\n" + "=" * 36 + "\n\n"
           f"Samples       : {batch_size}\n"
           f"Time steps    : {time_steps}\n"
           f"Channels      : {num_features}\n"
           f"Total points  : {total_pts:,}\n\n"
           f"Noise points  : {int(np.sum(noise_masks)):,}\n"
           f"Noise ratio   : {np.mean(noise_masks):.2%}\n")
    if all_lengths:
        txt += (f"\nRegions       : {len(all_lengths)}\n"
                f"Avg length    : {np.mean(all_lengths):.1f} steps\n"
                f"Min / Max     : {np.min(all_lengths)} / {np.max(all_lengths)} steps\n")
    axes[1, 1].text(0.1, 0.9, txt, transform=axes[1, 1].transAxes, fontsize=11, va='top',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'noise_statistics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    logger.info("Noise region visualization saved to: %s", vis_dir)


def visualize_fusion_results(original_signal, denoised_denorm, fused_signal, noise_masks,
                              config: SignalFusionConfig) -> None:
    if not config.enable_visualization:
        return
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    vis_dir = os.path.join(config.output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    n_samples = min(config.num_samples_to_visualize, original_signal.shape[0])
    t_ax = np.arange(original_signal.shape[1])

    for si in range(n_samples):
        for ch in config.channels_to_visualize:
            ch_name = config.channel_names[ch]
            fig, axes = plt.subplots(3, 1, figsize=config.figsize, sharex=True)

            axes[0].plot(t_ax, original_signal[si, :, ch], 'b-', linewidth=0.8, alpha=0.8, label='Original')
            axes[0].set_ylabel(f'{ch_name}\nOriginal', fontsize=11)
            axes[0].set_title(f'Sample {si + 1} — Channel {ch_name}', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(t_ax, denoised_denorm[si, :, ch], 'g-', linewidth=0.8, alpha=0.8,
                         label='Denoised (denorm)')
            axes[1].set_ylabel(f'{ch_name}\nDenoised', fontsize=11)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            axes[2].plot(t_ax, original_signal[si, :, ch], 'b--', linewidth=0.6, alpha=0.6, label='Original')
            axes[2].plot(t_ax, fused_signal[si, :, ch], 'r-', linewidth=0.8, alpha=0.9, label='Fused')
            nr = np.where(noise_masks[si, :, ch])[0]
            if len(nr) > 0:
                axes[2].axvspan(nr[0], nr[-1], alpha=0.2, color='yellow', label='Noise Regions')
            axes[2].set_ylabel(f'{ch_name}\nFused', fontsize=11)
            axes[2].set_xlabel('Time Steps', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'sample{si + 1}_{ch_name}.png'),
                        dpi=200, bbox_inches='tight')
            plt.close()

    logger.info("Fusion visualization saved to: %s", vis_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = SignalFusionConfig()

    try:
        original_signal  = load_original_txt(config)
        denoised_signal  = load_denoised_signal_data(config)
        means, stds, _   = load_denormalization_params(config)

        if original_signal.shape != denoised_signal.shape:
            raise ValueError(
                f"Shape mismatch: original={original_signal.shape} denoised={denoised_signal.shape}"
            )

        noise_masks = load_model_noise_masks(original_signal, config)

        if config.enable_noise_region_visualization:
            visualize_initial_noise_regions(original_signal, noise_masks, config)

        denoised_denorm = denormalize_signals(denoised_signal, means, stds, config)
        fused_signal    = fuse_signals(original_signal, denoised_denorm, noise_masks, config)

        save_fusion_results(fused_signal, original_signal, denoised_denorm, noise_masks, config)
        visualize_fusion_results(original_signal, denoised_denorm, fused_signal, noise_masks, config)

        logger.info("Pipeline complete. Output: %s", config.output_dir)

    except Exception as exc:
        logger.error("Pipeline failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()