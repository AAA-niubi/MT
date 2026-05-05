#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from vmdpy import VMD
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VMD_OutlierFree_UltraSmooth')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class VMDLowFreqConfig:
    input_signal_path = os.path.join('data', 'extracted_five_channels_timeline.txt')
    output_dir = './vmd_outlierfree_ultrasmooth_results'
    save_results = True
    save_figures = True
    enable_visualization = True

    num_channels = 5
    channel_names = ['EX', 'EY', 'HX', 'HY', 'HZ']
    time_steps = 67500
    samples_per_channel = 1

    enable_outlier_removal = True
    outlier_method = "MAD"
    mad_thresholds = [3, 3, 4, 2.0, 0.1]
    outlier_direction = "both"
    outlier_replace = "median"
    outlier_window = 450

    points_per_figure = 450
    overlap_points = 0
    figure_prefix = "outlierfree_ultrasmooth_window_"

    vmd_params = {
        'alpha': 10000,
        'tau':   0.0,
        'K':     1,
        'DC':    1,
        'init':  1,
        'tol':   1e-10,
    }

    enable_multi_smoothing = False
    first_smooth_window = 50
    second_smooth_sigma = 30
    max_trend_rate = 0.0005

    band_fixed_widths = [300, 650, 50000, 30000, 10]
    min_band_width = 0.01

    visualize_channels = [0, 1, 2, 3, 4]
    figsize = (20, 16)
    dpi = 150
    line_width = 0.6
    lowfreq_line_width = 2.5
    band_alpha = 0.3
    colors = {
        'original': '#1f77b4',
        'lowfreq':  '#c0392b',
        'band':     '#3498db',
        'outlier':  '#f39c12',
    }


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def remove_outliers(signal: np.ndarray, config: VMDLowFreqConfig, channel_idx: int):
    if not config.enable_outlier_removal:
        return signal, np.zeros_like(signal, dtype=bool)

    if not (0 <= channel_idx < len(config.mad_thresholds)):
        raise ValueError(f"Channel index {channel_idx} out of range for mad_thresholds (len={len(config.mad_thresholds)}).")
    if config.outlier_direction not in ("both", "upper", "lower"):
        raise ValueError(f"Invalid outlier_direction '{config.outlier_direction}'; expected 'both', 'upper', or 'lower'.")

    signal_clean = signal.copy()
    outlier_mask = np.zeros_like(signal, dtype=bool)
    window = config.outlier_window
    mad_thr = config.mad_thresholds[channel_idx]

    median_global = np.median(signal)
    mad = np.median(np.abs(signal - median_global))
    threshold = mad_thr * mad
    upper_bound = median_global + threshold
    lower_bound = median_global - threshold

    for i in range(len(signal)):
        s = max(0, i - window // 2)
        e = min(len(signal), i + window // 2 + 1)
        local = signal[s:e]

        if config.outlier_direction == "both":
            is_outlier = signal[i] > upper_bound or signal[i] < lower_bound
        elif config.outlier_direction == "upper":
            is_outlier = signal[i] > upper_bound
        else:
            is_outlier = signal[i] < lower_bound

        if is_outlier:
            outlier_mask[i] = True
            signal_clean[i] = np.median(local) if config.outlier_replace == "median" else np.mean(local)

    n_out = int(np.sum(outlier_mask))
    logger.info(
        "Channel %d outlier removal: %s direction, %d outliers (%.2f%%)",
        channel_idx, config.outlier_direction, n_out, n_out / len(signal) * 100,
    )
    return signal_clean, outlier_mask


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_preprocess_data(config: VMDLowFreqConfig):
    input_path = os.path.normpath(config.input_signal_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if len(config.mad_thresholds) != config.num_channels:
        raise ValueError(
            f"mad_thresholds length ({len(config.mad_thresholds)}) != num_channels ({config.num_channels})."
        )
    if len(config.band_fixed_widths) != config.num_channels:
        raise ValueError(
            f"band_fixed_widths length ({len(config.band_fixed_widths)}) != num_channels ({config.num_channels})."
        )

    original_data = np.loadtxt(input_path)
    if original_data.shape[1] != config.num_channels:
        raise ValueError(
            f"Expected {config.num_channels} columns, got {original_data.shape[1]}."
        )
    if original_data.shape[0] != config.time_steps:
        logger.warning("time_steps adjusted: config=%d → actual=%d", config.time_steps, original_data.shape[0])
        config.time_steps = original_data.shape[0]

    original_data = original_data.reshape(config.samples_per_channel, config.time_steps, config.num_channels)
    outlier_masks = np.zeros_like(original_data, dtype=bool)
    data_clean = np.zeros_like(original_data)

    for ch in range(config.num_channels):
        data_clean[0, :, ch], outlier_masks[0, :, ch] = remove_outliers(
            original_data[0, :, ch], config, channel_idx=ch
        )

    return original_data, data_clean, outlier_masks


# ---------------------------------------------------------------------------
# VMD & smoothing
# ---------------------------------------------------------------------------

def ultrasmooth_trend(trend: np.ndarray, config: VMDLowFreqConfig) -> np.ndarray:
    if not config.enable_multi_smoothing:
        return trend
    smoothed = uniform_filter1d(trend, size=config.first_smooth_window, mode='nearest')
    smoothed = gaussian_filter1d(smoothed, sigma=config.second_smooth_sigma, mode='nearest')
    diffs = np.clip(np.diff(smoothed), -config.max_trend_rate, config.max_trend_rate)
    clamped = np.empty_like(smoothed)
    clamped[0] = smoothed[0]
    for i in range(1, len(clamped)):
        clamped[i] = clamped[i - 1] + diffs[i - 1]
    return clamped


def vmd_extract_lowfreq(signal: np.ndarray, config: VMDLowFreqConfig) -> np.ndarray:
    u, _, _ = VMD(
        signal,
        alpha=config.vmd_params['alpha'],
        tau=config.vmd_params['tau'],
        K=config.vmd_params['K'],
        DC=config.vmd_params['DC'],
        init=config.vmd_params['init'],
        tol=config.vmd_params['tol'],
    )
    return ultrasmooth_trend(u[0, :], config)


def calculate_band(lowfreq: np.ndarray, config: VMDLowFreqConfig, channel_idx: int):
    width = max(config.band_fixed_widths[channel_idx], config.min_band_width)
    if width <= 0:
        raise ValueError(f"Band width for channel {channel_idx} must be positive, got {width}.")
    half = width / 2.0
    return lowfreq + half, lowfreq - half


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _split_windows(total_len: int, window_size: int, overlap: int):
    windows, start, idx = [], 0, 0
    while start < total_len:
        end = min(start + window_size, total_len)
        windows.append((idx, start, end))
        start = max(end - overlap, start + 1)
        idx += 1
    return windows


def visualize_results(original_data, data_clean, lowfreq_data, upper_bands, lower_bands, outlier_masks, config):
    fig_dir = os.path.join(config.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    windows = _split_windows(config.time_steps, config.points_per_figure, config.overlap_points)
    plt.rcParams["axes.unicode_minus"] = False

    for win_idx, start, end in tqdm(windows, desc="rendering figures"):
        t = np.arange(start, end)
        fig, axes = plt.subplots(len(config.visualize_channels), 1, figsize=config.figsize, sharex=True)
        if len(config.visualize_channels) == 1:
            axes = [axes]

        for ax_idx, ch in enumerate(config.visualize_channels):
            ax = axes[ax_idx]
            bw = config.band_fixed_widths[ch]

            ax.plot(t, original_data[0, start:end, ch],
                    color=config.colors['original'], linewidth=config.line_width, label="Original")
            ax.plot(t, data_clean[0, start:end, ch],
                    color='#2ecc71', linewidth=0.8, label="Outlier-Free")
            ax.fill_between(t, lower_bands[0, start:end, ch], upper_bands[0, start:end, ch],
                            color=config.colors['band'], alpha=config.band_alpha, label=f"Band (±{bw/2})")
            ax.plot(t, lowfreq_data[0, start:end, ch],
                    color=config.colors['lowfreq'], linewidth=config.lowfreq_line_width, label="Ultra-Smooth Trend")

            out = outlier_masks[0, start:end, ch]
            if np.any(out):
                ax.scatter(t[out], original_data[0, start:end, ch][out],
                           color=config.colors['outlier'], s=10, alpha=0.8, label="Outliers")

            ax.set_title(
                f"{config.channel_names[ch]} | window {win_idx + 1} | t={start}-{end} | "
                f"MAD thr={config.mad_thresholds[ch]} | band width={bw}"
            )
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.savefig(
            os.path.join(fig_dir, f"{config.figure_prefix}{win_idx + 1:03d}_{start}-{end}.png"),
            dpi=config.dpi,
        )
        plt.close()

    logger.info("Figures saved to: %s", fig_dir)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(original, data_clean, lowfreq, upper, lower, outlier_masks, config: VMDLowFreqConfig):
    if not config.save_results:
        return
    data_dir = os.path.join(config.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "original_data.npy"),            original)
    np.save(os.path.join(data_dir, "outlierfree_data.npy"),         data_clean)
    np.save(os.path.join(data_dir, "ultrasmooth_lowfreq_data.npy"), lowfreq)
    np.save(os.path.join(data_dir, "upper_band.npy"),               upper)
    np.save(os.path.join(data_dir, "lower_band.npy"),               lower)
    np.save(os.path.join(data_dir, "outlier_masks.npy"),            outlier_masks)

    with open(os.path.join(data_dir, "config.csv"), 'w') as f:
        f.write("channel_name,mad_threshold,detect_direction,replace_method,outlier_window,band_fixed_width,min_band_width\n")
        for name, thr, bw in zip(config.channel_names, config.mad_thresholds, config.band_fixed_widths):
            f.write(f"{name},{thr},{config.outlier_direction},{config.outlier_replace},"
                    f"{config.outlier_window},{bw},{config.min_band_width}\n")

    with open(os.path.join(data_dir, "channel_names.txt"), 'w') as f:
        f.writelines(name + "\n" for name in config.channel_names)

    for ch in range(config.num_channels):
        ch_data = np.column_stack([
            original[0, :, ch],
            data_clean[0, :, ch],
            lowfreq[0, :, ch],
            upper[0, :, ch],
            lower[0, :, ch],
            outlier_masks[0, :, ch].astype(int),
        ])
        np.savetxt(
            os.path.join(data_dir, f"channel_{config.channel_names[ch]}.txt"),
            ch_data,
            header="original,outlierfree,ultrasmooth_lowfreq,upper_band,lower_band,outlier_mark",
            comments="",
            fmt="%.6f",
        )

    logger.info("Results saved to: %s", data_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = VMDLowFreqConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    try:
        original_data, data_clean, outlier_masks = load_and_preprocess_data(config)

        lowfreq_data = np.zeros_like(original_data)
        upper_bands  = np.zeros_like(original_data)
        lower_bands  = np.zeros_like(original_data)

        for ch in tqdm(range(config.num_channels), desc="processing channels"):
            lowfreq = vmd_extract_lowfreq(data_clean[0, :, ch], config)
            upper, lower = calculate_band(lowfreq, config, channel_idx=ch)
            lowfreq_data[0, :, ch] = lowfreq
            upper_bands[0, :, ch]  = upper
            lower_bands[0, :, ch]  = lower
            logger.info(
                "Channel %s done | outlier ratio=%.2f%% | band width=%s",
                config.channel_names[ch],
                np.mean(outlier_masks[0, :, ch]) * 100,
                config.band_fixed_widths[ch],
            )

        if config.save_figures and config.enable_visualization:
            visualize_results(original_data, data_clean, lowfreq_data, upper_bands, lower_bands, outlier_masks, config)

        save_results(original_data, data_clean, lowfreq_data, upper_bands, lower_bands, outlier_masks, config)

        logger.info("Pipeline complete. Output: %s", os.path.abspath(config.output_dir))

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()