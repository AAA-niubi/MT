#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


mpl.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif'],
    'axes.unicode_minus': False,
    'svg.fonttype': 'none',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'image.interpolation': 'none',
    'image.resample': False,
    'agg.path.chunksize': 10000,
    'backend': 'Agg',
})


try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False
    try:
        from scipy import signal as sp_signal  # type: ignore
    except Exception as e:
        raise ImportError("Either pywt or scipy is required for CWT computation.") from e


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def load_signal_txt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    data = np.loadtxt(path)
    return data.astype(float) if data.ndim == 1 else data[:, 0].astype(float)


def load_multichannel_matrix(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    arr = np.load(path) if ext == '.npy' else np.loadtxt(path, delimiter=',' if ext == '.csv' else None)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"Expected 2-D matrix with at least 5 columns, got shape {arr.shape}")
    return arr[:, :5].astype(float)


# ---------------------------------------------------------------------------
# CWT & detection
# ---------------------------------------------------------------------------

def compute_cwt_energy(
    signal: np.ndarray,
    fs: float,
    min_freq: float,
    max_freq: float,
    n_freqs: int,
    fc: float = 0.802,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = (signal - np.mean(signal)) / np.std(signal) if np.std(signal) != 0 else signal
    freq_range = np.linspace(min_freq, max_freq, n_freqs)
    scales = (fc * fs) / freq_range
    scales = scales[scales > 0]
    if scales.size == 0:
        raise ValueError("Invalid frequency range: no valid scales generated.")

    if _HAS_PYWT:
        cwt_result, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1.0 / fs)
    else:
        cwt_result = sp_signal.cwt(sig, lambda M, s: sp_signal.morlet2(M, s, w=5.0), scales)

    frequencies = (fc * fs) / scales
    order = np.argsort(frequencies)
    energy_db = 10 * np.log10(np.abs(cwt_result[order]) + np.finfo(float).eps)
    t = np.arange(len(signal)) / fs
    return energy_db, frequencies[order], t


def detect_bright_time_mask(
    energy_db: np.ndarray,
    frequencies: np.ndarray,
    threshold_db: Optional[float] = None,
    th_percentile: float = 95.0,
    min_band_hz: float = 3.0,
) -> np.ndarray:
    thr = (
        threshold_db
        if (threshold_db is not None and np.isfinite(threshold_db))
        else np.percentile(energy_db, th_percentile)
    )
    mask = energy_db >= thr
    freq_step = np.median(np.diff(frequencies)) if frequencies.size > 1 else np.inf
    min_bins = max(1, int(math.ceil(min_band_hz / freq_step))) if freq_step > 0 else 1

    bright_time = np.zeros(mask.shape[1], dtype=bool)
    for j in range(mask.shape[1]):
        run = 0
        for v in mask[:, j]:
            run = run + 1 if v else 0
            if run >= min_bins:
                bright_time[j] = True
                break
    return bright_time


def merge_time_mask_to_intervals(
    bright_time: np.ndarray,
    t: np.ndarray,
    min_duration_s: float = 0.066,
    join_gap_s: float = 0.5,
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    in_seg, start_idx = False, 0
    for i, v in enumerate(bright_time):
        if v and not in_seg:
            in_seg, start_idx = True, i
        elif not v and in_seg:
            intervals.append((t[start_idx], t[i - 1]))
            in_seg = False
    if in_seg:
        intervals.append((t[start_idx], t[-1]))

    intervals = [(s, e) for s, e in intervals if (e - s) >= min_duration_s]
    if not intervals:
        return []

    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe <= join_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Index conversion
# ---------------------------------------------------------------------------

def seconds_to_sample_indices(
    seconds: float,
    sampling_rate: float = 15.0,
    points_per_sample: int = 450,
) -> Tuple[int, int, int]:
    total_point_idx = int(seconds * sampling_rate)
    sample_idx = total_point_idx // points_per_sample
    point_idx = total_point_idx % points_per_sample
    return sample_idx, point_idx, total_point_idx


def convert_intervals_to_indices(
    intervals: List[Dict[str, float]],
    sampling_rate: float = 15.0,
    points_per_sample: int = 450,
) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for it in intervals:
        s, e = float(it['start']), float(it['end'])
        ss, sp, st = seconds_to_sample_indices(s, sampling_rate, points_per_sample)
        es, ep, et = seconds_to_sample_indices(e, sampling_rate, points_per_sample)
        out.append({
            'original_start_seconds': s,
            'original_end_seconds': e,
            'original_duration_seconds': float(it.get('duration', e - s)),
            'start_sample_idx': ss,
            'start_point_idx': sp,
            'start_total_point_idx': st,
            'end_sample_idx': es,
            'end_point_idx': ep,
            'end_total_point_idx': et,
            'duration_points': et - st,
            'spans_multiple_samples': ss != es,
        })
    return out


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_masks_from_indices(
    per_channel_indices: Dict[str, List[Dict[str, int]]],
    channel_order: List[str],
    points_per_sample: int = 450,
) -> Tuple[np.ndarray, Dict]:
    max_end_total = 0
    for items in per_channel_indices.values():
        for it in items:
            max_end_total = max(max_end_total, int(it['end_total_point_idx']))
    num_samples = (max_end_total // points_per_sample) + 1 if max_end_total > 0 else 1
    num_channels = len(channel_order)
    masks = np.zeros((num_samples, points_per_sample, num_channels), dtype=bool)

    stats: Dict = {'total_intervals': 0, 'total_noise_points': 0, 'channels': {}}
    name_to_idx = {name: i for i, name in enumerate(channel_order)}

    for ch_name, items in per_channel_indices.items():
        if ch_name not in name_to_idx:
            continue
        ch_idx = name_to_idx[ch_name]
        ch_points = 0
        for it in items:
            ss, sp = int(it['start_sample_idx']), int(it['start_point_idx'])
            es, ep = int(it['end_sample_idx']), int(it['end_point_idx'])
            if bool(it['spans_multiple_samples']):
                if ss < num_samples:
                    masks[ss, sp:, ch_idx] = True
                    ch_points += points_per_sample - sp
                for sidx in range(ss + 1, es):
                    if sidx < num_samples:
                        masks[sidx, :, ch_idx] = True
                        ch_points += points_per_sample
                if es < num_samples:
                    masks[es, :ep + 1, ch_idx] = True
                    ch_points += ep + 1
            else:
                if ss < num_samples:
                    masks[ss, sp:ep + 1, ch_idx] = True
                    ch_points += ep - sp + 1

        ch_total = num_samples * points_per_sample
        stats['channels'][ch_name] = {
            'intervals': len(items),
            'noise_points': int(ch_points),
            'total_points': int(ch_total),
            'noise_ratio': float(ch_points / ch_total) if ch_total > 0 else 0.0,
        }
        stats['total_intervals'] += len(items)
        stats['total_noise_points'] += int(ch_points)

    total_points = num_samples * points_per_sample * num_channels
    stats['overall'] = {
        'total_points': int(total_points),
        'total_noise_points': int(stats['total_noise_points']),
        'overall_noise_ratio': float(stats['total_noise_points'] / total_points) if total_points > 0 else 0.0,
    }
    return masks, stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_CWT_CMAP = LinearSegmentedColormap.from_list(
    'cwt_custom', ['#ADC4D4', '#b3b3da', '#fbe2d3', '#bbe6eb'], N=256
)


def visualize_bright_bands_chunked(
    data_matrix: np.ndarray,
    channels: List[str],
    per_channel_intervals: Dict[str, List[Dict[str, float]]],
    save_dir: str,
    prefix: str = "bright_bands",
    dpi: int = 300,
    show: bool = False,
    chunk_size: int = 4500,
    fs: float = 15.0,
    min_freq: float = 0.1,
    max_freq: float = 7.5,
    n_freqs: int = 200,
    fc: float = 0.802,
) -> None:
    N_total = data_matrix.shape[0]
    num_chunks = (N_total + chunk_size - 1) // chunk_size

    freq_range = np.linspace(min_freq, max_freq, n_freqs)
    scales = (fc * fs) / freq_range
    scales = scales[scales > 0]

    for ch_idx, ch_name in enumerate(channels):
        signal = data_matrix[:, ch_idx].astype(float)
        intervals = per_channel_intervals.get(ch_name, [])

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, N_total)
            seg = signal[start:end]
            if seg.size < 2:
                continue

            t_seg = np.arange(start, end) / fs
            t_start, t_end = float(t_seg[0]), float(t_seg[-1])

            chunk_intervals = [
                {
                    'start': max(iv['start'], t_start),
                    'end': min(iv['end'], t_end),
                    'duration': iv['duration'],
                }
                for iv in intervals
                if iv['end'] >= t_start and iv['start'] <= t_end
            ]

            fig = plt.figure(
                f"{ch_name}_chunk{chunk_idx + 1}",
                figsize=(12, 8),
                constrained_layout=True,
                dpi=dpi,
            )
            gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5], width_ratios=[1, 0.05], hspace=0.1)
            ax_time = fig.add_subplot(gs[0, 0])
            ax_cwt = fig.add_subplot(gs[1, 0], sharex=ax_time)
            cbar_ax = fig.add_subplot(gs[1, 1])

            ax_time.plot(t_seg, seg, "b-", linewidth=1.0, alpha=0.7)
            ax_time.set_xlim(t_start, t_end)
            ax_time.margins(x=0)
            ax_time.set_ylabel("Amplitude")

            ymin, ymax = seg.min(), seg.max()
            y_range = ymax - ymin if ymax > ymin else 1.0
            plot_ymin = ymin - 0.05 * y_range
            plot_ymax = ymax + 0.05 * y_range

            for iv in chunk_intervals:
                ax_time.add_patch(Rectangle(
                    (iv['start'], plot_ymin),
                    iv['end'] - iv['start'],
                    plot_ymax - plot_ymin,
                    facecolor='red', alpha=0.2, edgecolor='red', linewidth=1.5,
                ))

            title = f"{ch_name} - chunk {chunk_idx + 1}"
            if chunk_intervals:
                title += f" ({len(chunk_intervals)} bright band(s))"
            ax_time.set_title(title)
            ax_time.set_ylim(plot_ymin, plot_ymax)
            plt.setp(ax_time.get_xticklabels(), visible=False)

            sig_std = float(np.std(seg))
            if sig_std == 0.0 or not np.isfinite(sig_std):
                ax_cwt.text(0.5, 0.5, "Zero-variance segment; CWT skipped.",
                            transform=ax_cwt.transAxes, ha="center", va="center")
                ax_cwt.set_xlabel("Time (s)")
                ax_cwt.set_ylabel("Frequency (Hz)")
            else:
                sig_norm = (seg - float(np.mean(seg))) / sig_std

                if _HAS_PYWT:
                    cwt_result, _ = pywt.cwt(sig_norm, scales, 'morl', sampling_period=1.0 / fs)
                else:
                    cwt_result = sp_signal.cwt(sig_norm, lambda M, s: sp_signal.morlet2(M, s, w=5.0), scales)

                frequencies = (fc * fs) / scales
                valid = (frequencies >= min_freq) & (frequencies <= max_freq)
                freqs_v = frequencies[valid]
                cwt_v = cwt_result[valid, :]

                if cwt_v.shape[0] >= 2 and cwt_v.shape[1] >= 2:
                    order = np.argsort(freqs_v)
                    freqs_v = freqs_v[order]
                    cwt_v = cwt_v[order, :]
                    energy = 10.0 * np.log10(np.abs(cwt_v) + np.finfo(float).eps)
                    vmin = float(np.percentile(energy, 5))
                    vmax = float(np.percentile(energy, 95))

                    im = ax_cwt.imshow(
                        energy, aspect="auto", origin="lower",
                        extent=[t_start, t_end, freqs_v[0], freqs_v[-1]],
                        cmap=_CWT_CMAP, vmin=vmin, vmax=vmax,
                        interpolation='none', resample=False,
                    )
                    ax_cwt.set_xlabel("Time (s)")
                    ax_cwt.set_ylabel("Frequency (Hz)")
                    ax_cwt.set_title(f"{ch_name} CWT spectrogram")
                    ax_cwt.set_ylim(min_freq, max_freq)
                    ax_cwt.set_xlim(t_start, t_end)
                    ax_cwt.set_yticks(np.arange(0, int(np.floor(max_freq)) + 1, 1))
                    ax_cwt.margins(x=0)
                    fig.colorbar(im, cax=cbar_ax, label="Energy (dB)")

                    # Save SVG (spectrogram only, no labels)
                    svg_dir = os.path.join(save_dir, "cwt_svg_only")
                    os.makedirs(svg_dir, exist_ok=True)
                    fig_svg = plt.figure(figsize=(11.42, 4.8), constrained_layout=True, dpi=dpi)
                    gs_svg = GridSpec(1, 2, figure=fig_svg, width_ratios=[1, 0.05], hspace=0.1)
                    ax_svg = fig_svg.add_subplot(gs_svg[0, 0])
                    cbar_svg = fig_svg.add_subplot(gs_svg[0, 1])

                    im_svg = ax_svg.imshow(
                        energy, aspect="auto", origin="lower",
                        extent=[t_start, t_end, freqs_v[0], freqs_v[-1]],
                        cmap=_CWT_CMAP, vmin=vmin, vmax=vmax,
                        interpolation='none', resample=False,
                    )
                    ax_svg.set_xlim(t_start, t_end)
                    ax_svg.set_ylim(min_freq, max_freq)
                    ax_svg.set_yticks(np.arange(0, int(np.floor(max_freq)) + 1, 1))
                    ax_svg.margins(x=0)
                    for attr in ('title', 'xlabel', 'ylabel'):
                        getattr(ax_svg, f'set_{attr}')('')
                    ax_svg.set_xticklabels([])
                    ax_svg.set_yticklabels([])

                    cb = fig_svg.colorbar(im_svg, cax=cbar_svg)
                    cb.set_label("")
                    cb.ax.set_yticklabels([])

                    svg_path = os.path.join(svg_dir, f"{prefix}_{ch_name}_chunk{chunk_idx + 1:04d}_cwt_only.svg")
                    fig_svg.savefig(svg_path, format='svg', dpi=dpi, bbox_inches=None,
                                    pad_inches=0.0, transparent=True, facecolor='none', edgecolor='none')
                    plt.close(fig_svg)

            os.makedirs(save_dir, exist_ok=True)
            png_path = os.path.join(save_dir, f"{prefix}_{ch_name}_chunk{chunk_idx + 1:04d}.png")
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor='white', edgecolor='none')
            if not show:
                plt.close(fig)

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _process_channel(
    signal_or_path: np.ndarray | str,
    fs: float,
    min_freq: float,
    max_freq: float,
    n_freqs: int,
    fc: float,
    threshold_db: Optional[float],
    th_percentile: float,
    min_band_hz: float,
    min_duration_s: float,
    join_gap_s: float,
) -> List[Dict[str, float]]:
    sig = load_signal_txt(signal_or_path) if isinstance(signal_or_path, str) else np.asarray(signal_or_path, dtype=float).reshape(-1)
    energy_db, freqs, t = compute_cwt_energy(sig, fs, min_freq, max_freq, n_freqs, fc)
    bright_mask = detect_bright_time_mask(energy_db, freqs, threshold_db, th_percentile, min_band_hz)
    intervals = merge_time_mask_to_intervals(bright_mask, t, min_duration_s, join_gap_s)
    return [{"start": s, "end": e, "duration": e - s} for s, e in intervals]


def run_pipeline_from_matrix(
    data_matrix: np.ndarray,
    channels: List[str],
    channel_th_percentiles: Dict[str, float],
    default_th_percentile: float,
    sampling_rate: float = 15.0,
    points_per_sample: int = 450,
    min_freq: float = 0.1,
    max_freq: float = 7.5,
    n_freqs: int = 200,
    fc: float = 0.802,
    threshold_db: Optional[float] = None,
    min_band_hz: float = 1.0,
    min_duration_s: float = 0.066,
    join_gap_s: float = 0.5,
) -> Tuple[np.ndarray, Dict, Dict[str, List[Dict[str, float]]]]:
    if data_matrix.ndim != 2 or data_matrix.shape[1] < len(channels):
        raise ValueError(f"Expected matrix shape [N, {len(channels)}], got {data_matrix.shape}")

    per_channel_seconds: Dict[str, List[Dict[str, float]]] = {}
    per_channel_indices: Dict[str, List[Dict[str, int]]] = {}

    for i, ch in enumerate(channels):
        th = channel_th_percentiles.get(ch, default_th_percentile)
        per_channel_seconds[ch] = _process_channel(
            data_matrix[:, i], sampling_rate, min_freq, max_freq,
            n_freqs, fc, threshold_db, th, min_band_hz, min_duration_s, join_gap_s,
        )

    for ch in channels:
        per_channel_indices[ch] = convert_intervals_to_indices(
            per_channel_seconds.get(ch, []), sampling_rate, points_per_sample
        )

    masks, stats = build_masks_from_indices(per_channel_indices, channel_order=channels, points_per_sample=points_per_sample)
    return masks, stats, per_channel_seconds


def run_pipeline(
    data_dir: str,
    output_dir: str,
    channels: List[str],
    channel_th_percentiles: Dict[str, float],
    default_th_percentile: float,
    sampling_rate: float = 15.0,
    points_per_sample: int = 450,
    min_freq: float = 0.1,
    max_freq: float = 7.5,
    n_freqs: int = 200,
    fc: float = 0.802,
    threshold_db: Optional[float] = None,
    min_band_hz: float = 1.0,
    min_duration_s: float = 0.066,
    join_gap_s: float = 0.5,
    *,
    input_path: Optional[str] = None,
    save_intermediate: bool = False,
    mask_path: Optional[str] = None,
    save_stats: bool = False,
    enable_visualization: bool = False,
    vis_save_dir: Optional[str] = None,
    vis_prefix: str = "bright_bands",
    vis_dpi: int = 300,
    vis_show: bool = False,
    vis_chunk_size: int = 4500,
) -> Tuple[np.ndarray, Dict]:
    per_channel_seconds: Dict[str, List[Dict[str, float]]] = {}

    if input_path is not None:
        matrix = load_multichannel_matrix(input_path)
        masks, stats, per_channel_seconds = run_pipeline_from_matrix(
            data_matrix=matrix,
            channels=channels,
            channel_th_percentiles=channel_th_percentiles,
            default_th_percentile=default_th_percentile,
            sampling_rate=sampling_rate,
            points_per_sample=points_per_sample,
            min_freq=min_freq,
            max_freq=max_freq,
            n_freqs=n_freqs,
            fc=fc,
            threshold_db=threshold_db,
            min_band_hz=min_band_hz,
            min_duration_s=min_duration_s,
            join_gap_s=join_gap_s,
        )
        if enable_visualization:
            vis_dir = vis_save_dir or os.path.join(output_dir, 'bright_bands_visualization')
            visualize_bright_bands_chunked(
                data_matrix=matrix,
                channels=channels,
                per_channel_intervals=per_channel_seconds,
                save_dir=vis_dir,
                prefix=vis_prefix,
                dpi=vis_dpi,
                show=vis_show,
                chunk_size=vis_chunk_size,
                fs=sampling_rate,
                min_freq=min_freq,
                max_freq=max_freq,
                n_freqs=n_freqs,
                fc=fc,
            )
    else:
        per_channel_indices: Dict[str, List[Dict[str, int]]] = {}
        for ch in channels:
            th = channel_th_percentiles.get(ch, default_th_percentile)
            seconds_list = _process_channel(
                os.path.join(data_dir, f"{ch}_test.txt"),
                sampling_rate, min_freq, max_freq, n_freqs, fc,
                threshold_db, th, min_band_hz, min_duration_s, join_gap_s,
            )
            per_channel_seconds[ch] = seconds_list
            if save_intermediate:
                out_path = os.path.join(output_dir, f"bright_intervals_{ch}.json")
                os.makedirs(output_dir, exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(seconds_list, f, indent=2, ensure_ascii=False)

        for ch in channels:
            idx_list = convert_intervals_to_indices(per_channel_seconds.get(ch, []), sampling_rate, points_per_sample)
            per_channel_indices[ch] = idx_list
            if save_intermediate:
                out_path = os.path.join(output_dir, f"bright_intervals_{ch}_sample_indices.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(idx_list, f, indent=2, ensure_ascii=False)

        masks, stats = build_masks_from_indices(per_channel_indices, channel_order=channels, points_per_sample=points_per_sample)

    os.makedirs(output_dir, exist_ok=True)
    mask_out = mask_path or os.path.join(output_dir, 'noise_masks_from_bright_bands.npy')
    np.save(mask_out, masks.astype(bool))
    if save_stats:
        with open(os.path.join(output_dir, 'noise_masks_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    return masks, stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_MATRIX_PATH: Optional[str] = os.path.join('data', 'extracted_five_channels_timeline.txt')
DATA_DIR: str = '.'
OUTPUT_DIR: str = '.'
OUTPUT_MASK_PATH: Optional[str] = os.path.join(OUTPUT_DIR, 'noise_masks_from_bright_bands.npy')
CHANNELS: List[str] = ['EX', 'EY', 'HX', 'HY', 'HZ']

CHANNEL_TH_PERCENTILES: Dict[str, float] = {
    'EX': 94.0,
    'EY': 94.0,
    'HX': 94.0,
    'HY': 95.0,
    'HZ': 94.0,
}
DEFAULT_TH_PERCENTILE: float = 94.0

SAVE_INTERMEDIATE: bool = False
SAVE_STATS: bool = False
SAMPLING_RATE: float = 15.0
POINTS_PER_SAMPLE: int = 450
MIN_FREQ: float = 0.1
MAX_FREQ: float = 7.5
N_FREQS: int = 200
FC: float = 0.802
THRESHOLD_DB: Optional[float] = None
MIN_BAND_HZ: float = 1.0
MIN_DURATION_S: float = 0.066
JOIN_GAP_S: float = 0.5
ENABLE_VISUALIZATION: bool = True
VIS_SAVE_DIR: Optional[str] = None
VIS_PREFIX: str = "bright_bands"
VIS_DPI: int = 300
VIS_SHOW: bool = False
VIS_CHUNK_SIZE: int = 4500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    masks, stats = run_pipeline(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        channels=CHANNELS,
        channel_th_percentiles=CHANNEL_TH_PERCENTILES,
        default_th_percentile=DEFAULT_TH_PERCENTILE,
        sampling_rate=SAMPLING_RATE,
        points_per_sample=POINTS_PER_SAMPLE,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        n_freqs=N_FREQS,
        fc=FC,
        threshold_db=THRESHOLD_DB,
        min_band_hz=MIN_BAND_HZ,
        min_duration_s=MIN_DURATION_S,
        join_gap_s=JOIN_GAP_S,
        input_path=INPUT_MATRIX_PATH,
        save_intermediate=SAVE_INTERMEDIATE,
        mask_path=OUTPUT_MASK_PATH,
        save_stats=SAVE_STATS,
        enable_visualization=ENABLE_VISUALIZATION,
        vis_save_dir=VIS_SAVE_DIR,
        vis_prefix=VIS_PREFIX,
        vis_dpi=VIS_DPI,
        vis_show=VIS_SHOW,
        vis_chunk_size=VIS_CHUNK_SIZE,
    )

    print(f"Mask shape : {masks.shape}")
    if 'overall' in stats:
        print(f"Noise ratio: {stats['overall']['overall_noise_ratio']:.2%}")


if __name__ == '__main__':
    main()