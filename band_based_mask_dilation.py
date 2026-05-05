#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
from typing import List, Tuple, Dict

import numpy as np


class Config:
    band_data_dir = os.path.join('vmd_outlierfree_ultrasmooth_results', 'data')
    original_txt_path = os.path.join('data', 'extracted_five_channels_timeline.txt')
    old_mask_path = os.path.join('denoised_results_test1', 'noise_masks.npy')
    points_per_sample = 450
    include_low = True
    min_run_keep = 2
    min_run_update = 5
    output_dir = os.path.join('denoised_results_test1')
    save_stats = True


def _find_true_runs_1d(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask_1d, dtype=bool)
    if not np.any(m):
        return []
    idx = np.where(m)[0]
    runs: List[Tuple[int, int]] = []
    s = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
        else:
            runs.append((s, prev))
            s = i
            prev = i
    runs.append((s, prev))
    return runs


def _length_open_1d(mask_1d: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1:
        return mask_1d.astype(bool)
    m = mask_1d.astype(bool).copy()
    for s, e in _find_true_runs_1d(m):
        if (e - s + 1) < min_len:
            m[s: e + 1] = False
    return m


def load_custom_original_data(txt_path: str) -> np.ndarray:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Original signal file not found: {txt_path}")
    try:
        original_data = np.loadtxt(txt_path)
    except Exception as e:
        raise ValueError(f"File format error: {e}")
    if original_data.ndim == 1 or original_data.shape[1] != 5:
        raise ValueError(
            f"Expected 5 columns, got shape {original_data.shape}"
        )
    T = original_data.shape[0]
    return original_data.reshape(1, T, 5)


def load_band_data(band_dir: str, custom_txt_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper_path = os.path.join(band_dir, 'upper_band.npy')
    lower_path = os.path.join(band_dir, 'lower_band.npy')
    if not os.path.exists(upper_path):
        raise FileNotFoundError(f"Upper band file not found: {upper_path}")
    if not os.path.exists(lower_path):
        raise FileNotFoundError(f"Lower band file not found: {lower_path}")
    upper = np.load(upper_path)
    lower = np.load(lower_path)
    original = load_custom_original_data(custom_txt_path)
    if upper.shape[1] != original.shape[1] or upper.shape[2] != original.shape[2]:
        raise ValueError(
            f"Shape mismatch between original signal and band arrays: "
            f"band=({upper.shape[1]}, {upper.shape[2]}), "
            f"original=({original.shape[1]}, {original.shape[2]})"
        )
    return upper, lower, original


def build_high_regions_mask(
    upper: np.ndarray,
    lower: np.ndarray,
    original: np.ndarray,
    include_low: bool,
    min_run_keep: int,
) -> np.ndarray:
    above_upper = original >= upper
    base_mask = np.logical_or(above_upper, original <= lower) if include_low else above_upper
    _, T, C = base_mask.shape
    refined_mask = np.zeros_like(base_mask, dtype=bool)
    for ch in range(C):
        refined_mask[0, :, ch] = _length_open_1d(base_mask[0, :, ch], min_run_keep)
    return refined_mask


def reshape_long_to_samples(mask_long: np.ndarray, points_per_sample: int) -> np.ndarray:
    if mask_long.ndim != 3 or mask_long.shape[0] != 1:
        raise ValueError(f"Expected shape [1, T, C], got {mask_long.shape}")
    _, T, C = mask_long.shape
    if T % points_per_sample != 0:
        raise ValueError(
            f"Total length {T} is not divisible by points_per_sample={points_per_sample}"
        )
    N = T // points_per_sample
    return mask_long.reshape(N, points_per_sample, C)


def update_old_mask_with_new(
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    min_run_update: int,
) -> Tuple[np.ndarray, Dict]:
    if old_mask.shape != new_mask.shape:
        raise ValueError(
            f"Shape mismatch: old={old_mask.shape}, new={new_mask.shape}"
        )
    N, T, C = old_mask.shape
    old_bool = old_mask.astype(bool)
    new_bool = new_mask.astype(bool)
    final_mask = old_bool.copy()
    updated_runs = 0
    total_long_runs = 0

    for n in range(N):
        for ch in range(C):
            for s, e in _find_true_runs_1d(new_bool[n, :, ch]):
                if (e - s + 1) >= min_run_update:
                    total_long_runs += 1
                    final_mask[n, s:e + 1, ch] = new_bool[n, s:e + 1, ch]
                    updated_runs += 1

    old_sum = int(np.sum(old_bool))
    final_sum = int(np.sum(final_mask))
    stats = {
        'long_runs_in_new_mask': total_long_runs,
        'updated_runs': updated_runs,
        'old_mask_true_count': old_sum,
        'new_mask_true_count': int(np.sum(new_bool)),
        'final_mask_true_count': final_sum,
        'added_true_count': final_sum - old_sum,
        'coverage_increase_ratio': float((final_sum - old_sum) / old_sum) if old_sum > 0 else 0.0,
    }
    return final_mask, stats


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    upper, lower, original = load_band_data(
        band_dir=cfg.band_data_dir,
        custom_txt_path=cfg.original_txt_path,
    )
    high_mask_long = build_high_regions_mask(
        upper=upper,
        lower=lower,
        original=original,
        include_low=cfg.include_low,
        min_run_keep=cfg.min_run_keep,
    )
    new_mask = reshape_long_to_samples(high_mask_long, cfg.points_per_sample)
    np.save(os.path.join(cfg.output_dir, 'high_regions_mask.npy'), new_mask.astype(bool))

    if not os.path.exists(cfg.old_mask_path):
        raise FileNotFoundError(f"Old mask not found: {cfg.old_mask_path}")
    old_mask = np.load(cfg.old_mask_path).astype(bool)
    if old_mask.shape != new_mask.shape:
        raise ValueError(
            f"Shape mismatch: old_mask={old_mask.shape}, new_mask={new_mask.shape}"
        )

    final_mask, update_stats = update_old_mask_with_new(
        old_mask=old_mask,
        new_mask=new_mask,
        min_run_update=cfg.min_run_update,
    )
    np.save(
        os.path.join(cfg.output_dir, 'noise_masks_updated_by_high_regions.npy'),
        final_mask.astype(bool),
    )

    if cfg.save_stats:
        summary = {
            'config': {
                'band_data_dir': cfg.band_data_dir,
                'original_txt_path': cfg.original_txt_path,
                'old_mask_path': cfg.old_mask_path,
                'points_per_sample': cfg.points_per_sample,
                'include_low': cfg.include_low,
                'min_run_keep': cfg.min_run_keep,
                'min_run_update': cfg.min_run_update,
            },
            'shapes': {
                'original': list(original.shape),
                'band': list(upper.shape),
                'new_mask': list(new_mask.shape),
                'old_mask': list(old_mask.shape),
                'final_mask': list(final_mask.shape),
            },
            'update_stats': update_stats,
        }
        stats_path = os.path.join(cfg.output_dir, 'band_based_mask_update_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()