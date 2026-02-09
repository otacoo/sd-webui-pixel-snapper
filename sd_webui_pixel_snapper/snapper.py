"""Pixel grid snapping and color quantization for AI-generated pixel art.

Original author and repo: https://github.com/Hugo-Dz/spritefusion-pixel-snapper
Ported from Rust to Python by otacoo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


class PixelSnapperError(Exception):
    pass


@dataclass
class SnapperConfig:
    k_colors: int = 16
    k_seed: int = 42
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8


def _validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise PixelSnapperError("Image dimensions cannot be zero")
    if width > 10000 or height > 10000:
        raise PixelSnapperError("Image dimensions too large (max 10000x10000)")


def _quantize_image(rgba: np.ndarray, config: SnapperConfig) -> np.ndarray:
    h, w = rgba.shape[:2]
    if config.k_colors <= 0:
        raise PixelSnapperError("Number of colors must be greater than 0")

    mask = rgba[:, :, 3] > 0
    opaque = rgba[mask, :3].astype(np.float32)
    n_pixels = len(opaque)
    if n_pixels == 0:
        return rgba.copy()

    k = min(config.k_colors, n_pixels)
    rng = np.random.default_rng(config.k_seed)

    centroids = np.array([opaque[rng.integers(0, n_pixels)]], dtype=np.float32)
    for _ in range(1, k):
        dists = np.full(n_pixels, np.finfo(np.float32).max, dtype=np.float32)
        for i in range(len(centroids)):
            d = np.sum((opaque - centroids[i]) ** 2, axis=1)
            np.minimum(dists, d, out=dists)
        total = float(dists.sum())
        if total <= 0:
            centroids = np.vstack([centroids, opaque[rng.integers(0, n_pixels)]])
        else:
            probs = dists / total
            centroids = np.vstack([centroids, opaque[rng.choice(n_pixels, p=probs)]])

    n_c = len(centroids)
    dists = np.empty((n_pixels, n_c), dtype=np.float32)
    for _ in range(config.max_kmeans_iterations):
        for i in range(n_c):
            dists[:, i] = np.sum((opaque - centroids[i]) ** 2, axis=1)
        labels = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for i in range(n_c):
            cluster = opaque[labels == i]
            if len(cluster) > 0:
                new_centroids[i] = cluster.mean(axis=0)

        movement_sq = np.sum((new_centroids - centroids) ** 2, axis=1)
        centroids = new_centroids
        if np.max(movement_sq) < 0.01:
            break

    out = rgba.copy()
    out[mask, :3] = np.clip(np.round(centroids[labels]), 0, 255).astype(np.uint8)
    return out


def _compute_profiles(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgba.shape[:2]
    if w < 3 or h < 3:
        raise PixelSnapperError("Image too small (minimum 3x3)")

    gray = np.dot(rgba[:, :, :3].astype(np.float64), [0.299, 0.587, 0.114])
    gray[rgba[:, :, 3] == 0] = 0.0

    col_proj = np.zeros(w)
    col_proj[1 : w - 1] = np.sum(np.abs(gray[:, 2:] - gray[:, :-2]), axis=0)
    row_proj = np.zeros(h)
    row_proj[1 : h - 1] = np.sum(np.abs(gray[2:, :] - gray[:-2, :]), axis=1)
    return col_proj, row_proj


def _estimate_step_size(profile: np.ndarray, config: SnapperConfig) -> Optional[float]:
    if len(profile) == 0:
        return None
    max_val = float(np.max(profile))
    if max_val == 0:
        return None
    threshold = max_val * config.peak_threshold_multiplier
    peaks = [i for i in range(1, len(profile) - 1) if profile[i] > threshold and profile[i] > profile[i - 1] and profile[i] > profile[i + 1]]
    if len(peaks) < 2:
        return None
    clean = [peaks[0]]
    for p in peaks[1:]:
        if p - clean[-1] > (config.peak_distance_filter - 1):
            clean.append(p)
    if len(clean) < 2:
        return None
    diffs = np.diff(clean).astype(float)
    diffs.sort()
    return float(diffs[len(diffs) // 2])


def _resolve_step_sizes(step_x: Optional[float], step_y: Optional[float], width: int, height: int, config: SnapperConfig) -> tuple[float, float]:
    if step_x is not None and step_y is not None:
        ratio = max(step_x, step_y) / min(step_x, step_y)
        if ratio > config.max_step_ratio:
            s = min(step_x, step_y)
            return (s, s)
        return ((step_x + step_y) / 2.0, (step_x + step_y) / 2.0)
    if step_x is not None:
        return (step_x, step_x)
    if step_y is not None:
        return (step_y, step_y)
    fallback = max((min(width, height) / config.fallback_target_segments), 1.0)
    return (fallback, fallback)


def _walk(profile: np.ndarray, step_size: float, limit: int, config: SnapperConfig) -> list[int]:
    if len(profile) == 0:
        raise PixelSnapperError("Cannot walk on empty profile")
    cuts = [0]
    current_pos = 0.0
    search_window = max(step_size * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = float(np.mean(profile))
    while current_pos < limit:
        target = current_pos + step_size
        if target >= limit:
            cuts.append(limit)
            break
        start = max(int(target - search_window), int(current_pos) + 1)
        end = min(int(target + search_window), limit)
        if end <= start:
            current_pos = target
            continue
        segment = profile[start:end]
        max_idx = start + int(np.argmax(segment))
        if segment.max() > mean_val * config.walker_strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target
    return cuts


def _sanitize_cuts(cuts: list[int], limit: int) -> list[int]:
    if limit == 0:
        return [0]
    cuts = list(cuts)
    for i in range(len(cuts)):
        if cuts[i] >= limit:
            cuts[i] = limit
    if 0 not in cuts:
        cuts.append(0)
    if limit not in cuts:
        cuts.append(limit)
    return sorted(set(cuts))


def _snap_uniform_cuts(profile: np.ndarray, limit: int, target_step: float, config: SnapperConfig, min_required: int) -> list[int]:
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]
    desired_cells = max(
        min(int(round(limit / target_step)) if target_step > 0 else 0, limit),
        min_required - 1 if min_required > 1 else 0,
        1,
    )
    cell_width = limit / desired_cells
    search_window = max(cell_width * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = float(np.mean(profile)) if len(profile) > 0 else 0.0
    cuts = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break
        start = max(int(math.floor(target - search_window)), prev + 1, 0)
        end_incl = min(int(math.ceil(target + search_window)), limit - 1)
        if end_incl < start:
            start = prev + 1
            end_incl = start
        segment = profile[start : end_incl + 1] if end_incl + 1 <= len(profile) else profile[start:]
        if len(segment) == 0:
            best_idx = int(round(target))
        else:
            best_idx = start + int(np.argmax(segment))
            if segment.max() < mean_val * config.walker_strength_threshold:
                best_idx = max(prev + 1, min(int(round(target)), limit - 1))
        cuts.append(best_idx)
    if cuts[-1] != limit:
        cuts.append(limit)
    return _sanitize_cuts(cuts, limit)


def _stabilize_both_axes(profile_x: np.ndarray, profile_y: np.ndarray, raw_col_cuts: list[int], raw_row_cuts: list[int], width: int, height: int, config: SnapperConfig) -> tuple[list[int], list[int]]:
    col_cuts_pass1 = _stabilize_cuts(profile_x, raw_col_cuts.copy(), width, raw_row_cuts, height, config)
    row_cuts_pass1 = _stabilize_cuts(profile_y, raw_row_cuts.copy(), height, raw_col_cuts, width, config)
    col_cells = max(len(col_cuts_pass1) - 1, 1)
    row_cells = max(len(row_cuts_pass1) - 1, 1)
    col_step = width / col_cells
    row_step = height / row_cells
    step_ratio = max(col_step, row_step) / min(col_step, row_step)
    if step_ratio > config.max_step_ratio:
        target_step = min(col_step, row_step)
        final_col = _snap_uniform_cuts(profile_x, width, target_step, config, config.min_cuts_per_axis) if col_step > target_step * 1.2 else col_cuts_pass1
        final_row = _snap_uniform_cuts(profile_y, height, target_step, config, config.min_cuts_per_axis) if row_step > target_step * 1.2 else row_cuts_pass1
        return (final_col, final_row)
    return (col_cuts_pass1, row_cuts_pass1)


def _stabilize_cuts(profile: np.ndarray, cuts: list[int], limit: int, sibling_cuts: list[int], sibling_limit: int, config: SnapperConfig) -> list[int]:
    if limit == 0:
        return [0]
    cuts = _sanitize_cuts(cuts, limit)
    min_required = max(min(config.min_cuts_per_axis, limit + 1), 2)
    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)
    sibling_has_grid = sibling_limit > 0 and sibling_cells >= min_required - 1 and sibling_cells > 0
    axis_step = limit / axis_cells if axis_cells else 0
    sibling_step = sibling_limit / sibling_cells if sibling_cells else 0
    steps_skewed = (
        sibling_has_grid and axis_cells > 0 and sibling_cells > 0
        and (axis_step / sibling_step > config.max_step_ratio or axis_step / sibling_step < 1.0 / config.max_step_ratio)
    )
    if len(cuts) >= min_required and not steps_skewed:
        return cuts
    if sibling_has_grid and sibling_cells > 0:
        target_step = sibling_limit / sibling_cells
    elif config.fallback_target_segments > 1:
        target_step = limit / config.fallback_target_segments
    elif axis_cells > 0:
        target_step = limit / axis_cells
    else:
        target_step = float(limit)
    if not np.isfinite(target_step) or target_step <= 0:
        target_step = 1.0
    return _snap_uniform_cuts(profile, limit, target_step, config, min_required)


def _resample(rgba: np.ndarray, cols: list[int], rows: list[int]) -> np.ndarray:
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")
    out_w = len(cols) - 1
    out_h = len(rows) - 1
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for y_i in range(out_h):
        ys, ye = rows[y_i], rows[y_i + 1]
        for x_i in range(out_w):
            xs, xe = cols[x_i], cols[x_i + 1]
            if xe <= xs or ye <= ys:
                continue
            region = rgba[ys:ye, xs:xe].reshape(-1, 4)
            linear = region[:, 0].astype(np.uint32) + (region[:, 1].astype(np.uint32) << 8) + (region[:, 2].astype(np.uint32) << 16) + (region[:, 3].astype(np.uint32) << 24)
            vals, counts = np.unique(linear, return_counts=True)
            best_linear = np.min(vals[counts == np.max(counts)])
            out[y_i, x_i, 0] = best_linear & 0xFF
            out[y_i, x_i, 1] = (best_linear >> 8) & 0xFF
            out[y_i, x_i, 2] = (best_linear >> 16) & 0xFF
            out[y_i, x_i, 3] = (best_linear >> 24) & 0xFF
    return out


def process_pil_image(image: Image.Image, k_colors: int = 16, config: Optional[SnapperConfig] = None, max_dimension: int = 0) -> Image.Image:
    config = config or SnapperConfig()
    config.k_colors = max(1, k_colors)

    img = image.convert("RGBA")
    width, height = img.size
    _validate_dimensions(width, height)

    if max_dimension > 0 and max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        img = img.resize((new_w, new_h), Image.NEAREST)
        width, height = new_w, new_h
        print(f"[Pixel Snapper] Downscaled to {width}x{height} for processing", flush=True)

    rgba = np.array(img)
    print(f"[Pixel Snapper] Processing {width}x{height} (k={config.k_colors})...", flush=True)
    quantized = _quantize_image(rgba, config)
    print("[Pixel Snapper] Quantizing done, detecting grid...", flush=True)
    profile_x, profile_y = _compute_profiles(quantized)

    step_x = _estimate_step_size(profile_x, config)
    step_y = _estimate_step_size(profile_y, config)
    step_x, step_y = _resolve_step_sizes(step_x, step_y, width, height, config)

    raw_col_cuts = _walk(profile_x, step_x, width, config)
    raw_row_cuts = _walk(profile_y, step_y, height, config)
    col_cuts, row_cuts = _stabilize_both_axes(profile_x, profile_y, raw_col_cuts, raw_row_cuts, width, height, config)

    print("[Pixel Snapper] Resampling...", flush=True)
    out_rgba = _resample(quantized, col_cuts, row_cuts)
    print(f"[Pixel Snapper] Done. Output {out_rgba.shape[1]}x{out_rgba.shape[0]}", flush=True)
    return Image.fromarray(out_rgba, "RGBA").convert("RGB")
