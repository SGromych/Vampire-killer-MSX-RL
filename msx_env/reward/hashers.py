"""
Стабильный хэш комнаты/сцены по кадру. Устойчив к мерцанию (спрайты, мелкие изменения HUD).
С гистерезисом: новая комната засчитывается только если хэш держится K кадров подряд.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


def downscale_grayscale(obs: np.ndarray, size: tuple[int, int] = (21, 21)) -> np.ndarray:
    """Низкое разрешение для хэша: меньше чувствительность к анимации."""
    if obs.ndim == 3:
        obs = np.asarray(obs).mean(axis=-1)
    h, w = obs.shape[:2]
    if (h, w) == size:
        return np.asarray(obs, dtype=np.uint8)
    from PIL import Image
    img = Image.fromarray(np.asarray(obs, dtype=np.uint8))
    img = img.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)


def block_mean_hash(
    obs: np.ndarray,
    grid: tuple[int, int] = (4, 4),
    crop_rows_top: int = 0,
    crop_rows_bottom: int = 0,
) -> str:
    """
    Устойчивый к мелким изменениям хэш: разбить кадр на блоки, среднее по блоку, затем хэш.
    obs: (H, W) или (H, W, 3), будет приведён к grayscale и downscale.
    crop_rows_top/bottom: исключить N строк сверху/снизу (для room_hash без HUD).
    """
    if obs.ndim == 3:
        obs = np.asarray(obs).mean(axis=-1)
    h, w = obs.shape[:2]
    if crop_rows_top > 0 or crop_rows_bottom > 0:
        y0 = min(crop_rows_top, h - 1)
        y1 = max(y0 + 1, h - crop_rows_bottom)
        obs = np.asarray(obs[y0:y1, :], dtype=np.uint8)
    obs = downscale_grayscale(obs, (32, 32))
    h, w = obs.shape
    bh, bw = max(1, h // grid[0]), max(1, w // grid[1])
    blocks: list[float] = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            y0, y1 = i * bh, min((i + 1) * bh, h)
            x0, x1 = j * bw, min((j + 1) * bw, w)
            block = obs[y0:y1, x0:x1]
            blocks.append(float(np.mean(block)))
    key = hashlib.sha1(np.array(blocks, dtype=np.float64).tobytes()).hexdigest()
    return key


def room_hash_with_hysteresis(
    obs: np.ndarray,
    candidate_deque: deque[str],
    persistence: int,
    crop_rows_top: int = 0,
    crop_rows_bottom: int = 0,
) -> str | None:
    """
    Мутирует candidate_deque: добавляет текущий хэш. Возвращает stable_room_hash или None.
    stable_room_hash не None только если один и тот же хэш повторялся persistence раз подряд.
    crop_*: исключить строки (игнорировать HUD при хэше комнаты).
    """
    h = block_mean_hash(obs, crop_rows_top=crop_rows_top, crop_rows_bottom=crop_rows_bottom)
    candidate_deque.append(h)
    while len(candidate_deque) > persistence:
        candidate_deque.popleft()
    if len(candidate_deque) < persistence:
        return None
    first = candidate_deque[0]
    if all(x == first for x in candidate_deque):
        return first
    return None


def frame_diff_metric(prev: np.ndarray, curr: np.ndarray) -> float:
    """Нормализованная разница кадров (0..1). Для детекции залипания."""
    if prev.shape != curr.shape:
        return 1.0
    p = np.asarray(prev, dtype=np.float64)
    c = np.asarray(curr, dtype=np.float64)
    if p.ndim == 3:
        p = p.mean(axis=-1)
        c = c.mean(axis=-1)
    diff = np.abs(c - p)
    return float(np.mean(diff) / 255.0)


def position_proxy_x(obs: np.ndarray) -> float:
    """
    Горизонтальный центр масс «светлых» пикселей (proxy позиции игрока).
    Возвращает нормализованную X-координату 0..1. Устойчив к ключам в стенах (низкая доля кадра).
    """
    if obs.ndim == 3:
        obs = np.asarray(obs).mean(axis=-1)
    img = np.asarray(obs, dtype=np.float64)
    thr = np.percentile(img, 70)
    mask = (img >= thr).astype(np.float64)
    h, w = mask.shape
    column_sums = mask.sum(axis=0)
    total = column_sums.sum()
    if total < 1:
        return 0.5
    x_coords = np.arange(w, dtype=np.float64)
    x_center = (column_sums * x_coords).sum() / total
    return float(x_center / max(1, w - 1))
