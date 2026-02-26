"""
Оценка полоски жизни (PLAYER) по кадру для Vampire Killer.
Полоска вверху экрана; при ударе врагов уменьшается, при нуле — смерть.
"""
from __future__ import annotations

import numpy as np

# ROI: верх экрана, область HUD с надписью PLAYER и полоской жизни.
# Для 84x84 кадра — верхние строки (полоска обычно в первых ~10–15 пикселях по высоте).
LIFE_TOP = 0
LIFE_BOTTOM = 10
LIFE_LEFT = 8
LIFE_RIGHT = 76


def get_life_estimate(frame: np.ndarray) -> float:
    """
    По кадру (H, W) grayscale uint8 вернуть оценку заполненности полоски жизни 0…1.
    Чем ярче ROI вверху — тем «полнее» считаем жизнь (упрощённая эвристика).
    """
    if frame.ndim != 2:
        frame = np.asarray(frame)
        if frame.ndim == 3:
            frame = frame.mean(axis=-1)
    h, w = frame.shape
    y0 = max(0, min(LIFE_TOP, h - 1))
    y1 = max(y0 + 1, min(LIFE_BOTTOM, h))
    x0 = max(0, min(LIFE_LEFT, w - 1))
    x1 = max(x0 + 1, min(LIFE_RIGHT, w))
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.5
    # Нормализуем среднюю яркость к 0..1 (пустая полоска — тёмная)
    mean_val = float(np.mean(roi))
    return min(1.0, max(0.0, mean_val / 255.0))
