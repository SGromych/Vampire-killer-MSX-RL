"""
Парсер HUD Vampire Killer: оружие, ключи (сундук/дверь), предметы.

Структура HUD: верхняя строка SCORE, STAGE-XX, сердечко-XX, P-XX (жизни); ниже PLAYER/ENEMY
и слоты оружия, ключей, предметов (см. docs/VAMPIRE_KILLER_SPEC.md, docs/ITEMS_AND_REWARDS.md).

ROI заданы долями ширины/высоты кадра (0..1), справа от полосок жизни (PLAYER/ENEMY).
Слот считается «заполненным», если средняя яркость в ROI >= SLOT_FILLED_THRESHOLD.
Используется: reward за подбор в env.step() (legacy и reward_config), info['hud'].
Масштабы наград за подбор настраиваются в RewardConfig; см. docs/REWARD.md, docs/TRAINING.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ROI как доли ширины/высоты (0..1) — подстраиваются под разрешение скриншота
# Ориентир: HUD вверху, блоки справа от жизней
WEAPON_ROI = (0.52, 0.02, 0.62, 0.12)   # (x0, y0, x1, y1)
KEY_CHEST_ROI = (0.68, 0.02, 0.77, 0.12)  # ключ для сундуков (skeleton)
KEY_DOOR_ROI = (0.77, 0.02, 0.86, 0.12)   # ключ для двери с уровня (cross)
ITEMS_ROI = (0.88, 0.02, 0.98, 0.12)      # блок предметов (hourglass и др.)

# STAGE: два знака "00", "01", ... в верхней части HUD (доли 0..1)
STAGE_DIGITS_ROI = (0.08, 0.02, 0.18, 0.12)  # (x0, y0, x1, y1) — подстрой под свой скриншот

# Порог: средняя яркость выше = слот занят (пустой слот тёмный)
SLOT_FILLED_THRESHOLD = 60

# Размер одного символа для шаблонов (W, H)
_STAGE_DIGIT_SIZE = (5, 7)

# Минимальные бинарные шаблоны 5x7 для цифр 0–9 (1 = светлый пиксель). MSX-подобный шрифт.
_STAGE_DIGIT_TEMPLATES: list[np.ndarray] = [
    np.array([  # 0
        [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 1
        [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 2
        [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 3
        [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 4
        [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 5
        [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 6
        [0, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 7
        [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 8
        [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0],
    ], dtype=np.uint8).reshape(7, 5),
    np.array([  # 9
        [0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 1, 1, 0, 0],
    ], dtype=np.uint8).reshape(7, 5),
]


@dataclass
class HudState:
    """Состояние HUD-слотов (оружие, ключи, предметы)."""
    weapon: bool
    key_chest: bool
    key_door: bool
    items: int  # 0..N, упрощённо: bool как 0/1

    def to_tuple(self) -> tuple[bool, bool, bool, int]:
        return (self.weapon, self.key_chest, self.key_door, self.items)


def _roi_mean(img: np.ndarray, roi: tuple[float, float, float, float]) -> float:
    """Средняя яркость в ROI (доли 0..1). img: (H,W) или (H,W,3)."""
    h, w = img.shape[:2]
    x0 = int(roi[0] * w)
    y0 = int(roi[1] * h)
    x1 = max(x0 + 1, int(roi[2] * w))
    y1 = max(y0 + 1, int(roi[3] * h))
    x0, x1 = max(0, min(x0, w - 1)), max(x0 + 1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(y0 + 1, min(y1, h))
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    if patch.ndim == 3:
        patch = patch.mean(axis=-1)
    return float(np.mean(patch))


def parse_hud(frame_or_path: np.ndarray | Path) -> HudState:
    """
    Распарсить HUD из кадра или пути к скриншоту.
    Возвращает HudState с флагами/количеством по слотам.
    """
    if isinstance(frame_or_path, Path):
        from PIL import Image
        img = np.array(
            Image.open(frame_or_path).convert("RGB"),
            dtype=np.uint8,
        )
    else:
        img = np.asarray(frame_or_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

    thr = SLOT_FILLED_THRESHOLD
    weapon = _roi_mean(img, WEAPON_ROI) >= thr
    key_chest = _roi_mean(img, KEY_CHEST_ROI) >= thr
    key_door = _roi_mean(img, KEY_DOOR_ROI) >= thr
    items_val = _roi_mean(img, ITEMS_ROI)
    items = 1 if items_val >= thr else 0  # упрощение: есть предметы или нет

    return HudState(
        weapon=weapon,
        key_chest=key_chest,
        key_door=key_door,
        items=items,
    )


def _crop_roi(img: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Вернуть patch (H,W) в границах roi (доли)."""
    h, w = img.shape[:2]
    x0 = max(0, int(roi[0] * w))
    y0 = max(0, int(roi[1] * h))
    x1 = min(w, max(x0 + 1, int(roi[2] * w)))
    y1 = min(h, max(y0 + 1, int(roi[3] * h)))
    patch = img[y0:y1, x0:x1]
    if patch.ndim == 3:
        patch = patch.mean(axis=-1)
    return np.asarray(patch, dtype=np.float64)


def _resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize (H,W) to (out_h, out_w)."""
    if arr.size == 0:
        return np.zeros((out_h, out_w), dtype=arr.dtype)
    in_h, in_w = arr.shape[:2]
    y_idx = np.linspace(0, in_h - 1, out_h, dtype=int)
    x_idx = np.linspace(0, in_w - 1, out_w, dtype=int)
    return arr[np.ix_(y_idx, x_idx)]


def _match_digit(patch: np.ndarray, templates: list[np.ndarray]) -> tuple[int, float]:
    """patch: (H,W). Нормализует к 5x7, бинаризует, сравнивает с шаблонами. Возвращает (digit 0-9, confidence 0..1)."""
    h, w = _STAGE_DIGIT_SIZE[1], _STAGE_DIGIT_SIZE[0]
    if patch.size == 0:
        return 0, 0.0
    resized = _resize_nearest(patch, h, w)
    thr = float(np.median(resized))
    binary = (resized >= thr).astype(np.uint8)
    best_digit = 0
    best_score = -1.0
    for d, tpl in enumerate(templates):
        if tpl.shape != (h, w):
            tpl_rs = _resize_nearest(tpl.astype(np.float64), h, w)
            tpl_rs = (tpl_rs >= 0.5).astype(np.uint8)
        else:
            tpl_rs = tpl
        score = float(np.sum(binary == tpl_rs)) / max(1, binary.size)
        if score > best_score:
            best_score = score
            best_digit = d
    return best_digit, best_score


def parse_stage(frame_or_path: np.ndarray | Path) -> tuple[int, float]:
    """
    Прочитать номер STAGE (00, 01, ...) из HUD.
    Возвращает (stage_int, confidence). При ошибке или низкой уверенности: (0, 0.0).
    """
    if isinstance(frame_or_path, Path):
        from PIL import Image
        img = np.array(Image.open(frame_or_path).convert("L"), dtype=np.float64)
    else:
        img = np.asarray(frame_or_path)
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float64)
    roi = STAGE_DIGITS_ROI
    patch = _crop_roi(img, roi)
    if patch.size < 10:
        return 0, 0.0
    h, w = patch.shape[0], patch.shape[1]
    mid = w // 2
    left = patch[:, :mid]
    right = patch[:, mid:]
    d1, c1 = _match_digit(left, _STAGE_DIGIT_TEMPLATES)
    d2, c2 = _match_digit(right, _STAGE_DIGIT_TEMPLATES)
    stage = d1 * 10 + d2
    confidence = min(c1, c2)
    return stage, confidence


def compute_pickup_reward(
    prev: HudState | None,
    curr: HudState,
    *,
    reward_weapon: float = 0.3,
    reward_key_chest: float = 1.0,
    reward_key_door: float = 1.0,
    reward_item: float = 0.5,
) -> float:
    """Награда за подбор: +reward_* при переходе пусто→заполнено."""
    if prev is None:
        return 0.0
    r = 0.0
    if not prev.weapon and curr.weapon:
        r += reward_weapon
    if not prev.key_chest and curr.key_chest:
        r += reward_key_chest
    if not prev.key_door and curr.key_door:
        r += reward_key_door
    if prev.items < curr.items:
        r += reward_item * (curr.items - prev.items)
    return r
