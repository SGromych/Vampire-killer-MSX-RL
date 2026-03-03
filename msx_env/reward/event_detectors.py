"""
Плагинные детекторы событий (key/door/HP, progress). EventDetector interface.
v3: KeyDetector (HUD + confidence), DoorDetector (transition flash).
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class EventDetector(Protocol):
    """Интерфейс детектора: по obs/info возвращает события для наград или логирования."""

    def detect(self, obs: np.ndarray, info: dict[str, Any], rgb: np.ndarray | None = None) -> dict[str, Any]:
        """Возвращает словарь событий, например {"key_detected": True, "detection_confidence": 0.9}."""
        ...


class KeyDetector:
    """
    Детекция ключа по HUD (key_chest / key_door). Учитывает ключи в стенах — по яркости ROI.
    confidence 0..1 по нормализованной яркости слота. Один раз за эпизод награда в policy.
    """
    def detect(self, obs: np.ndarray, info: dict[str, Any], rgb: np.ndarray | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"key_detected": False, "door_detected": False, "detection_confidence": 0.0}
        if "hud" not in info:
            return out
        hud = info["hud"]
        key_chest = hud.get("key_chest", False)
        key_door = hud.get("key_door", False)
        out["key_detected"] = key_chest or key_door
        if key_chest or key_door:
            out["detection_confidence"] = 0.9 if (key_chest and key_door) else 0.7
        out["door_detected"] = False
        return out


class DoorDetector:
    """
    Детекция перехода через дверь: вспышка экрана (резкий скачок frame_diff) или смена комнаты.
    Эвристика: в policy передаётся prev_obs/obs — сильный frame_diff при смене room_hash может означать дверь.
    Здесь возвращаем door_detected=False; фактическая детекция «дверь открыта» по смене комнаты в policy.
    """
    def detect(self, obs: np.ndarray, info: dict[str, Any], rgb: np.ndarray | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"key_detected": False, "door_detected": False, "detection_confidence": 0.0}
        if info.get("reward_room_hash") and info.get("reward_prev_room_hash"):
            if info["reward_room_hash"] != info["reward_prev_room_hash"]:
                out["door_detected"] = True
                out["detection_confidence"] = 0.6
        return out


class KeyDoorDetector:
    """Объединённый детектор: ключ и дверь (обратная совместимость)."""
    def __init__(self) -> None:
        self._key = KeyDetector()
        self._door = DoorDetector()

    def detect(self, obs: np.ndarray, info: dict[str, Any], rgb: np.ndarray | None = None) -> dict[str, Any]:
        k = self._key.detect(obs, info, rgb)
        d = self._door.detect(obs, info, rgb)
        return {
            "has_key_chest": info.get("hud", {}).get("key_chest", False),
            "has_key_door": info.get("hud", {}).get("key_door", False),
            "key_detected": k["key_detected"],
            "door_detected": d["door_detected"],
            "detection_confidence": max(k["detection_confidence"], d["detection_confidence"]),
        }


class HPDetector:
    """Заглушка: HP/смерть уже через life_bar.get_life_estimate и terminated_on_death."""
    def detect(self, obs: np.ndarray, info: dict[str, Any], rgb: np.ndarray | None = None) -> dict[str, Any]:
        return {}


class ProgressProxyDetector:
    """Заглушка: прогресс вправо (например по x спрайта). Ненадёжно без детекции позиции — отключено по умолчанию."""
    def detect(self, obs: np.ndarray, info: dict[str, Any], rgb: np.ndarray | None = None) -> dict[str, Any]:
        return {"delta_x": 0}
