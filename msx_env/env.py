from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from openmsx_bridge import OpenMSXFileControl


# Дискретный экшн‑спейс для Vampire Killer
ACTION_NOOP = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_ATTACK = 5
ACTION_RIGHT_JUMP = 6
ACTION_LEFT_JUMP = 7
ACTION_RIGHT_JUMP_ATTACK = 8
ACTION_LEFT_JUMP_ATTACK = 9

ACTION_SET_VERSION = "vkiller_v1"
NUM_ACTIONS = 10


@dataclass
class EnvConfig:
    rom_path: str
    workdir: str
    frame_size: Tuple[int, int] = (84, 84)


class VampireKillerEnv:
    """
    Минимальный Gym‑подобный env вокруг OpenMSXFileControl.

    reset() -> obs, info
    step(action) -> obs, reward, terminated, truncated, info
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._emu: OpenMSXFileControl | None = None

    # --------- низкоуровневые helpers ---------

    def _ensure_emu(self) -> OpenMSXFileControl:
        if self._emu is None:
            self._emu = OpenMSXFileControl(
                rom_path=self.cfg.rom_path,
                workdir=self.cfg.workdir,
                poll_ms=20,
                boot_timeout_s=30.0,
            )
        return self._emu

    def _skip_intro(self, emu: OpenMSXFileControl) -> None:
        # Ждём ухода BIOS‑экрана
        emu.wait_for_nonblue_screen(timeout_s=30.0, check_every_s=0.5)
        time.sleep(1.0)  # дать заголовку/меню стабилизироваться
        # Пропустить заставку Konami
        emu.press_type("SPACE")
        time.sleep(2.0)
        # PRESS START — запуск игры
        emu.press_type("SPACE")
        time.sleep(3.0)  # загрузка уровня

    def _grab_obs(self, emu: OpenMSXFileControl) -> np.ndarray:
        """
        Захват obs так, как будет видеть агент:
        - делаем скриншот в фиксированный файл step_frame.png
        - resize до frame_size
        - grayscale, uint8
        """
        from PIL import Image as _PILImage

        # Делаем скриншот; файл всегда кладём в workdir/step_frame.png
        emu.screenshot("step_frame.png")
        img_path = Path(self.cfg.workdir) / "step_frame.png"

        img = _PILImage.open(img_path).convert("RGB")
        w, h = self.cfg.frame_size
        img = img.resize((w, h)).convert("L")  # 84x84 gray
        arr = np.array(img, dtype=np.uint8)  # (H, W)
        return arr

    def _apply_action(self, emu: OpenMSXFileControl, action: int) -> None:
        if action == ACTION_NOOP:
            return
        elif action == ACTION_RIGHT:
            emu.press("RIGHT", hold_ms=120)
        elif action == ACTION_LEFT:
            emu.press("LEFT", hold_ms=120)
        elif action == ACTION_UP:
            emu.press("UP", hold_ms=200)
        elif action == ACTION_DOWN:
            emu.press("DOWN", hold_ms=200)
        elif action == ACTION_ATTACK:
            emu.press("SPACE", hold_ms=120)
        elif action == ACTION_RIGHT_JUMP:
            emu.press_combo(["RIGHT", "UP"], hold_ms=220)
        elif action == ACTION_LEFT_JUMP:
            emu.press_combo(["LEFT", "UP"], hold_ms=220)
        elif action == ACTION_RIGHT_JUMP_ATTACK:
            emu.press_combo(["RIGHT", "UP", "SPACE"], hold_ms=220)
        elif action == ACTION_LEFT_JUMP_ATTACK:
            emu.press_combo(["LEFT", "UP", "SPACE"], hold_ms=220)
        else:
            raise ValueError(f"Unknown action id: {action}")

    # --------- публичный API ---------

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if self._emu is not None:
            self._emu.close()
            self._emu = None
        emu = self._ensure_emu()
        self._skip_intro(emu)
        obs = self._grab_obs(emu)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int):
        emu = self._ensure_emu()
        self._apply_action(emu, int(action))
        obs = self._grab_obs(emu)
        reward = 0.0  # пока заглушка для демо‑записей
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._emu is not None:
            self._emu.close()
            self._emu = None

