"""
CNN для Behavior Cloning: obs (84, 84) grayscale, с опциональным frame stacking.
Ввод: (B, C, 84, 84), C=1 (один кадр) или C=4 (стопка из 4 кадров).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .env import NUM_ACTIONS

# По умолчанию 4 кадра для временного контекста (переходы между экранами, движение)
FRAME_STACK = 4


class BCNet(nn.Module):
    """CNN: Cx84x84 -> 10 классов действий. C=1 или FRAME_STACK (4)."""

    def __init__(self, num_actions: int = NUM_ACTIONS, in_channels: int = FRAME_STACK):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self._num_actions = num_actions
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def in_channels(self) -> int:
        return self._in_channels


class BCNetDeep(nn.Module):
    """Усиленная CNN: больше каналов и слой — лучше выучивает прыжки, удары, свечки."""

    def __init__(self, num_actions: int = NUM_ACTIONS, in_channels: int = FRAME_STACK):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
        )
        # 84 -> 20 -> 9 -> 7 -> 5
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_actions),
        )
        self._num_actions = num_actions
        self._in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) или (B, H, W) -> (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def in_channels(self) -> int:
        return self._in_channels


def load_bc_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    frame_stack: int | None = None,
) -> BCNet | BCNetDeep:
    """
    Загрузить модель из чекпоинта.
    Чекпоинт может быть: state_dict или dict с ключами state_dict, frame_stack, arch.
    arch=="deep" -> BCNetDeep, иначе BCNet.
    """
    path = Path(path)
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
        in_ch = raw.get("frame_stack", FRAME_STACK)
        arch = raw.get("arch", "default")
    else:
        state = raw
        in_ch = 1
        arch = "default"
    if frame_stack is not None:
        in_ch = frame_stack
    model_cls = BCNetDeep if arch == "deep" else BCNet
    model = model_cls(num_actions=NUM_ACTIONS, in_channels=in_ch)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
