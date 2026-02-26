"""
Простая CNN для Behavior Cloning: obs (84, 84) grayscale -> logits по NUM_ACTIONS.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .env import NUM_ACTIONS


class BCNet(nn.Module):
    """CNN: 1x84x84 -> 10 классов действий."""

    def __init__(self, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        # 64 * 7 * 7 после conv по 84x84 (примерно)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self._num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) или (B, H, W) -> (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)

    @property
    def num_actions(self) -> int:
        return self._num_actions


def load_bc_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> BCNet:
    """Загрузить модель из чекпоинта (state_dict)."""
    path = Path(path)
    model = BCNet(num_actions=NUM_ACTIONS)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)
