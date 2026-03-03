"""
Загрузка конфига эксперимента из JSON. CLI переопределяет значения из файла.
Используется в train_ppo при --config <path>.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Загрузить JSON-конфиг. Ключи: lr, entropy_coef, value_loss_coef, clip_eps, rollout_steps, ppo_epochs, batch_size, ..."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_config_to_args(args: Any, config: dict[str, Any]) -> None:
    """Применить конфиг к namespace args (только существующие атрибуты)."""
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
