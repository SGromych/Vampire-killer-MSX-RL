"""
Фабрика env для параллельного запуска (несколько openMSX без коллизий).

Каждый инстанс получает уникальный workdir = tmp_root/rank и instance_id = rank,
чтобы команды и скриншоты не пересекались между процессами openMSX.
Используется в train_ppo при --num-envs > 1: [make_env(i, base_cfg)() for i in range(num_envs)].
При window capture и num_envs>1 можно передать per_env_window: {"0": {"title": "...", "crop": [x,y,w,h]}, ...}.
Подробнее: docs/MODULES_AND_FLAGS.md (мульти-инстанс), docs/TRAINING.md (эксперименты PPO).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict

from msx_env.env import EnvConfig, VampireKillerEnv


def make_env(
    rank: int,
    base_cfg: EnvConfig,
    per_env_window: Dict[str, Dict[str, Any]] | None = None,
) -> Callable[[], VampireKillerEnv]:
    """
    Возвращает фабрику (callable без аргументов), создающую VampireKillerEnv
    с workdir = base_cfg.tmp_root / str(rank) и instance_id = rank.
    Если per_env_window задан и содержит ключ str(rank), подставляются window_crop и window_title для этого env.
    """
    workdir = Path(base_cfg.tmp_root) / str(rank)
    workdir.mkdir(parents=True, exist_ok=True)
    kwargs = {"workdir": str(workdir.resolve()), "instance_id": rank}
    if per_env_window and str(rank) in per_env_window:
        entry = per_env_window[str(rank)]
        if "crop" in entry:
            kwargs["window_crop"] = tuple(entry["crop"])
        if "title" in entry:
            kwargs["window_title"] = entry["title"]
    cfg = replace(base_cfg, **kwargs)
    return lambda: VampireKillerEnv(cfg)
