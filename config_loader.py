"""
Thin wrapper around the central project configuration.

- Provides a single place to load the night training config used by the supervisor.
- Keeps defaults for night_training.json close to code and reusable from tools/tests.

The heavy lifting (resolved PPO/env/reward config, snapshots, validation) lives in
`project_config.py`. This module only deals with the small night supervisor JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
NIGHT_TRAINING_PATH = ROOT / "configs" / "night_training.json"

# Defaults mirror train_supervisor semantics and are merged with night_training.json.
NIGHT_DEFAULTS: Dict[str, Any] = {
    "bc_checkpoint": None,
    "num_envs": 1,
    "max_updates": 5000,
    "checkpoint_every": 50,
    "restart_limit": 20,
    "watchdog_timeout_minutes": 60,
    "entropy_floor": 0.3,
    "auto_entropy_boost": 0.0,
    "run_name": "auto_night",
    "checkpoint_dir": "checkpoints/ppo",
    "log_dir": None,
    "restart_delay_seconds": 30,
    "nudge_right_steps": 0,
    "stuck_nudge_steps": 0,
    "novelty_reward": 0.35,
    "recurrent": True,
    # PPO‑related knobs that must be in sync with train_ppo defaults
    "rollout_steps": 256,
    "entropy_coef": 0.02,
    "action_repeat": 2,
}


def load_night_training_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load configs/night_training.json (if present), merge with NIGHT_DEFAULTS and return a dict.

    - Unknown keys from JSON are preserved.
    - Missing keys are filled from NIGHT_DEFAULTS.
    """
    if path is None:
        p = NIGHT_TRAINING_PATH
    else:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p

    cfg: Dict[str, Any] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"night_training config must be a JSON object, got {type(loaded)!r}")
        cfg.update(loaded)

    merged = NIGHT_DEFAULTS.copy()
    merged.update(cfg)
    return merged

