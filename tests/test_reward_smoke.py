"""
Smoke test: 500 шагов с RewardPolicy v1, проверка разбивки наград и отсутствия NaNs.
Запуск без эмулятора: тест только на моках (импорты + вызов policy.compute с фейковыми obs).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from msx_env.reward import RewardPolicy, default_v1_config
from msx_env.reward.logger import EpisodeRewardLogger


def test_reward_policy_smoke() -> None:
    cfg = default_v1_config()
    policy = RewardPolicy(cfg)
    logger = EpisodeRewardLogger()

    policy.reset()
    logger.reset()

    prev_obs = np.zeros((84, 84), dtype=np.uint8)
    info: dict = {}

    for step in range(500):
        obs = prev_obs + (step % 10)
        obs = np.clip(obs, 0, 255).astype(np.uint8)
        terminated = False
        truncated = step == 499
        terminated_by_death = False
        rgb = np.stack([obs] * 3, axis=-1)

        total, components, extra = policy.compute(
            prev_obs,
            obs,
            info,
            terminated,
            truncated,
            terminated_by_death,
            rgb_for_hud=rgb,
        )

        assert not np.isnan(total), f"NaN total at step {step}"
        for k, v in components.items():
            assert not np.isnan(v), f"NaN component {k} at step {step}"
        assert isinstance(total, (int, float)), type(total)

        logger.add_step(total, components)
        prev_obs = obs.copy()

    summary = logger.get_summary()
    assert "total" in summary
    assert "steps" in summary
    assert summary["steps"] == 500.0
    assert not np.isnan(summary["total"])
    assert summary["step"] <= 0
    print("Reward breakdown (last step):", components)
    print("Episode summary:", summary)
    print("Smoke test passed: no NaNs, reasonable scales.")


if __name__ == "__main__":
    test_reward_policy_smoke()
