#!/usr/bin/env python3
"""
Acceptance test для episode-metrics-fix (PR episode-metrics-fix).
Запускает env на 1–2 эпизода с scripted RIGHT и выводит эпизодные метрики.
Убедиться, что stage00_exit_rate_ep > 0, если агент реально выходит из Stage00.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import ACTION_RIGHT, EnvConfig, VampireKillerEnv, NUM_ACTIONS
from msx_env.reward import default_v1_config
import numpy as np


def main() -> None:
    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        print("VAMPIRE.ROM not found, skipping test")
        return

    cfg = EnvConfig(
        rom_path=str(rom),
        workdir=str((ROOT / "checkpoints" / "ppo" / "run").resolve()),
        frame_size=(84, 84),
        terminated_on_death=True,
        max_episode_steps=500,
        reward_config=default_v1_config(),
        soft_reset=True,
        quiet=True,
    )
    env = VampireKillerEnv(cfg)

    print("=== Episode metrics acceptance test ===\n")
    print("Running 1–2 episodes with scripted RIGHT...\n")

    obs, _ = env.reset()
    episode = 0
    max_episodes = 2

    while episode < max_episodes:
        obs, reward, terminated, truncated, info = env.step(ACTION_RIGHT)
        done = terminated or truncated

        if done:
            episode += 1
            ur_ep = info.get("reward_unique_rooms_ep", 0)
            trans_ep = info.get("reward_room_transitions_ep", 0)
            dwell_ep = info.get("reward_room_dwell_steps_ep_mean", -1.0)
            s00_recorded = info.get("reward_stage00_exit_recorded_ep", 0)
            s00_steps_ep = info.get("reward_stage00_exit_steps_ep", -1)
            backt_ep = info.get("reward_backtrack_rate_ep", 0.0)
            stable_hash = info.get("reward_stable_room_hash_ep", "")

            print(f"--- Episode {episode} finished ---")
            print(f"  unique_rooms_ep:        {ur_ep}")
            print(f"  room_transitions_ep:    {trans_ep}")
            print(f"  room_dwell_steps_ep_mean: {dwell_ep:.1f}")
            print(f"  stage00_exit_recorded_ep: {s00_recorded} (1=yes)")
            print(f"  stage00_exit_steps_ep:  {s00_steps_ep}")
            print(f"  backtrack_rate_ep:      {backt_ep:.4f}")
            print(f"  stable_room_hash_ep:    {(stable_hash or '-')[:16]}...")
            print()

            obs, _ = env.reset()

    env.close()
    print("Done. Check stage00_exit_recorded_ep=1 if agent reached Stage 01.")


if __name__ == "__main__":
    main()
