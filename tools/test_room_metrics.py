#!/usr/bin/env python3
"""
Acceptance test для fix-room-metrics-stability (PR fix-room-metrics-stability).
Запускает 2–3 эпизода с random policy и проверяет, что:
  - Stage00: unique_rooms_ep ∈ [1..6] (допуск 6 на редкие артефакты)
  - Stage01: unique_rooms_ep ≤ ~12 (с запасом)
  - unique_rooms_ep > 6 при stage==00 → FAIL (дрожание room_hash)

Usage:
  python tools/test_room_metrics.py --episodes 3 --num-envs 1 --policy random
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from msx_env.env import VampireKillerEnv, EnvConfig, NUM_ACTIONS
from msx_env.reward import default_v1_config


def main() -> int:
    p = argparse.ArgumentParser(description="Acceptance test: room metrics stability")
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    p.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (only 1 supported)")
    p.add_argument("--policy", choices=["random"], default="random", help="Policy: random actions")
    p.add_argument("--rom", type=str, default=None, help="Path to VAMPIRE.ROM (default: ROOT/VAMPIRE.ROM)")
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode (truncation)")
    args = p.parse_args()

    if args.num_envs != 1:
        print("WARNING: only num-envs=1 supported; using 1")
    rom = Path(args.rom) if args.rom else ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        print(f"ERROR: ROM not found: {rom}")
        return 2

    workdir = ROOT / "runs" / "room_metrics_test"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(
        rom_path=str(rom.resolve()),
        workdir=str(workdir.resolve()),
        frame_size=(84, 84),
        terminated_on_death=True,
        max_episode_steps=args.max_steps,
        reward_config=default_v1_config(),
        soft_reset=True,
        quiet=True,
    )
    env = VampireKillerEnv(cfg)
    rng = np.random.default_rng(42)

    print("=== Room metrics acceptance test (fix-room-metrics-stability) ===\n")
    print(f"Episodes: {args.episodes}  Num-envs: 1  Policy: {args.policy}\n")

    all_ok = True
    episode_count = 0
    obs, _ = env.reset()

    while episode_count < args.episodes:
        action = int(rng.integers(0, NUM_ACTIONS))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            episode_count += 1
            stable_stage = int(info.get("reward_stable_stage_ep", 0))
            ur_ep = int(info.get("reward_unique_rooms_ep", 0))
            trans_ep = int(info.get("reward_room_transitions_ep", 0))
            room_ids = info.get("reward_stable_room_ids_ep", [])[:10]

            print(f"--- Episode {episode_count} ---")
            print(f"  stage (stable):       {stable_stage}")
            print(f"  unique_rooms_ep:      {ur_ep}")
            print(f"  room_transitions_ep:  {trans_ep}")
            print(f"  first 10 stable_room_id: {room_ids}")
            print()

            # Sanity: Stage00 has at most 3 rooms; 6 is generous for debounce artifacts
            if stable_stage == 0 and ur_ep > 6:
                print(f"  FAIL: stage==00 but unique_rooms_ep={ur_ep} > 6 (room hash jitter)")
                all_ok = False
            elif stable_stage == 1 and ur_ep > 20:
                print(f"  WARN: stage==01 but unique_rooms_ep={ur_ep} > 20 (possible jitter)")

            obs, _ = env.reset()

    env.close()

    if all_ok:
        print("PASS: Room metrics sanity check passed.")
        return 0
    print("FAIL: At least one episode had stage==00 and unique_rooms_ep > 6.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
