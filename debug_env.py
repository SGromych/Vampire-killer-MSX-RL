"""
DRY run env с принудительным действием (например RIGHT) для проверки:
- есть ли движение (frame_diff),
- меняется ли room_hash,
- растёт ли unique_rooms.

Без обучения: только env.step(forced_action) в цикле. Вывод [debug] и [debug EPISODE] из env.

Пример:
  python debug_env.py --capture-backend window --debug --debug-force-action RIGHT --debug-episode-max-steps 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import EnvConfig, VampireKillerEnv, ACTION_RIGHT
from msx_env.reward import default_v1_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug env: forced action, room_hash, stage, stuck (no training)")
    p.add_argument("--capture-backend", choices=["png", "single", "window"], default="png")
    p.add_argument("--workdir", type=str, default=None)
    p.add_argument("--debug", action="store_true", default=True, help="включить [debug] вывод (по умолчанию вкл)")
    p.add_argument("--debug-force-action", type=str, default="RIGHT", help="действие: RIGHT, LEFT, ...")
    p.add_argument("--debug-episode-max-steps", type=int, default=400)
    p.add_argument("--debug-every", type=int, default=10)
    p.add_argument("--debug-dump-frames", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        raise FileNotFoundError(f"ROM не найден: {rom}")

    workdir = args.workdir or str(ROOT / "checkpoints" / "ppo" / "run")
    Path(workdir).mkdir(parents=True, exist_ok=True)

    cfg = EnvConfig(
        rom_path=str(rom),
        workdir=workdir,
        frame_size=(84, 84),
        terminated_on_death=True,
        max_episode_steps=args.debug_episode_max_steps,
        capture_backend=args.capture_backend,
        reward_config=default_v1_config(),
        debug=args.debug,
        debug_every=args.debug_every,
        debug_episode_max_steps=args.debug_episode_max_steps,
        debug_dump_frames=args.debug_dump_frames,
        debug_force_action=args.debug_force_action or "RIGHT",
        debug_dump_dir=workdir,
    )

    env = VampireKillerEnv(cfg)
    forced_action = ACTION_RIGHT
    if args.debug_force_action:
        from msx_env.env import _debug_force_action_to_id
        forced_action = _debug_force_action_to_id(args.debug_force_action.strip().upper(), ACTION_RIGHT)

    print(f"Debug env: forced_action={args.debug_force_action} (id={forced_action}) max_steps={args.debug_episode_max_steps}")
    print("Run until done or max_steps; watch [debug] and [debug EPISODE] for room_hash, unique_rooms, frame_diff.\n")

    obs, _ = env.reset()
    steps = 0
    while steps < args.debug_episode_max_steps:
        obs, reward, terminated, truncated, info = env.step(forced_action)
        steps += 1
        if terminated or truncated:
            break

    env.close()
    print(f"\nDone: {steps} steps. Check room_changes and unique_rooms in [debug EPISODE] above.")


if __name__ == "__main__":
    main()
