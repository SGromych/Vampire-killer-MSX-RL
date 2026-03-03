"""
Запуск обученной PPO-политики в Vampire Killer.
Аналогично test_policy.py (BC), но для чекпоинтов PPO (ActorCritic).
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import EnvConfig, VampireKillerEnv
from msx_env.ppo_model import load_ppo_checkpoint
from msx_env.life_bar import get_life_estimate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Тест PPO-политики в Vampire Killer")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ppo/last.pt")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workdir", type=str, default=None)
    p.add_argument("--stop-on-death", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="argmax вместо sample")
    p.add_argument("--smooth", type=int, default=0)
    p.add_argument("--sticky", action="store_true")
    p.add_argument("--max-idle-steps", type=int, default=0)
    p.add_argument("--transition-assist", action="store_true")
    p.add_argument("--stair-assist-steps", type=int, default=0, help="при right-left цикле пробовать UP")
    p.add_argument("--capture-backend", choices=["png", "single"], default="png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    device = torch.device(args.device)
    model = load_ppo_checkpoint(ckpt_path, device=device)
    stack_size = model.in_channels

    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        raise FileNotFoundError(f"ROM не найден: {rom}")

    workdir = args.workdir or str(ROOT / "checkpoints" / "ppo" / "run")
    env = VampireKillerEnv(
        EnvConfig(
            rom_path=str(rom),
            workdir=workdir,
            frame_size=(84, 84),
            capture_backend=getattr(args, "capture_backend", "png"),
        )
    )

    frame_buffer: deque[np.ndarray] = deque(maxlen=stack_size)

    try:
        obs, info = env.reset()
        for _ in range(stack_size):
            frame_buffer.append(obs.copy())

        step = 0
        life_prev = get_life_estimate(obs)
        death_warmup_steps = 300
        action_history: deque[int] = deque(maxlen=args.smooth) if args.smooth else deque()
        last_non_noop: int | None = None
        last_move_action: int | None = None
        idle_steps = 0
        transition_cooldown = 0
        horizontal_only_steps = 0
        stair_assist_cooldown = 0
        hidden = model.zero_hidden(1, device) if getattr(model, "recurrent", False) else None

        while step < args.max_steps:
            stack = np.stack(list(frame_buffer), axis=0).astype(np.float32) / 255.0
            x = torch.from_numpy(stack).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, next_hidden = model.get_action(x, deterministic=args.deterministic, hidden=hidden)
            if next_hidden is not None:
                hidden = next_hidden

            if args.smooth > 0:
                action_history.append(action)
                if len(action_history) == args.smooth:
                    counts = np.bincount(list(action_history), minlength=10)
                    action = int(np.argmax(counts))
            if args.sticky and action == 0 and last_non_noop is not None:
                action = last_non_noop
            if args.transition_assist and transition_cooldown > 0 and action == 0 and last_non_noop is not None:
                action = last_non_noop
            if transition_cooldown > 0:
                transition_cooldown -= 1
            if action != 0:
                last_non_noop = action

            move_actions = {1, 2, 3, 4}
            if action in move_actions:
                idle_steps = 0
                last_move_action = action
            else:
                idle_steps += 1
            if args.max_idle_steps > 0 and idle_steps >= args.max_idle_steps:
                action = last_move_action if last_move_action is not None else 1
                idle_steps = 0

            if args.stair_assist_steps > 0:
                if action in {1, 2}:
                    horizontal_only_steps += 1
                elif action in {3, 4}:
                    horizontal_only_steps = 0
                else:
                    horizontal_only_steps = 0
                if (
                    stair_assist_cooldown <= 0
                    and horizontal_only_steps >= args.stair_assist_steps
                ):
                    action = 3
                    horizontal_only_steps = 0
                    stair_assist_cooldown = 60
                if stair_assist_cooldown > 0:
                    stair_assist_cooldown -= 1

            obs, reward, terminated, truncated, info = env.step(action)
            frame_buffer.append(obs.copy())
            step += 1

            if args.transition_assist and len(frame_buffer) >= 2:
                curr = frame_buffer[-1].astype(np.float32)
                prev = frame_buffer[-2].astype(np.float32)
                diff = np.abs(curr - prev).mean() / 255.0
                has_key = bool(info.get("hud", {}).get("key_door") or info.get("hud", {}).get("key_chest"))
                thr, cooldown = (0.28, 4) if has_key else (0.35, 2)
                if diff > thr:
                    transition_cooldown = max(transition_cooldown, cooldown)

            life = get_life_estimate(obs)
            if (
                args.stop_on_death
                and step > death_warmup_steps
                and life < 0.15
                and life_prev > 0.3
            ):
                print(f"Step {step}  смерть")
                break
            life_prev = life

            if step % 100 == 0:
                print(f"Step {step}  action={action}  life≈{life:.2f}  reward={reward:.2f}")
            if terminated or truncated:
                print("Episode ended.")
                break

        print(f"Итого шагов: {step}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
