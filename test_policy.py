"""
Запуск обученной BC-политики в окружении Vampire Killer.
Поддерживается frame stacking: в буфере хранятся последние N кадров.
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
from msx_env.bc_model import load_bc_checkpoint
from msx_env.life_bar import get_life_estimate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Тест BC-политики в Vampire Killer")
    p.add_argument("--checkpoint", type=str, default="checkpoints/bc/best.pt")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workdir", type=str, default=None, help="workdir для openMSX; по умолчанию checkpoints/bc/run")
    p.add_argument("--stop-on-death", action="store_true", help="завершить прогон при падении полоски жизни (смерть)")
    p.add_argument("--smooth", type=int, default=0, metavar="N", help="сглаживание: majority vote по последним N действиям (0=выкл)")
    p.add_argument("--sticky", action="store_true", help="при NOOP один раз повторить последнее ненулевое действие (меньше замираний)")
    p.add_argument(
        "--max-idle-steps",
        type=int,
        default=0,
        help="анти-залипание: если N шагов подряд нет движения (RIGHT/LEFT/UP/DOWN), принудительно повторить последнее движение; 0=выкл, 40–50 рекомендуется",
    )
    p.add_argument(
        "--transition-assist",
        action="store_true",
        help="при резкой смене кадра (переход между экранами) при NOOP повторять последнее движение 1–2 шага — меньше глюков у дверей",
    )
    p.add_argument(
        "--stair-assist-steps",
        type=int,
        default=0,
        help="если N шагов подряд только RIGHT/LEFT (без UP/DOWN) — попробовать UP (лестница); 0=выкл, 40–60 рекомендуется",
    )
    p.add_argument("--capture-backend", choices=["png", "single"], default="png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    device = torch.device(args.device)
    model = load_bc_checkpoint(ckpt_path, device=device)
    stack_size = model.in_channels

    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        raise FileNotFoundError(f"ROM не найден: {rom}")

    workdir = args.workdir or str(ROOT / "checkpoints" / "bc" / "run")
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
        # Заполнить буфер первым кадром (как при обучении: первые шаги дублируют кадр)
        for _ in range(stack_size):
            frame_buffer.append(obs.copy())

        step = 0
        life_prev = get_life_estimate(obs)
        death_warmup_steps = 300  # первые шаги: заставки/переходы, игнорируем падения жизни
        action_history: deque[int] = deque(maxlen=args.smooth) if args.smooth else deque()
        last_non_noop: int | None = None
        last_move_action: int | None = None
        idle_steps = 0
        transition_cooldown = 0
        horizontal_only_steps = 0  # только RIGHT/LEFT без UP/DOWN
        stair_assist_cooldown = 0  # после попытки UP — пауза

        while step < args.max_steps:
            stack = np.stack(list(frame_buffer), axis=0).astype(np.float32) / 255.0  # (stack_size, 84, 84)
            x = torch.from_numpy(stack).unsqueeze(0).to(device)  # (1, stack_size, 84, 84)
            with torch.no_grad():
                logits = model(x)
                action = logits.argmax(dim=1).item()

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

            # Анти-залипание: если давно не двигались (RIGHT/LEFT/UP/DOWN), принудительно повторить последнее движение
            move_actions = {1, 2, 3, 4}
            if action in move_actions:
                idle_steps = 0
                last_move_action = action
            else:
                idle_steps += 1
            if args.max_idle_steps > 0 and idle_steps >= args.max_idle_steps:
                if last_move_action is not None:
                    action = last_move_action
                else:
                    action = 1  # по умолчанию пойдём вправо
                idle_steps = 0

            # Лестничная помощь: бесконечный right-left — попробовать UP
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
                    action = 3  # UP — попробовать лестницу
                    horizontal_only_steps = 0
                    stair_assist_cooldown = 60
                if stair_assist_cooldown > 0:
                    stair_assist_cooldown -= 1

            obs, reward, terminated, truncated, info = env.step(action)
            frame_buffer.append(obs.copy())
            step += 1

            # Детекция перехода между экранами: резкое изменение кадра → продолжать движение
            # При наличии ключа — агрессивнее (вероятно ищем выход)
            if args.transition_assist and len(frame_buffer) >= 2:
                curr, prev = frame_buffer[-1].astype(np.float32), frame_buffer[-2].astype(np.float32)
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
                print(f"Step {step}  смерть (полоска жизни упала)")
                break
            life_prev = life

            if step % 100 == 0:
                print(f"Step {step}  action={action}  life≈{life:.2f}")
            if terminated or truncated:
                print("Episode ended.")
                break
        opts = []
        if args.smooth:
            opts.append(f"smooth={args.smooth}")
        if args.sticky:
            opts.append("sticky")
        if args.max_idle_steps:
            opts.append(f"anti-stall={args.max_idle_steps}")
        if args.transition_assist:
            opts.append("transition-assist")
        if args.stair_assist_steps:
            opts.append(f"stair-assist={args.stair_assist_steps}")
        print(f"Итого шагов: {step}" + (f"  [{', '.join(opts)}]" if opts else ""))
    finally:
        env.close()


if __name__ == "__main__":
    main()
