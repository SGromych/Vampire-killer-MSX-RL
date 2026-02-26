"""
Запуск обученной BC-политики в окружении Vampire Killer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import EnvConfig, VampireKillerEnv
from msx_env.bc_model import load_bc_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Тест BC-политики в Vampire Killer")
    p.add_argument("--checkpoint", type=str, default="checkpoints/bc/best.pt")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--workdir", type=str, default=None, help="workdir для openMSX; по умолчанию checkpoints/bc/run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    device = torch.device(args.device)
    model = load_bc_checkpoint(ckpt_path, device=device)

    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        raise FileNotFoundError(f"ROM не найден: {rom}")

    workdir = args.workdir or str(ROOT / "checkpoints" / "bc" / "run")
    env = VampireKillerEnv(
        EnvConfig(rom_path=str(rom), workdir=workdir, frame_size=(84, 84))
    )

    try:
        obs, info = env.reset()
        step = 0
        while step < args.max_steps:
            x = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                action = logits.argmax(dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if step % 100 == 0:
                print(f"Step {step}  action={action}")
            if terminated or truncated:
                print("Episode ended.")
                break
        print(f"Итого шагов: {step}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
