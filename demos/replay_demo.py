import argparse
from pathlib import Path
import sys

import pygame

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.dataset import load_demo_run
from msx_env.env import EnvConfig, VampireKillerEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay recorded Vampire Killer demo.")
    p.add_argument("run_id", help="имя подкаталога в demos/runs/")
    p.add_argument("--mode", choices=["frames", "env"], default="frames")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def replay_frames(base_dir: Path, run_dir: Path, fps: int) -> None:
    obs, actions, rewards, next_obs, dones, timestamps, manifest = load_demo_run(run_dir)
    if len(obs) == 0:
        print("No frames to replay.")
        return

    pygame.init()
    h, w = obs[0].shape[:2]
    screen = pygame.display.set_mode((w * 3, h * 3))
    clock = pygame.time.Clock()

    for i, frame in enumerate(obs):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        img = frame
        if img.ndim == 2:
            surf = pygame.surfarray.make_surface(
                (img.repeat(3).reshape(h, w, 3)).swapaxes(0, 1)
            )
        else:
            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (w * 3, h * 3))
        screen.blit(surf, (0, 0))
        pygame.display.set_caption(f"Replay frame {i+1}/{len(obs)}")
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def replay_env(base_dir: Path, run_dir: Path, fps: int) -> None:
    obs, actions, rewards, next_obs, dones, timestamps, manifest = load_demo_run(run_dir)
    rom = base_dir / "VAMPIRE.ROM"
    # Для env‑replay логично тоже использовать отдельный workdir под ран
    env = VampireKillerEnv(
        EnvConfig(
            rom_path=str(rom),
            workdir=str(run_dir),
        )
    )
    try:
        obs_env, info = env.reset()
        for a in actions:
            obs_env, r, terminated, truncated, info = env.step(int(a))
            if terminated or truncated:
                break
    finally:
        env.close()
    print("Env replay finished.")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    run_dir = base_dir / "demos" / "runs" / args.run_id
    if args.mode == "frames":
        replay_frames(base_dir, run_dir, fps=args.fps)
    else:
        replay_env(base_dir, run_dir, fps=args.fps)


if __name__ == "__main__":
    main()

