"""
Behavior Cloning: обучение политики на записанных демонстрациях.
Использует все прогоны из demos/runs/ или указанные --runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Корень проекта в path для msx_env
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.dataset import load_demo_run
from msx_env.env import NUM_ACTIONS
from msx_env.bc_model import BCNet


def discover_runs(demos_runs: Path) -> list[Path]:
    """Найти все каталоги с data.npz в demos/runs/."""
    if not demos_runs.exists():
        return []
    out = []
    for d in demos_runs.iterdir():
        if d.is_dir() and (d / "data.npz").exists():
            out.append(d)
    return sorted(out)


def load_all_demos(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Склеить obs и actions из нескольких прогонов."""
    all_obs, all_actions = [], []
    for run_dir in run_dirs:
        obs, actions, _, _, _, _, _ = load_demo_run(run_dir)
        all_obs.append(obs)
        all_actions.append(actions)
    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    return obs, actions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavior Cloning на демонстрациях Vampire Killer")
    p.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="run_id через пробел (например run_full_01 run_full_02). По умолчанию — все прогоны в demos/runs/",
    )
    p.add_argument("--epochs", type=int, default=40, help="больше эпох — лучше выучивает редкие действия")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/bc")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Смещение в сторону действий (меньше NOOP в предсказаниях)
    p.add_argument("--noop-weight", type=float, default=0.5, help="вес класса NOOP в loss (меньше 1 = реже предсказывать NOOP)")
    p.add_argument("--oversample", type=float, default=2.0, help="во сколько раз чаще сэмплировать шаги с action!=NOOP")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    demos_runs = ROOT / "demos" / "runs"

    if args.runs:
        run_dirs = [demos_runs / rid for rid in args.runs]
        missing = [r for r in run_dirs if not (r / "data.npz").exists()]
        if missing:
            raise FileNotFoundError(f"Нет data.npz в: {[str(m) for m in missing]}")
    else:
        run_dirs = discover_runs(demos_runs)
        if not run_dirs:
            raise FileNotFoundError(
                f"Нет прогонов в {demos_runs}. Запишите демо: python demos/record_demo.py --run-id run_full_01 ..."
            )

    print(f"Загрузка прогонов: {[d.name for d in run_dirs]}")
    obs, actions = load_all_demos(run_dirs)
    obs = obs.astype(np.float32) / 255.0  # [0,1]
    actions = actions.astype(np.int64)
    n = len(actions)
    n_noop = (actions == 0).sum()
    n_action = n - n_noop
    print(f"Всего сэмплов: {n}  (NOOP: {n_noop}, действия: {n_action})")

    device = torch.device(args.device)
    dataset = TensorDataset(
        torch.from_numpy(obs).unsqueeze(1),  # (N, 1, 84, 84)
        torch.from_numpy(actions),
    )
    # Oversampling: чаще подаём примеры с action != NOOP
    weights = np.where(actions == 0, 1.0, float(args.oversample))
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )

    # Взвешенный loss: за ошибки по ненулевым действиям штрафуем сильнее
    class_weights = torch.ones(NUM_ACTIONS, device=device)
    class_weights[0] = args.noop_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = BCNet(num_actions=NUM_ACTIONS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = ROOT / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

    print(f"Чекпоинты сохранены в {ckpt_dir}: best.pt, last.pt")
    print(f"Параметры: noop_weight={args.noop_weight}, oversample={args.oversample}, epochs={args.epochs}")


if __name__ == "__main__":
    main()
