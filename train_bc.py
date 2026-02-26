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
from msx_env.bc_model import BCNet, BCNetDeep, FRAME_STACK


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


def build_stacked_obs(obs: np.ndarray, stack_size: int) -> np.ndarray:
    """По однокадровым obs (N, H, W) собрать стопки (N, stack_size, H, W). Первые шаги дополняются первым кадром."""
    n, h, w = obs.shape
    out = np.zeros((n, stack_size, h, w), dtype=obs.dtype)
    for t in range(n):
        for s in range(stack_size):
            src = max(0, t - (stack_size - 1) + s)
            out[t, s] = obs[src]
    return out


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
    p.add_argument("--frame-stack", type=int, default=FRAME_STACK, help="число кадров в стопке (4 — рекомендуется)")
    p.add_argument("--deep", action="store_true", help="усиленная сеть (BCNetDeep): лучше прыжки/удары/свечки")
    p.add_argument("--rare-weight", type=float, default=1.5, help="доп. вес редких действий (ATTACK, прыжки) в loss")
    p.add_argument(
        "--move-weight",
        type=float,
        default=1.6,
        help="вес движения RIGHT/LEFT/UP/DOWN в loss; приоритет «идти по лабиринту (в т.ч. по лестницам)», а не стоять и бить",
    )
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

    stack_size = max(1, int(args.frame_stack))
    if stack_size > 1:
        obs_stacked = build_stacked_obs(obs, stack_size)  # (N, stack_size, 84, 84)
        print(f"Frame stack: {stack_size} кадров")
    else:
        obs_stacked = obs[:, np.newaxis, :, :]  # (N, 1, 84, 84)

    device = torch.device(args.device)
    dataset = TensorDataset(
        torch.from_numpy(obs_stacked),
        torch.from_numpy(actions),
    )
    # Oversampling: движение (RIGHT/LEFT/UP/DOWN) — чаще всего; редкие (прыжки, удар) — чаще; NOOP — реже
    rare_actions = {5, 6, 7, 8, 9}  # ATTACK, jump*, jump+attack
    move_actions = {1, 2, 3, 4}  # RIGHT, LEFT, UP, DOWN — приоритет: идти к выходу/ключу, ходить по лестницам, подбирать призы
    sample_w = np.ones(n, dtype=np.float64)
    sample_w[actions == 0] = 1.0
    sample_w[np.isin(actions, list(move_actions))] = float(args.oversample) * 1.4  # движение важнее
    sample_w[(actions != 0) & np.isin(actions, list(rare_actions)) & ~np.isin(actions, list(move_actions))] = float(args.oversample) * 1.2
    sample_w[(actions != 0) & ~np.isin(actions, list(rare_actions)) & ~np.isin(actions, list(move_actions))] = float(args.oversample)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )

    # Взвешенный loss: NOOP слабее; движение (RIGHT/LEFT) сильнее всего; редкие (ATTACK, прыжки) сильнее
    class_weights = torch.ones(NUM_ACTIONS, device=device)
    class_weights[0] = args.noop_weight
    for a in move_actions:
        if a < NUM_ACTIONS:
            class_weights[a] = args.move_weight
    for a in rare_actions:
        if a < NUM_ACTIONS:
            class_weights[a] = args.rare_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model_cls = BCNetDeep if args.deep else BCNet
    model = model_cls(num_actions=NUM_ACTIONS, in_channels=stack_size).to(device)
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

        ckpt = {
            "state_dict": model.state_dict(),
            "frame_stack": stack_size,
            "arch": "deep" if args.deep else "default",
        }
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
        torch.save(ckpt, ckpt_dir / "last.pt")

    print(f"Чекпоинты сохранены в {ckpt_dir}: best.pt, last.pt (frame_stack={stack_size}, arch={'deep' if args.deep else 'default'})")
    print(f"Параметры: noop_weight={args.noop_weight}, oversample={args.oversample}, move_weight={args.move_weight}, rare_weight={args.rare_weight}, epochs={args.epochs}")


if __name__ == "__main__":
    main()
