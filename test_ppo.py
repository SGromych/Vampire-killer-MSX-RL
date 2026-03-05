"""
Запуск обученной PPO-политики в Vampire Killer.
Аналогично test_policy.py (BC), но для чекпоинтов PPO (ActorCritic).
"""
from __future__ import annotations

import argparse
import io
import json
import hashlib
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import NUM_ACTIONS, EnvConfig, VampireKillerEnv
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
    p.add_argument(
        "--diagnose-policy",
        action="store_true",
        help="диагностика детерминированной/стохастической политики, без изменения обучения",
    )
    return p.parse_args()


def _load_raw_checkpoint(path: Path, device: torch.device) -> Dict[str, Any] | None:
    """
    Загрузка чекпоинта для диагностики.
    Логика намеренно зеркалит msx_env.ppo_model.load_ppo_checkpoint:
    - сначала пробуем weights_only=True (PyTorch 2.6+),
    - при ЛЮБОЙ ошибке загрузки fallback на weights_only=False (более либеральный режим).
    Использовать только для доверенных чекпоинтов из собственного обучения.
    """
    try:
        try:
            raw = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            # weights_only режим не смог аккуратно распаковать файл (WeightsUnpickler / safe globals и т.п.)
            # Переходим на полностью доверенный режим weights_only=False, как в старых версиях torch.load.
            raw = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING] Не удалось загрузить чекпоинт ({path}): {e}")
        return None

    if not isinstance(raw, dict):
        print(f"[WARNING] Неверный формат чекпоинта (ожидался dict): {path}")
        return None
    return raw


def _checkpoint_info(ckpt_path: Path, raw: Dict[str, Any] | None, model: torch.nn.Module) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(ckpt_path),
        "exists": ckpt_path.exists(),
        "file_size": ckpt_path.stat().st_size if ckpt_path.exists() else 0,
        "keys": sorted(list(raw.keys())) if isinstance(raw, dict) else [],
    }
    if isinstance(raw, dict):
        info["update"] = int(raw.get("update", -1))
        info["frame_stack"] = int(raw.get("frame_stack", getattr(model, "in_channels", 0)))
        info["arch"] = str(raw.get("arch", getattr(model, "arch", "unknown")))
        info["recurrent"] = bool(raw.get("recurrent", getattr(model, "recurrent", False)))
        info["lstm_hidden_size"] = int(raw.get("lstm_hidden_size", getattr(model, "_lstm_hidden_size", 0)))

        state = raw.get("state_dict")
        checksum = None
        if isinstance(state, dict):
            try:
                buf = io.BytesIO()
                torch.save(state, buf)
                data = buf.getvalue()[: 1_000_000]
                checksum = hashlib.sha1(data).hexdigest()
            except Exception as e:  # noqa: BLE001
                print(f"[WARNING] Не удалось посчитать checksum state_dict: {e}")
        info["state_dict_checksum_sha1_first_1mb"] = checksum

    total_params = sum(p.numel() for p in model.parameters())
    info["total_params"] = int(total_params)
    return info


def _find_config_snapshot(ckpt_path: Path) -> Path | None:
    # 1) В той же директории
    candidate = ckpt_path.parent / "config_snapshot.json"
    if candidate.exists():
        return candidate
    # 2) Для layout runs/<run>/checkpoints/ppo/last.pt -> runs/<run>/config_snapshot.json
    parent = ckpt_path.parent.parent
    candidate2 = parent / "config_snapshot.json"
    if candidate2.exists():
        return candidate2
    return None


def _load_config_snapshot(path: Path) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING] Не удалось прочитать config_snapshot.json ({path}): {e}")
        return None


def _config_parity(
    cfg_snapshot: Dict[str, Any] | None,
    ckpt_meta: Dict[str, Any],
    env: VampireKillerEnv,
    model: torch.nn.Module,
) -> Dict[str, Any]:
    parity: Dict[str, Any] = {"status": "NO_SNAPSHOT", "mismatches": [], "train": {}, "test": {}}
    if cfg_snapshot is None:
        return parity

    parity["status"] = "OK"

    # Train side (из snapshot + чекпоинта)
    train_cfg: Dict[str, Any] = {
        "frame_stack": int(ckpt_meta.get("frame_stack", 0)),
        "image_size": (84, 84),  # в train_ppo жестко зашито (84,84)
        "recurrent": bool(cfg_snapshot.get("recurrent", False)),
        "lstm_hidden_size": int(cfg_snapshot.get("lstm_hidden_size", 256)),
        "action_space_size": NUM_ACTIONS,
    }
    reward_cfg = cfg_snapshot.get("reward_config") or {}
    train_cfg["room_hash_crop_top"] = int(reward_cfg.get("room_hash_crop_top", 14))
    train_cfg["room_hash_crop_bottom"] = int(reward_cfg.get("room_hash_crop_bottom", 0))
    train_cfg["grayscale"] = True
    train_cfg["normalization"] = "obs/255.0"

    # Test side (из env + модели)
    test_frame_size = getattr(env.cfg, "frame_size", (84, 84))
    test_cfg: Dict[str, Any] = {
        "frame_stack": getattr(model, "in_channels", 0),
        "image_size": tuple(test_frame_size),
        "recurrent": getattr(model, "recurrent", False),
        "lstm_hidden_size": int(getattr(model, "_lstm_hidden_size", 256)) if getattr(model, "recurrent", False) else 0,
        "action_space_size": getattr(model, "num_actions", NUM_ACTIONS),
        "room_hash_crop_top": int(reward_cfg.get("room_hash_crop_top", 14)),
        "room_hash_crop_bottom": int(reward_cfg.get("room_hash_crop_bottom", 0)),
        "grayscale": True,
        "normalization": "obs/255.0",
    }

    parity["train"] = train_cfg
    parity["test"] = test_cfg

    def _check(key: str) -> None:
        if train_cfg.get(key) != test_cfg.get(key):
            parity["status"] = "MISMATCH"
            parity["mismatches"].append({"param": key, "train": train_cfg.get(key), "test": test_cfg.get(key)})

    for k in (
        "frame_stack",
        "image_size",
        "room_hash_crop_top",
        "room_hash_crop_bottom",
        "grayscale",
        "normalization",
        "recurrent",
        "lstm_hidden_size",
        "action_space_size",
    ):
        _check(k)

    return parity


def _run_policy_diagnostics(
    args: argparse.Namespace,
    ckpt_path: Path,
    model: torch.nn.Module,
    env: VampireKillerEnv,
    device: torch.device,
) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    diag_root = ROOT / "diagnostics" / f"policy_test_{timestamp}"
    obs_dir = diag_root / "obs_samples"
    diag_root.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)

    print(f"DIAGNOSTICS DIR: {diag_root}")

    # ------- CHECKPOINT INFO -------
    raw_ckpt = _load_raw_checkpoint(ckpt_path, device)
    ckpt_meta = _checkpoint_info(ckpt_path, raw_ckpt, model)
    print("\nCHECKPOINT INFO")
    print("---------------")
    print(f"path: {ckpt_meta['path']}")
    print(f"file_size: {ckpt_meta['file_size']} bytes")
    print(f"keys: {ckpt_meta.get('keys', [])}")
    print(f"update: {ckpt_meta.get('update', -1)}")
    print(f"arch: {ckpt_meta.get('arch')}, recurrent={ckpt_meta.get('recurrent')} lstm_hidden_size={ckpt_meta.get('lstm_hidden_size')}")
    print(f"frame_stack: {ckpt_meta.get('frame_stack')}")
    print(f"total_params: {ckpt_meta.get('total_params')}")
    print(f"policy_checksum_sha1_first_1mb: {ckpt_meta.get('state_dict_checksum_sha1_first_1mb')}")
    if not ckpt_path.exists() or raw_ckpt is None:
        print("[WARNING] Чекпоинт не найден или не загружен, диагностика прервана.")
        return

    # ------- CONFIG PARITY -------
    cfg_snapshot_path = _find_config_snapshot(ckpt_path)
    cfg_snapshot = _load_config_snapshot(cfg_snapshot_path) if cfg_snapshot_path else None
    parity = _config_parity(cfg_snapshot, ckpt_meta, env, model)
    print("\nCONFIG PARITY CHECK")
    print("-------------------")
    if cfg_snapshot_path is None:
        print("NO_SNAPSHOT (config_snapshot.json не найден рядом с чекпоинтом)")
    else:
        print(f"snapshot: {cfg_snapshot_path}")
        print(f"STATUS: {parity['status']}")
        if parity["status"] == "MISMATCH":
            print("MISMATCHED PARAMS:")
            for m in parity["mismatches"]:
                print(f"  - {m['param']}: train={m['train']}  test={m['test']}")

    # ------- ROLLOUT 300 STEPS -------
    max_diag_steps = 300
    stack_size = getattr(model, "in_channels", 1)
    frame_buffer: deque[np.ndarray] = deque(maxlen=stack_size)

    obs_stats: List[Dict[str, Any]] = []
    actions_det: List[int] = []
    actions_stoch: List[int] = []
    logits_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []
    entropy_list: List[float] = []
    top1_probs: List[float] = []
    noop_probs: List[float] = []
    hidden_norms: List[float] = []
    hidden_deltas: List[float] = []
    hidden_reset_events = 0

    is_recurrent = bool(getattr(model, "recurrent", False))
    hidden: Tuple[torch.Tensor, torch.Tensor] | None = model.zero_hidden(1, device) if is_recurrent else None
    prev_h: torch.Tensor | None = None

    global_step = 0
    life_prev = None

    obs, info = env.reset()
    for _ in range(stack_size):
        frame_buffer.append(obs.copy())
    life_prev = get_life_estimate(obs)

    while global_step < max_diag_steps:
        stack = np.stack(list(frame_buffer), axis=0).astype(np.float32) / 255.0
        x = torch.from_numpy(stack).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_recurrent:
                h_in, c_in = hidden  # type: ignore[misc]
                h_norm = h_in.norm().item()
                hidden_norms.append(h_norm)
                if prev_h is not None:
                    hidden_deltas.append((h_in - prev_h).norm().item())
                prev_h = h_in.clone()
                logits, value, h_n, c_n = model(x, hidden)  # type: ignore[arg-type]
                next_hidden = (h_n, c_n)
            else:
                logits, value = model(x)
                next_hidden = None

            dist = torch.distributions.Categorical(logits=logits)
            action_det = logits.argmax(dim=-1).item()
            action_stoch_t = dist.sample()
            action_stoch = action_stoch_t.item()

            probs = torch.softmax(logits, dim=-1)
            probs_np = probs.squeeze(0).cpu().numpy()
            logits_np = logits.squeeze(0).cpu().numpy()
            ent = dist.entropy().item()
            top1 = float(probs_np.max())
            noop_p = float(probs_np[0]) if probs_np.shape[0] > 0 else float("nan")

        actions_det.append(int(action_det))
        actions_stoch.append(int(action_stoch))
        logits_list.append(logits_np)
        probs_list.append(probs_np)
        entropy_list.append(ent)
        top1_probs.append(top1)
        noop_probs.append(noop_p)

        env_action = action_det if args.deterministic else action_stoch
        obs, reward, terminated, truncated, info = env.step(int(env_action))
        frame_buffer.append(obs.copy())
        global_step += 1

        if global_step <= 5:
            o = obs.astype(np.float32)
            stats = {
                "step": global_step,
                "shape": list(obs.shape),
                "dtype": str(obs.dtype),
                "min": float(o.min()),
                "max": float(o.max()),
                "mean": float(o.mean()),
                "std": float(o.std()),
            }
            obs_stats.append(stats)
            print(f"\nPREPROCESSING STEP {global_step}")
            print("-----------------------")
            print(
                f"obs.shape={stats['shape']} dtype={stats['dtype']} "
                f"min={stats['min']:.1f} max={stats['max']:.1f} mean={stats['mean']:.2f} std={stats['std']:.2f}"
            )
            try:
                img = Image.fromarray(obs)
                img.save(obs_dir / f"frame_{global_step:03d}.png")
            except Exception as e:  # noqa: BLE001
                print(f"[WARNING] Не удалось сохранить obs_samples/frame_{global_step:03d}.png: {e}")

        life = get_life_estimate(obs)
        if life_prev is not None and args.stop_on_death and global_step > 10 and life < 0.15 and life_prev > 0.3:
            print(f"Step {global_step} смерть (диагностика прервана по stop-on-death)")
            break
        life_prev = life

        if is_recurrent and next_hidden is not None:
            hidden = next_hidden

        if terminated or truncated:
            print(f"Episode ended at step {global_step}, resetting env (diag mode)")
            if is_recurrent:
                hidden = model.zero_hidden(1, device)
                prev_h = None
                hidden_reset_events += 1
            obs, info = env.reset()
            frame_buffer.clear()
            for _ in range(stack_size):
                frame_buffer.append(obs.copy())

    # ------- ACTION DISTRIBUTION -------
    det_counts = Counter(actions_det)
    stoch_counts = Counter(actions_stoch)
    num_actions = getattr(model, "num_actions", NUM_ACTIONS)

    unique_actions_det = len(det_counts)
    unique_actions_stoch = len(stoch_counts)
    entropy_mean = float(np.mean(entropy_list)) if entropy_list else float("nan")
    top1_prob_mean = float(np.mean(top1_probs)) if top1_probs else float("nan")
    noop_prob_mean = float(np.mean(noop_probs)) if noop_probs else float("nan")

    det_hist = [det_counts.get(a, 0) for a in range(num_actions)]
    stoch_hist = [stoch_counts.get(a, 0) for a in range(num_actions)]
    total_det = sum(det_hist)
    noop_freq = det_hist[0] / total_det if total_det > 0 else 0.0

    print("\nACTION HISTOGRAM (deterministic vs stochastic)")
    print("---------------------------------------------")
    for a in range(num_actions):
        d = det_hist[a]
        s = stoch_hist[a]
        dp = (d / total_det) * 100 if total_det > 0 else 0.0
        print(f"action {a}: det={d} ({dp:.1f}%), stoch={s}")

    if total_det > 0 and noop_freq > 0.8:
        print("\nPOLICY COLLAPSE LIKELY: NOOP > 80% deterministic")

    # ------- RECURRENT STATS -------
    recurrent_stats: Dict[str, Any] = {
        "is_recurrent": is_recurrent,
        "hidden_norm_mean": float(np.mean(hidden_norms)) if hidden_norms else float("nan"),
        "hidden_delta_mean": float(np.mean(hidden_deltas)) if hidden_deltas else float("nan"),
        "hidden_reset_events": hidden_reset_events,
    }

    if is_recurrent:
        print("\nRECURRENT STATS")
        print("---------------")
        print(
            f"hidden_norm_mean={recurrent_stats['hidden_norm_mean']:.6f} "
            f"hidden_delta_mean={recurrent_stats['hidden_delta_mean']:.6f} "
            f"hidden_reset_events={hidden_reset_events}"
        )

    # ------- VERDICT -------
    verdict_case = "CASE 5"
    verdict_reason = "Expected behaviour: стохастическая политика использует exploration, детерминированная — argmax."

    checkpoint_mismatch = False
    if cfg_snapshot is not None:
        snap_ckpt_dir = Path(str(cfg_snapshot.get("checkpoint_dir", "")))
        if snap_ckpt_dir.exists() and snap_ckpt_dir not in ckpt_path.parents:
            checkpoint_mismatch = True

    config_mismatch = parity["status"] == "MISMATCH"
    recurrent_bug = is_recurrent and recurrent_stats["hidden_delta_mean"] != float("nan") and recurrent_stats[
        "hidden_delta_mean"
    ] < 1e-4
    policy_collapse = total_det > 0 and (unique_actions_det == 1 or noop_freq > 0.8)

    if checkpoint_mismatch:
        verdict_case = "CASE 1"
        verdict_reason = "Checkpoint mismatch: загруженный чекпоинт не соответствует checkpoint_dir из config_snapshot."
    elif config_mismatch:
        verdict_case = "CASE 2"
        verdict_reason = "Config mismatch: train/test preprocessing или модельные параметры не совпадают."
    elif recurrent_bug:
        verdict_case = "CASE 3"
        verdict_reason = "Recurrent state bug: скрытое состояние LSTM почти не меняется по шагам."
    elif policy_collapse:
        verdict_case = "CASE 4"
        verdict_reason = "Policy collapse: детерминированная политика почти всегда выбирает один и тот же action (часто NOOP)."

    print("\nVERDICT")
    print("-------")
    print(f"{verdict_case} - {verdict_reason}")

    # ------- SAVE STATS / REPORT -------
    stats: Dict[str, Any] = {
        "checkpoint": ckpt_meta,
        "config_parity": {
            "status": parity["status"],
            "mismatches": parity["mismatches"],
            "snapshot_path": str(cfg_snapshot_path) if cfg_snapshot_path else None,
        },
        "obs_stats": obs_stats,
        "policy_stats": {
            "unique_actions_det": unique_actions_det,
            "unique_actions_stochastic": unique_actions_stoch,
            "entropy_mean": entropy_mean,
            "top1_prob_mean": top1_prob_mean,
            "noop_prob_mean": noop_prob_mean,
        },
        "action_histogram": {
            "deterministic": det_hist,
            "stochastic": stoch_hist,
        },
        "recurrent_stats": recurrent_stats,
        "verdict": {"case": verdict_case, "reason": verdict_reason},
    }

    with open(diag_root / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    report_lines: List[str] = []
    report_lines.append("### Checkpoint")
    report_lines.append(f"- Path: `{ckpt_meta['path']}`")
    report_lines.append(f"- File size: {ckpt_meta['file_size']} bytes")
    report_lines.append(f"- Update: {ckpt_meta.get('update', -1)}")
    report_lines.append(f"- Arch: {ckpt_meta.get('arch')} recurrent={ckpt_meta.get('recurrent')} "
                        f"lstm_hidden_size={ckpt_meta.get('lstm_hidden_size')}")
    report_lines.append(f"- Frame stack: {ckpt_meta.get('frame_stack')}")
    report_lines.append(f"- Policy checksum (SHA1 first 1MB): {ckpt_meta.get('state_dict_checksum_sha1_first_1mb')}")
    report_lines.append("")

    report_lines.append("### Config parity")
    if cfg_snapshot_path is None:
        report_lines.append("- Snapshot: not found (NO_SNAPSHOT)")
    else:
        report_lines.append(f"- Snapshot: `{cfg_snapshot_path}`")
        report_lines.append(f"- Status: **{parity['status']}**")
        if parity["status"] == "MISMATCH":
            report_lines.append("- Mismatches:")
            for m in parity["mismatches"]:
                report_lines.append(
                    f"  - {m['param']}: train={m['train']}  test={m['test']}"
                )
    report_lines.append("")

    report_lines.append("### Action histogram (deterministic)")
    for a in range(num_actions):
        cnt = det_hist[a]
        if total_det > 0:
            p = (cnt / total_det) * 100
        else:
            p = 0.0
        report_lines.append(f"- action {a}: {cnt} ({p:.1f}%)")
    report_lines.append("")

    report_lines.append("### Policy statistics")
    report_lines.append(f"- Entropy mean: {entropy_mean:.6f}")
    report_lines.append(f"- Top-1 prob mean: {top1_prob_mean:.6f}")
    report_lines.append(f"- NOOP prob mean: {noop_prob_mean:.6f}")
    report_lines.append(f"- Unique deterministic actions: {unique_actions_det}")
    report_lines.append(f"- Unique stochastic actions: {unique_actions_stoch}")
    report_lines.append("")

    report_lines.append("### Hidden state stats")
    report_lines.append(f"- Recurrent: {is_recurrent}")
    if is_recurrent:
        report_lines.append(f"- hidden_norm_mean: {recurrent_stats['hidden_norm_mean']:.6f}")
        report_lines.append(f"- hidden_delta_mean: {recurrent_stats['hidden_delta_mean']:.6f}")
        report_lines.append(f"- hidden_reset_events: {hidden_reset_events}")
    report_lines.append("")

    report_lines.append("### Final VERDICT")
    report_lines.append(f"- {verdict_case}: {verdict_reason}")

    with open(diag_root / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nДиагностический отчёт сохранён в: {diag_root}")


def _run_regular_policy(
    args: argparse.Namespace,
    ckpt_path: Path,
    model: torch.nn.Module,
    env: VampireKillerEnv,
    device: torch.device,
) -> None:
    stack_size = model.in_channels
    frame_buffer: deque[np.ndarray] = deque(maxlen=stack_size)

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


def main() -> None:
    args = parse_args()
    ckpt_path = ROOT / args.checkpoint

    device = torch.device(args.device)

    if not ckpt_path.exists():
        msg = f"Чекпоинт не найден: {ckpt_path}"
        if getattr(args, "diagnose_policy", False):
            print(f"[WARNING] {msg}")
            return
        raise FileNotFoundError(msg)

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

    try:
        if getattr(args, "diagnose_policy", False):
            _run_policy_diagnostics(args, ckpt_path, model, env, device)
        else:
            _run_regular_policy(args, ckpt_path, model, env, device)
    finally:
        env.close()


if __name__ == "__main__":
    main()
