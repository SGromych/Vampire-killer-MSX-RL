"""
PPO: обучение политики по reward (подбор предметов, смерть).

- Инициализация из BC: --bc-checkpoint checkpoints/bc/best.pt.
- Несколько env: --num-envs 2 (у каждого свой workdir runs/tmp/0, runs/tmp/1); при num_envs>1 автоматически
  post_action_delay_ms=50 и soft_reset=True, чтобы герои двигались и окна не перезапускались.
- Rollout собирается по очереди с каждого env; GAE считается по траекториям каждого env отдельно.
- Эксперименты: --run-name, --config (JSON), --reward-config (JSON), логи метрик и guardrails (collapse, NaN).
BC: test_policy.py; PPO: test_ppo.py. Документация: docs/TRAINING.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import deque, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, EnvConfig, VampireKillerEnv, NUM_ACTIONS
from msx_env.env_diagnostics import clear_env_resource_registry
from msx_env.ppo_model import ActorCritic, FRAME_STACK, init_from_bc
from msx_env.reward import default_v1_config
from msx_env.reward.config import RewardConfig

try:
    from project_config import (
        build_resolved_config_from_args,
        load_config,
    )
    _has_project_config = True
except ImportError:
    _has_project_config = False


def build_stacked_obs_single(buffer: deque, stack_size: int) -> np.ndarray:
    """Из буфера кадров собрать стопку (stack_size, H, W)."""
    stacked = np.stack(list(buffer), axis=0)
    return stacked.astype(np.float32) / 255.0


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
    last_done: bool = True,
) -> tuple[list[float], list[float]]:
    """Generalized Advantage Estimation. Returns advantages, returns."""
    advantages = []
    returns = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
            next_done = last_done
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        advantages.append(gae)
        returns.append(gae + values[t])
    advantages.reverse()
    returns.reverse()
    return advantages, returns


_session_start_time: float | None = None


def _supervisor_columns() -> tuple[str, str, str]:
    """Колонки супервизора из env (для CSV). restart_count, uptime_seconds (от начала процесса), crash_flag."""
    global _session_start_time
    r = os.environ.get("SUPERVISOR_RESTART_COUNT", "")
    c = os.environ.get("SUPERVISOR_CRASH_FLAG", "")
    if r == "" and c == "":
        return ("", "", "")
    if _session_start_time is None:
        _session_start_time = time.time()
    u = str(int(time.time() - _session_start_time)) if _session_start_time else "0"
    return (r, u, c)


def _write_metrics_header(path: Path, run_id: str = "", hostname: str = "", pid: int = 0) -> None:
    """
    Создать заголовок metrics.csv, если файл ещё не существует.
    Одновременно сохранить явную схему колонок в metrics_schema.json в том же каталоге.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        base = (
            "update,steps_per_sec,steps_per_sec_per_env,rollout_fps,sample_throughput,policy_loss,value_loss,entropy,approx_kl,explained_var,"
            "reward_mean,ep_return_mean,ep_return_min,ep_return_max,ep_steps_mean,ep_len_min,ep_len_max,"
            "unique_rooms_mean,unique_rooms_min,unique_rooms_max,deaths,stuck_events,"
            "ep_return_env0,ep_return_env1,ep_len_env0,ep_len_env1,unique_rooms_env0,unique_rooms_env1,"
            "room_transition_env0,room_transition_env1,deaths_env0,deaths_env1,stage_env0,stage_env1,"
            "stage00_exit_rate,stage00_exit_steps_mean,stage00_room_trans_mean,stage00_time_to_exit_steps,candles_broken_stage00,"
            "backtrack_rate_mean,room_dwell_steps_mean,door_encounter_count_mean,loop_len_max_mean,"
            "steps_after_key_total_mean,steps_after_key_until_exit_mean,steps_after_key_until_death_mean,"
            "reward_step,reward_pickup,reward_death,reward_novelty,reward_pingpong,reward_stuck,reward_key,reward_door,reward_backtrack,"
            "reward_stage_step,reward_stage_advance,"
            "recurrent_hidden_norm_mean,recurrent_hidden_norm_std,recurrent_hidden_delta_mean,"
            "resume_count,last_resume_update,"
            "unique_rooms_ep_env0,unique_rooms_ep_env1,unique_rooms_ep_mean,unique_rooms_ep_min,unique_rooms_ep_max,"
            "room_transitions_ep_env0,room_transitions_ep_env1,room_transitions_ep_mean,"
            "room_dwell_steps_ep_mean,"
            "stage00_exit_rate_ep,stage00_time_to_exit_steps_ep_env0,stage00_time_to_exit_steps_ep_env1,"
            "stage00_time_to_exit_steps_ep_mean,stage00_time_to_exit_steps_ep_min,stage00_time_to_exit_steps_ep_max,"
            "backtrack_rate_ep_env0,backtrack_rate_ep_env1,backtrack_rate_ep_mean,backtrack_rate_ep_min,backtrack_rate_ep_max,"
            "stage_stable_env0,stage_stable_env1"
        )
        r, u, c = _supervisor_columns()
        if r != "" or u != "" or c != "":
            base += ",restart_count,uptime_seconds,crash_flag"
        base += ",run_id,hostname,pid,last_checkpoint_path,checkpoint_update"
        # Записать заголовок CSV
        with open(path, "w", encoding="utf-8") as f:
            f.write(base + "\n")
        # Записать явную схему колонок для диагностики/префлайта
        try:
            schema_path = path.parent / "metrics_schema.json"
            columns = base.split(",")
            with open(schema_path, "w", encoding="utf-8") as sf:
                json.dump({"columns": columns}, sf, indent=2, ensure_ascii=False)
        except Exception:
            # Схема полезна, но не критична для работы обучения
            pass


def _append_metrics(
    path: Path,
    update: int,
    steps_per_sec: float,
    sample_throughput: float,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    approx_kl: float,
    explained_var: float,
    reward_mean: float,
    ep_return_mean: float,
    ep_steps_mean: float,
    unique_rooms_mean: float,
    deaths: int,
    stuck_events: int,
    components_avg: dict[str, float],
    *,
    steps_per_sec_per_env: float = 0.0,
    rollout_fps: float = 0.0,
    unique_rooms_max: int = 0,
    unique_rooms_env0: float = 0.0,
    unique_rooms_env1: float = 0.0,
    ep_return_env0: float = 0.0,
    ep_return_env1: float = 0.0,
    ep_len_env0: float = 0.0,
    ep_len_env1: float = 0.0,
    room_transition_env0: int = 0,
    room_transition_env1: int = 0,
    deaths_env0: int = 0,
    deaths_env1: int = 0,
    stage_env0: float = 0.0,
    stage_env1: float = 0.0,
    stage00_exit_rate: float = 0.0,
    stage00_exit_steps_mean: float = -1.0,
    stage00_room_trans_mean: float = 0.0,
    stage00_time_to_exit_steps: float = -1.0,
    candles_broken_stage00: float = -1.0,
    backtrack_rate_mean: float = 0.0,
    room_dwell_steps_mean: float = -1.0,
    door_encounter_count_mean: float = 0.0,
    loop_len_max_mean: float = 0.0,
    steps_after_key_total_mean: float = -1.0,
    steps_after_key_until_exit_mean: float = -1.0,
    steps_after_key_until_death_mean: float = -1.0,
    ep_return_min: float = 0.0,
    ep_return_max: float = 0.0,
    ep_len_min: float = 0.0,
    ep_len_max: float = 0.0,
    unique_rooms_min: int = 0,
    run_id: str = "",
    hostname: str = "",
    pid: int = 0,
    last_checkpoint_path: str = "",
    checkpoint_update: int = 0,
    recurrent_hidden_norm_mean: float = 0.0,
    recurrent_hidden_norm_std: float = 0.0,
    recurrent_hidden_delta_mean: float = 0.0,
    resume_count: int = 0,
    last_resume_update: int = 0,
    unique_rooms_ep_env0: float = 0.0,
    unique_rooms_ep_env1: float = 0.0,
    unique_rooms_ep_mean: float = 0.0,
    unique_rooms_ep_min: int = 0,
    unique_rooms_ep_max: int = 0,
    room_transitions_ep_env0: int = 0,
    room_transitions_ep_env1: int = 0,
    room_transitions_ep_mean: float = 0.0,
    room_dwell_steps_ep_mean: float = -1.0,
    stage00_exit_rate_ep: float = 0.0,
    stage00_time_to_exit_steps_ep_env0: float = -1.0,
    stage00_time_to_exit_steps_ep_env1: float = -1.0,
    stage00_time_to_exit_steps_ep_mean: float = -1.0,
    stage00_time_to_exit_steps_ep_min: float = -1.0,
    stage00_time_to_exit_steps_ep_max: float = -1.0,
    backtrack_rate_ep_env0: float = 0.0,
    backtrack_rate_ep_env1: float = 0.0,
    backtrack_rate_ep_mean: float = 0.0,
    backtrack_rate_ep_min: float = 0.0,
    backtrack_rate_ep_max: float = 0.0,
    stage_stable_env0: float = 0.0,
    stage_stable_env1: float = 0.0,
) -> None:
    line = (
        f"{update},{steps_per_sec:.2f},{steps_per_sec_per_env:.2f},{rollout_fps:.1f},{sample_throughput:.0f},"
        f"{policy_loss:.6f},{value_loss:.6f},{entropy:.6f},{approx_kl:.6f},{explained_var:.6f},"
        f"{reward_mean:.4f},{ep_return_mean:.2f},{ep_return_min:.2f},{ep_return_max:.2f},"
        f"{ep_steps_mean:.1f},{ep_len_min:.1f},{ep_len_max:.1f},"
        f"{unique_rooms_mean:.2f},{unique_rooms_min},{unique_rooms_max},{deaths},{stuck_events},"
        f"{ep_return_env0:.2f},{ep_return_env1:.2f},{ep_len_env0:.1f},{ep_len_env1:.1f},"
        f"{unique_rooms_env0:.2f},{unique_rooms_env1:.2f},"
        f"{room_transition_env0},{room_transition_env1},{deaths_env0},{deaths_env1},{stage_env0:.1f},{stage_env1:.1f},"
        f"{stage00_exit_rate:.2f},{stage00_exit_steps_mean:.1f},{stage00_room_trans_mean:.1f},"
        f"{stage00_time_to_exit_steps:.1f},{candles_broken_stage00:.1f},"
        f"{backtrack_rate_mean:.4f},{room_dwell_steps_mean:.1f},{door_encounter_count_mean:.1f},{loop_len_max_mean:.1f},"
        f"{steps_after_key_total_mean:.1f},{steps_after_key_until_exit_mean:.1f},{steps_after_key_until_death_mean:.1f},"
        f"{components_avg.get('step', 0):.4f},{components_avg.get('pickup', 0):.4f},"
        f"{components_avg.get('death', 0):.4f},{components_avg.get('novelty', 0):.4f},"
        f"{components_avg.get('pingpong', 0):.4f},{components_avg.get('stuck', 0):.4f},"
        f"{components_avg.get('key', 0):.4f},{components_avg.get('door', 0):.4f},{components_avg.get('backtrack', 0):.4f},"
        f"{components_avg.get('stage_step', 0):.4f},{components_avg.get('stage_advance', 0):.4f},"
        f"{recurrent_hidden_norm_mean:.4f},{recurrent_hidden_norm_std:.4f},{recurrent_hidden_delta_mean:.4f},"
        f"{resume_count},{last_resume_update},"
        f"{unique_rooms_ep_env0:.1f},{unique_rooms_ep_env1:.1f},{unique_rooms_ep_mean:.2f},{unique_rooms_ep_min},{unique_rooms_ep_max},"
        f"{room_transitions_ep_env0},{room_transitions_ep_env1},{room_transitions_ep_mean:.1f},"
        f"{room_dwell_steps_ep_mean:.1f},"
        f"{stage00_exit_rate_ep:.2f},{stage00_time_to_exit_steps_ep_env0:.1f},{stage00_time_to_exit_steps_ep_env1:.1f},"
        f"{stage00_time_to_exit_steps_ep_mean:.1f},{stage00_time_to_exit_steps_ep_min:.1f},{stage00_time_to_exit_steps_ep_max:.1f},"
        f"{backtrack_rate_ep_env0:.4f},{backtrack_rate_ep_env1:.4f},{backtrack_rate_ep_mean:.4f},"
        f"{backtrack_rate_ep_min:.4f},{backtrack_rate_ep_max:.4f},"
        f"{stage_stable_env0:.1f},{stage_stable_env1:.1f}"
    )
    r, u, c = _supervisor_columns()
    if r != "" or u != "" or c != "":
        line += f",{r},{u},{c}"
    run_id_esc = run_id.replace(",", ";")
    line += f",{run_id_esc},{hostname},{pid},{last_checkpoint_path},{checkpoint_update}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _arch_signature(model: nn.Module, stack_size: int, arch: str) -> dict:
    """Signature for resume validation: arch, frame_stack, recurrent, lstm_hidden_size, hidden, num_actions, obs_shape."""
    return {
        "arch": arch,
        "frame_stack": stack_size,
        "recurrent": getattr(model, "recurrent", False),
        "lstm_hidden_size": getattr(model, "_lstm_hidden_size", 256),
        "hidden": 512,
        "num_actions": NUM_ACTIONS,
        "obs_shape": (stack_size, 84, 84),
    }


def _validate_resume_signature(ckpt: dict, expected: dict) -> None:
    """Raise ValueError with diff if checkpoint signature does not match expected (no silent load)."""
    sig = ckpt.get("arch_signature") or {}
    got = {
        "arch": sig.get("arch", ckpt.get("arch", "?")),
        "frame_stack": sig.get("frame_stack", ckpt.get("frame_stack", "?")),
        "recurrent": sig.get("recurrent", ckpt.get("recurrent", False)),
        "lstm_hidden_size": sig.get("lstm_hidden_size", ckpt.get("lstm_hidden_size", "?")),
    }
    diff = []
    for k in ("arch", "frame_stack", "recurrent", "lstm_hidden_size"):
        e = expected.get(k)
        g = got.get(k, "?")
        if e is not None and g != e:
            diff.append(f"  {k}: checkpoint={g!r} vs expected={e!r}")
    if diff:
        raise ValueError(
            "Resume architecture mismatch (fix config or use a compatible checkpoint):\n" + "\n".join(diff)
        )


def _save_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    update: int,
    stack_size: int,
    arch: str,
) -> None:
    """Сохранить last.pt с state_dict, optimizer, update, RNG, arch_signature. При recurrent — сохранить флаги."""
    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    ckpt = {
        "state_dict": model.state_dict(),
        "frame_stack": stack_size,
        "arch": arch,
        "update": update,
        "optimizer_state": opt.state_dict(),
        "rng_state": rng,
        "arch_signature": _arch_signature(model, stack_size, arch),
    }
    if getattr(model, "recurrent", False):
        ckpt["recurrent"] = True
        ckpt["lstm_hidden_size"] = getattr(model, "_lstm_hidden_size", 256)
    torch.save(ckpt, ckpt_dir / "last.pt")


def _rotate_backups(ckpt_dir: Path, model: nn.Module, opt: torch.optim.Optimizer, update: int, stack_size: int, arch: str) -> None:
    """Сохранить текущее состояние в backup_0; сдвинуть backup_0..backup_3 -> backup_1..backup_4 (последний удаляется)."""
    backup_names = [f"backup_{i}.pt" for i in range(5)]
    for i in range(4, 0, -1):
        src = ckpt_dir / backup_names[i - 1]
        dst = ckpt_dir / backup_names[i]
        if src.exists():
            dst.replace(src) if dst.exists() else src.rename(dst)
    ckpt = {
        "state_dict": model.state_dict(),
        "frame_stack": stack_size,
        "arch": arch,
        "update": update,
        "optimizer_state": opt.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "arch_signature": _arch_signature(model, stack_size, arch),
    }
    if getattr(model, "recurrent", False):
        ckpt["recurrent"] = True
        ckpt["lstm_hidden_size"] = getattr(model, "_lstm_hidden_size", 256)
    torch.save(ckpt, ckpt_dir / backup_names[0])


def _args_from_config(config: "ResolvedConfig") -> argparse.Namespace:
    """Build an args-like namespace from ResolvedConfig so rest of main() can use args.xxx."""
    from types import SimpleNamespace
    ppo = config.ppo
    env = config.env_schema
    args = SimpleNamespace(
        checkpoint_dir=str(ppo.checkpoint_dir),
        bc_checkpoint=str(ppo.bc_checkpoint) if ppo.bc_checkpoint else None,
        epochs=ppo.epochs,
        rollout_steps=ppo.rollout_steps,
        ppo_epochs=ppo.ppo_epochs,
        batch_size=ppo.batch_size,
        lr=ppo.lr,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_eps=ppo.clip_eps,
        max_episode_steps=ppo.max_episode_steps,
        step_penalty=ppo.step_penalty,
        device=config.run.device,
        arch=ppo.arch,
        action_repeat=env.action_repeat,
        decision_fps=env.decision_fps,
        capture_backend=env.capture_backend,
        num_envs=env.num_envs,
        tmp_root=env.tmp_root,
        post_action_delay_ms=env.post_action_delay_ms,
        run_name=config.run.experiment_name,
        log_dir=None,
        use_runs_dir=config.run.use_runs_dir,
        run_dir=str(config.run.run_dir),
        config=None,
        reward_config=str(config.reward_config_path) if config.reward_config_path else None,
        novelty_reward=None,
        entropy_coef=ppo.entropy_coef,
        value_loss_coef=ppo.value_loss_coef,
        entropy_floor=ppo.entropy_floor,
        stuck_updates=ppo.stuck_updates,
        resume=ppo.resume,
        checkpoint_every=ppo.checkpoint_every,
        recurrent=ppo.recurrent,
        lstm_hidden_size=ppo.lstm_hidden_size,
        sequence_length=ppo.sequence_length,
        dry_run_seconds=ppo.dry_run_seconds,
        no_quiet=not env.quiet,
        summary_every=ppo.summary_every,
        summary_interval_sec=ppo.summary_interval_sec,
        debug=env.debug,
        debug_room_change=env.debug_room_change,
        debug_every=env.debug_every,
        debug_episode_max_steps=env.debug_episode_max_steps,
        debug_dump_frames=env.debug_dump_frames,
        debug_force_action=env.debug_force_action,
        ignore_death=env.ignore_death,
        window_title_pattern=env.window_title,
        window_rects_json=str(env.window_rects_path) if env.window_rects_path else None,
        no_reset_handshake=ppo.no_reset_handshake,
        nudge_right_steps=ppo.nudge_right_steps,
        stuck_nudge_steps=ppo.stuck_nudge_steps,
        dump_hud_every_n_steps=env.dump_hud_every_n_steps,
        perf=env.perf_profile,
        export_metrics=str(ppo.export_metrics) if ppo.export_metrics else None,
    )
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO для Vampire Killer")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo")
    p.add_argument("--bc-checkpoint", type=str, default=None, help="инициализация из BC (best.pt)")
    p.add_argument("--epochs", type=int, default=100, help="число PPO-обновлений (каждое = rollout + train)")
    p.add_argument("--rollout-steps", type=int, default=128, help="шагов на один rollout")
    p.add_argument("--ppo-epochs", type=int, default=3, help="эпох обучения на rollout")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--step-penalty", type=float, default=-0.001)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--arch", choices=["default", "deep"], default="deep")
    p.add_argument("--action-repeat", type=int, default=1, help="N внутренних шагов на 1 захват (backward compat: 1)")
    p.add_argument("--decision-fps", type=float, default=None, help="фикс. частота решений (10–15 Hz), None=макс. скорость")
    p.add_argument("--capture-backend", choices=["png", "single", "window"], default="png")
    p.add_argument("--num-envs", type=int, default=1, help="число параллельных env (уникальный workdir на инстанс)")
    p.add_argument("--tmp-root", type=str, default="runs/tmp", help="база для workdir при num_envs>1")
    p.add_argument("--post-action-delay-ms", type=float, default=None, help="задержка после действия перед grab (мс). При num-envs>1 по умолчанию 50")
    # Эксперименты и стабильность
    p.add_argument("--run-name", type=str, default=None, help="имя эксперимента (суффикс в runs/<ts>_<git>_<name>/)")
    p.add_argument("--log-dir", type=str, default=None, help="каталог логов (по умолчанию checkpoint-dir)")
    p.add_argument("--use-runs-dir", action="store_true", help="использовать runs/<timestamp>_<git>_<runname>/ как run directory")
    p.add_argument("--run-dir", type=str, default=None, help="явно задать run directory (для supervisor)")
    p.add_argument("--export-metrics", type=str, default=None, help="скопировать metrics.csv в указанную папку после обучения")
    p.add_argument("--config", type=str, default=None, help="путь к JSON конфигу эксперимента (переопределяется CLI)")
    p.add_argument("--reward-config", type=str, default=None, help="путь к JSON конфигу наград (RewardConfig)")
    p.add_argument("--novelty-reward", type=float, default=None, help="переопределить novelty_reward (награда за новую комнату)")
    p.add_argument("--entropy-coef", type=float, default=0.01, help="коэффициент энтропии в loss PPO")
    p.add_argument("--value-loss-coef", type=float, default=0.5, help="коэффициент value loss в loss PPO")
    # Guardrails (пороги)
    p.add_argument("--entropy-floor", type=float, default=0.3, help="ниже — предупреждение о коллапсе политики")
    p.add_argument("--stuck-updates", type=int, default=20, help="после скольких обновлений без роста unique_rooms — предупреждение")
    p.add_argument("--resume", action="store_true", help="продолжить с last.pt (update, optimizer, RNG)")
    p.add_argument("--checkpoint-every", type=int, default=0, help="период сохранения rolling backup (0=выкл); хранятся последние 5")
    # Recurrent (POMDP)
    p.add_argument("--recurrent", action="store_true", help="LSTM после encoder (память для лабиринта)")
    p.add_argument("--lstm-hidden-size", type=int, default=256, help="размер скрытого состояния LSTM")
    p.add_argument("--sequence-length", type=int, default=0, help="длина последовательности для лога/диагностики (0=не используется)")
    p.add_argument("--dry-run-seconds", type=float, default=0, help="прогнать N секунд и вывести steps/sec, updates/hour (для оценки скорости перед ночным запуском); 0=выкл")
    p.add_argument("--no-quiet", action="store_true", help="включить подробный вывод (Update/env return, backup saved); по умолчанию quiet (ночной режим)")
    p.add_argument("--summary-every", type=int, default=10, help="печатать компактный summary каждые N обновлений")
    p.add_argument("--summary-interval-sec", type=float, default=60.0, help="печатать summary не реже чем раз в N секунд")
    # Debug STAGE 00 / room_hash / stuck (opt-in)
    p.add_argument("--debug", action="store_true", help="диагностика: room_hash, stage, stuck, действие каждые N шагов")
    p.add_argument("--debug-room-change", action="store_true", help="печатать [ENV i] room change prev->new при смене комнаты")
    p.add_argument("--debug-every", type=int, default=10, help="печатать [debug] строку каждые N шагов")
    p.add_argument("--debug-episode-max-steps", type=int, default=400, help="ограничить эпизод при debug (0=не менять)")
    p.add_argument("--debug-dump-frames", type=int, default=0, help="сохранить первые N кадров в debug_frames/ (0=выкл)")
    p.add_argument("--debug-force-action", type=str, default="", help="подменить действие: RIGHT, LEFT, ... для проверки")
    p.add_argument("--ignore-death", action="store_true", help="не ставить terminated=True при смерти (только логировать, для проверки ложных срабатываний)")
    p.add_argument("--window-title-pattern", type=str, default=None, help="подстрока заголовка окна для window capture (по умолчанию openMSX)")
    p.add_argument("--window-rects-json", type=str, default=None, help="путь к JSON с per-env окнами: {\"0\": {\"title\": \"...\", \"crop\": [x,y,w,h]}, ...}")
    p.add_argument("--debug-single-episode", action="store_true", help="запустить ровно 1 эпизод на каждый env, вывести reset/termination и выйти (для отладки)")
    p.add_argument("--no-reset-handshake", action="store_true", help="при num_envs>1 не включать reset handshake (для отладки, если кнопки в env 1 не работают)")
    p.add_argument("--nudge-right-steps", type=int, default=0, help="в начале каждого эпизода N шагов RIGHT (подталкивание вправо, чтобы не застревать на первом экране)")
    p.add_argument("--stuck-nudge-steps", type=int, default=20, help="при застревании (stuck) N шагов по очереди RIGHT/LEFT/UP/DOWN для попытки выхода")
    p.add_argument("--dump-hud-every-n-steps", type=int, default=0, help="fix-room-metrics: сохранять HUD crop каждые N шагов в run_dir/debug/ (0=выкл, напр. 300)")
    p.add_argument("--perf", action="store_true", help="throughput diagnostics: собирать t_action/t_capture/t_reward (p50/p95) в info")
    return p.parse_args()


def _stage_mean_from_episode_stats(
    episode_stats: list, components_avg: dict
) -> float:
    """Средний stage по завершённым эпизодам; 0 если нет данных."""
    if not episode_stats:
        return 0.0
    try:
        if len(episode_stats[0]) >= 6:
            return float(np.mean([s[5] for s in episode_stats]))
    except (IndexError, TypeError):
        pass
    return 0.0


def _load_reward_config(path: str | None) -> RewardConfig:
    """Загрузить RewardConfig из JSON или вернуть default v1. Если path задан и файла нет — fail (no silent default)."""
    if not path:
        return default_v1_config()
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Reward config explicitly specified but file not found: {p}. Fix --reward-config or omit to use default."
        )
    with open(p, encoding="utf-8") as f:
        return RewardConfig.from_dict(json.load(f))


def _training_health_string(
    *,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    unique_rooms: float,
    prev: dict | None,
    change_threshold: float = 0.001,
) -> str:
    """Строка «обучение идёт»: updates^, policy_loss/value_loss меняются, entropy v/^, unique_rooms v/^."""
    if prev is None:
        return "updates^ | policy_loss/value_loss/entropy/unique_rooms (first summary)"
    pl = "changing" if abs(policy_loss - prev.get("policy_loss", 0)) > change_threshold else "stable"
    vl = "changing" if abs(value_loss - prev.get("value_loss", 0)) > change_threshold else "stable"
    e_prev = prev.get("entropy", entropy)
    ent = "v" if entropy < e_prev - change_threshold else ("^" if entropy > e_prev + change_threshold else "-")
    ur_prev = prev.get("unique_rooms", unique_rooms)
    ur = "^" if unique_rooms > ur_prev + 0.01 else ("v" if unique_rooms < ur_prev - 0.01 else "-")
    return f"updates^ | policy_loss={pl} | value_loss={vl} | entropy={ent} | unique_rooms={ur}"


def _print_training_summary(
    *,
    uptime_sec: float,
    total_steps: int,
    steps_per_sec: float,
    update: int,
    steps_per_update: int,
    rollout_steps: int,
    num_envs: int,
    checkpoint_every: int,
    next_checkpoint_in: int,
    ckpt_path: Path,
    reward_mean: float,
    ep_return_mean: float,
    ep_steps_mean: float,
    components_avg: dict,
    unique_rooms_mean: float,
    unique_rooms_max: int,
    stage_mean: float,
    deaths: int,
    stuck_events: int,
    stage00_exit_rate: float = 0.0,
    backtrack_rate_mean: float = 0.0,
    entropy_avg: float,
    approx_kl_avg: float,
    value_loss_avg: float,
    policy_loss_avg: float,
    explained_var: float,
    use_recurrent: bool,
    h_norm_mean: float = 0.0,
    prev_metrics: dict | None = None,
    deaths_per_env: list | None = None,
    ep_steps_per_env: list | None = None,
) -> None:
    """Один компактный блок статистики для консоли (ночной режим). Включает проверку «обучение идёт»."""
    top3 = sorted(
        [(k, v) for k, v in components_avg.items() if v != 0],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]
    top3_str = " ".join(f"{k}={v:.3f}" for k, v in top3) if top3 else "-"
    rec_str = f" h_norm={h_norm_mean:.3f}" if use_recurrent else ""
    next_eval_str = "N/A (no eval)"
    health = _training_health_string(
        policy_loss=policy_loss_avg,
        value_loss=value_loss_avg,
        entropy=entropy_avg,
        unique_rooms=unique_rooms_mean,
        prev=prev_metrics,
    )
    lines = [
        f"--- update {update}  uptime {uptime_sec:.0f}s  total_steps {total_steps}  steps/s {steps_per_sec:.1f}",
        f"  Training health: {health}",
        f"  last_update_collected={steps_per_update} steps (= {rollout_steps}*{num_envs})  total_rollouts={update}  next_ckpt_in={next_checkpoint_in}  next_eval={next_eval_str}",
        f"  reward_mean={reward_mean:.4f}  ep_return={ep_return_mean:.2f}  ep_steps={ep_steps_mean:.1f}  top3: {top3_str}",
        f"  unique_rooms={unique_rooms_mean:.2f}(max={unique_rooms_max})  stage_mean={stage_mean:.2f}  deaths={deaths}  stuck={stuck_events}  s00_exit={stage00_exit_rate:.0%}  backtrack={backtrack_rate_mean:.3f}",
        f"  policy_loss={policy_loss_avg:.4f}  value_loss={value_loss_avg:.4f}  entropy={entropy_avg:.4f}  kl={approx_kl_avg:.4f}  expl_var={explained_var:.4f}{rec_str}",
        f"  checkpoint: {ckpt_path}",
    ]
    if steps_per_sec <= 0:
        lines.append("  [WARN] steps/s=0 - rollout может не собираться, проверьте env/openMSX.")
    if deaths_per_env is not None and ep_steps_per_env is not None and len(deaths_per_env) == num_envs and len(ep_steps_per_env) == num_envs:
        per_env_parts = [f"deaths_env{j}={deaths_per_env[j]} ep_steps_env{j}={ep_steps_per_env[j]:.1f}" for j in range(num_envs)]
        lines.append("  per_env: " + "  ".join(per_env_parts))
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("\n".join(f"[{ts}] {line}" for line in lines))


def main() -> None:
    config = None
    if _has_project_config and "--config" in sys.argv:
        config = load_config(sys.argv[1:])
        args = _args_from_config(config)
        logging.getLogger(__name__).info("Using config: %s", config.layout.config_snapshot())
        logging.getLogger(__name__).info("Resolved run_dir: %s", config.run.run_dir)
        logging.getLogger(__name__).info(
            "Reward version: %s  key_reward=%s  door_reward=%s  stage_reward=%s",
            getattr(config.reward_config, "version", "v1"),
            getattr(config.reward_config, "key_reward", 0),
            getattr(config.reward_config, "door_reward", 0),
            getattr(config.reward_config, "enable_stage_reward", False),
        )
        logging.getLogger(__name__).info("Capture backend: %s", config.env_schema.capture_backend)
    else:
        args = parse_args()
        if _has_project_config:
            config = build_resolved_config_from_args(args, ROOT)
            logging.getLogger(__name__).info("Resolved run_dir: %s", config.run.run_dir)
            logging.getLogger(__name__).info(
                "Reward version: %s  key_reward=%s  door_reward=%s  stage_reward=%s",
                getattr(config.reward_config, "version", "v1"),
                getattr(config.reward_config, "key_reward", 0),
                getattr(config.reward_config, "door_reward", 0),
                getattr(config.reward_config, "enable_stage_reward", False),
            )
            logging.getLogger(__name__).info("Capture backend: %s", config.env_schema.capture_backend)

    device = torch.device(args.device)
    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        raise FileNotFoundError(f"ROM не найден: {rom}")

    num_envs = max(1, int(getattr(args, "num_envs", 1)))
    tmp_root_arg = getattr(args, "tmp_root", "runs/tmp")
    base_workdir = ROOT / "checkpoints" / "ppo" / "run"
    tmp_root_abs = str((ROOT / tmp_root_arg).resolve()) if num_envs > 1 else tmp_root_arg

    post_delay = getattr(args, "post_action_delay_ms", None)
    if post_delay is None and num_envs > 1:
        post_delay = 50.0  # дать эмулятору отрисовать кадр при чередовании env
    elif post_delay is None:
        post_delay = 0.0

    if config is not None:
        reward_config = config.reward_config
        run_dir = config.run.run_dir
        ckpt_dir = config.ppo.checkpoint_dir
        metrics_file = config.layout.metrics_csv()
        log_file = config.layout.train_log()
        run_name = config.run.experiment_name
    else:
        reward_config = _load_reward_config(getattr(args, "reward_config", None))
        if getattr(args, "novelty_reward", None) is not None:
            reward_config.novelty_reward = float(args.novelty_reward)
        run_name = getattr(args, "run_name", None) or "run"
        if getattr(args, "run_dir", None):
            run_dir = Path(args.run_dir).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            _readme = ROOT / "scripts" / "run_readme_template.md"
            if _readme.exists() and not (run_dir / "README.md").exists():
                import shutil
                shutil.copy2(_readme, run_dir / "README.md")
        elif getattr(args, "use_runs_dir", False):
            from scripts.run_utils import make_run_dir
            run_dir = make_run_dir(run_name, runs_base="runs")
            _readme = ROOT / "scripts" / "run_readme_template.md"
            if _readme.exists() and not (run_dir / "README.md").exists():
                import shutil
                shutil.copy2(_readme, run_dir / "README.md")
        else:
            log_dir = ROOT / (getattr(args, "log_dir", None) or args.checkpoint_dir)
            run_dir = log_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = ROOT / args.checkpoint_dir if not (getattr(args, "use_runs_dir", False) or getattr(args, "run_dir", None)) else run_dir / "checkpoints" / "ppo"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = run_dir / "metrics.csv"
        log_file = run_dir / "train.log"

    quiet = not getattr(args, "no_quiet", False)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    log_fp = open(log_file, "a", encoding="utf-8")

    class TeeWriter:
        def __init__(self, stdout, log_file_handle):
            self._stdout = stdout
            self._log = log_file_handle
        def write(self, s):
            self._stdout.write(s)
            self._log.write(s)
            self._log.flush()
        def flush(self):
            self._stdout.flush()
            self._log.flush()

    sys.stdout = TeeWriter(sys.__stdout__, log_fp)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[file_handler, console_handler],
        force=True,
    )
    logger = logging.getLogger(__name__)
    try:
        from scripts.run_utils import code_version
        logger.info("code_version=%s (check this in train.log to confirm which commit ran)", code_version())
    except Exception:
        logger.info("code_version=unknown")

    base_cfg = EnvConfig(
        rom_path=str(rom),
        workdir=str(base_workdir.resolve()),
        frame_size=(84, 84),
        terminated_on_death=True,
        max_episode_steps=args.max_episode_steps,
        action_repeat=getattr(args, "action_repeat", 1),
        decision_fps=getattr(args, "decision_fps", None),
        capture_backend=getattr(args, "capture_backend", "png"),
        reward_config=reward_config,
        tmp_root=tmp_root_abs,
        post_action_delay_ms=post_delay,
        soft_reset=True,  # не перезапускать openMSX при done — окно не мигает при num_envs>1
        quiet=quiet,
        openmsx_log_dir=str(run_dir),  # openmsx_0.log, openmsx_1.log, ...
    )
    if getattr(args, "debug_room_change", False):
        base_cfg.debug_room_change = True
    if getattr(args, "debug", False):
        base_cfg.debug = True
        base_cfg.debug_every = max(1, getattr(args, "debug_every", 10))
        base_cfg.debug_episode_max_steps = max(0, getattr(args, "debug_episode_max_steps", 400))
        base_cfg.debug_dump_frames = max(0, getattr(args, "debug_dump_frames", 0))
        base_cfg.debug_force_action = (getattr(args, "debug_force_action", "") or "").strip()
        base_cfg.debug_dump_dir = str(run_dir)
        if base_cfg.debug_episode_max_steps > 0 and (not base_cfg.max_episode_steps or base_cfg.max_episode_steps > base_cfg.debug_episode_max_steps):
            base_cfg.max_episode_steps = base_cfg.debug_episode_max_steps

    if getattr(args, "ignore_death", False):
        base_cfg.ignore_death = True

    # fix-room-metrics-stability: dump HUD crop для отладки stage detector
    dump_hud_n = max(0, getattr(args, "dump_hud_every_n_steps", 0))
    if dump_hud_n > 0:
        base_cfg = replace(
            base_cfg,
            dump_hud_every_n_steps=dump_hud_n,
            dump_hud_dir=str(run_dir / "debug"),
        )

    # Throughput diagnostics (diagnose_throughput.py): perf_profile
    if getattr(args, "perf", False):
        base_cfg = replace(base_cfg, perf_profile=True)

    # Multi-env + window capture: require per-env rects or fallback to file capture
    per_env_window = None
    if num_envs > 1 and getattr(base_cfg, "capture_backend", "png") == "window":
        window_rects_path = getattr(args, "window_rects_json", None)
        if window_rects_path:
            try:
                path = Path(window_rects_path)
                full_path = path if path.is_absolute() else (ROOT / path)
                if full_path.exists():
                    with open(full_path, encoding="utf-8") as f:
                        data = json.load(f)
                    if all(str(i) in data for i in range(num_envs)):
                        per_env_window = data
                    else:
                        base_cfg = replace(base_cfg, capture_backend="png")
                        logger.warning(
                            "num_envs>1 and window: --window-rects-json missing keys for all ranks, falling back to file capture"
                        )
                else:
                    base_cfg = replace(base_cfg, capture_backend="png")
                    logger.warning("num_envs>1 and window: --window-rects-json file not found, falling back to file capture")
            except Exception as e:
                base_cfg = replace(base_cfg, capture_backend="png")
                logger.warning("num_envs>1 and window: failed to load --window-rects-json (%s), falling back to file capture", e)
        else:
            base_cfg = replace(base_cfg, capture_backend="png")
            logger.warning(
                "num_envs>1 and capture_backend=window: no per-env rects (use --window-rects-json). Falling back to file capture."
            )
    if getattr(args, "window_title_pattern", None):
        base_cfg = replace(base_cfg, window_title=args.window_title_pattern)

    clear_env_resource_registry()

    # Multi-env: optional reset handshake (wait for stable stage before first step)
    if (
        num_envs > 1
        and not getattr(args, "no_reset_handshake", False)
        and getattr(base_cfg, "reset_handshake_stable_frames", 0) == 0
    ):
        base_cfg = replace(
            base_cfg,
            reset_handshake_stable_frames=3,
            reset_handshake_conf_min=0.5,
            reset_handshake_timeout_s=15.0,
        )

    if num_envs == 1:
        base_workdir.mkdir(parents=True, exist_ok=True)
        env = VampireKillerEnv(base_cfg)
        envs = [env]
    else:
        from msx_env.make_env import make_env
        env_fns = [make_env(i, base_cfg, per_env_window=per_env_window) for i in range(num_envs)]
        envs = [fn() for fn in env_fns]
        env = envs[0]
        workdirs = [str(e.cfg.workdir) for e in envs]
        logger.info(
            "Multi-env: num_envs=%s tmp_root=%s workdirs=%s post_action_delay_ms=%s",
            num_envs, tmp_root_abs, workdirs, base_cfg.post_action_delay_ms,
        )

    stack_size = FRAME_STACK
    arch = args.arch
    if args.bc_checkpoint:
        bc_path = ROOT / args.bc_checkpoint
        if bc_path.exists():
            try:
                ckpt = torch.load(bc_path, map_location="cpu", weights_only=True)
            except TypeError:
                ckpt = torch.load(bc_path, map_location="cpu")
            if isinstance(ckpt, dict) and "arch" in ckpt:
                arch = ckpt["arch"]
                logger.info("Архитектура из BC: %s", arch)

    if config is None:
        if getattr(args, "use_runs_dir", False) or getattr(args, "run_dir", None):
            ckpt_dir = run_dir / "checkpoints" / "ppo"
        else:
            ckpt_dir = ROOT / args.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = None
    use_recurrent = getattr(args, "recurrent", False)
    lstm_hidden_size = getattr(args, "lstm_hidden_size", 256)
    if getattr(args, "resume", False):
        last_path = ckpt_dir / "last.pt"
        if last_path.exists():
            try:
                resume_ckpt = torch.load(last_path, map_location=device, weights_only=False)
                if isinstance(resume_ckpt, dict):
                    use_recurrent = resume_ckpt.get("recurrent", use_recurrent)
                    lstm_hidden_size = resume_ckpt.get("lstm_hidden_size", lstm_hidden_size)
            except Exception:
                pass

    model = ActorCritic(
        num_actions=NUM_ACTIONS,
        in_channels=stack_size,
        arch=arch,
        recurrent=use_recurrent,
        lstm_hidden_size=lstm_hidden_size,
    ).to(device)
    if args.bc_checkpoint:
        bc_path = ROOT / args.bc_checkpoint
        if bc_path.exists():
            init_from_bc(model, bc_path)
            logger.info("Инициализировано из BC: %s", bc_path)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_update = 0
    resume_count_val = 0
    last_resume_update_val = 0
    if resume_ckpt is not None and isinstance(resume_ckpt, dict) and "state_dict" in resume_ckpt:
        try:
            expected_sig = _arch_signature(model, stack_size, arch)
            _validate_resume_signature(resume_ckpt, expected_sig)
            model.load_state_dict(resume_ckpt["state_dict"], strict=False)
            if "optimizer_state" in resume_ckpt:
                try:
                    opt.load_state_dict(resume_ckpt["optimizer_state"])
                except Exception:
                    pass
            last_resume_update_val = int(resume_ckpt.get("update", 0))
            start_update = last_resume_update_val + 1
            resume_count_val = 1
            if "rng_state" in resume_ckpt:
                r = resume_ckpt["rng_state"]
                if "torch" in r:
                    torch.set_rng_state(r["torch"])
                if "numpy" in r:
                    np.random.set_state(r["numpy"])
                if "python" in r:
                    random.setstate(r["python"])
            logger.info("Resume from update %s (loaded %s)", start_update, ckpt_dir / "last.pt")
        except Exception as e:
            logger.warning("Resume failed: %s, starting from 0", e)

    # Снимок конфига (уже записан project_config при config is not None)
    if config is None:
        config_snapshot = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(ckpt_dir),
            "epochs": args.epochs,
            "rollout_steps": args.rollout_steps,
            "ppo_epochs": args.ppo_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_eps": args.clip_eps,
            "entropy_coef": getattr(args, "entropy_coef", 0.01),
            "value_loss_coef": getattr(args, "value_loss_coef", 0.5),
            "num_envs": num_envs,
            "max_episode_steps": args.max_episode_steps,
            "reward_config": reward_config.to_dict(),
            "recurrent": use_recurrent,
            "lstm_hidden_size": lstm_hidden_size,
            "sequence_length": getattr(args, "sequence_length", 0),
        }
        config_snapshot_path = run_dir / "config_snapshot.json"
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
        print(f"Experiment run_name={run_name} log_dir={run_dir}")
        print(f"Config snapshot: {config_snapshot_path}")
    else:
        print(f"Experiment run_name={run_name} log_dir={run_dir}")
        print(f"Config snapshot: {config.layout.config_snapshot()}")

    # Метрики за обновление (для guardrails и лога)
    recent_unique_rooms: deque = deque(maxlen=max(1, getattr(args, "stuck_updates", 20)))
    if config is None:
        metrics_file = run_dir / "metrics.csv"
    _run_id, _hostname, _pid = (str(run_dir), "", 0)
    try:
        from scripts.run_utils import run_metadata
        _run_id, _hostname, _pid = run_metadata(run_dir)
    except Exception:
        pass
    _write_metrics_header(metrics_file, run_id=_run_id, hostname=_hostname, pid=_pid)
    schema_path = metrics_file.parent / "metrics_schema.json"
    if schema_path.exists():
        print(f"Metrics schema: {schema_path}")

    checkpoint_every = max(0, getattr(args, "checkpoint_every", 0))
    dry_run_seconds = max(0.0, getattr(args, "dry_run_seconds", 0))
    dry_run_start = time.perf_counter() if dry_run_seconds > 0 else None
    summary_every = max(1, getattr(args, "summary_every", 10))
    summary_interval_sec = max(1.0, getattr(args, "summary_interval_sec", 60.0))
    training_start_time = time.perf_counter()
    last_summary_time = training_start_time
    last_summary_update = start_update - 1
    last_summary_metrics: dict | None = None  # для проверки «обучение идёт» (policy_loss, value_loss, entropy, unique_rooms)
    total_steps_so_far = 0

    nudge_right_steps = max(0, getattr(args, "nudge_right_steps", 0))
    stuck_nudge_steps = max(0, getattr(args, "stuck_nudge_steps", 0))
    stuck_nudge_dirs = (ACTION_RIGHT, ACTION_LEFT, ACTION_UP, ACTION_DOWN)
    frame_buffers = [deque(maxlen=stack_size) for _ in range(num_envs)]
    obs_list = []
    for i in range(num_envs):
        if num_envs > 1 and i > 0:
            # Дать предыдущему openMSX полностью подняться и занять порт, чтобы второй инстанс не конфликтовал
            time.sleep(3.0)
        obs, _ = envs[i].reset()
        obs_list.append(obs)
        for _ in range(stack_size):
            frame_buffers[i].append(obs.copy())
        # nudge-right: в начале эпизода N шагов RIGHT, чтобы не застревать на первом экране
        if nudge_right_steps > 0:
            for _ in range(nudge_right_steps):
                obs, _, term, trunc, _ = envs[i].step(ACTION_RIGHT)
                if term or trunc:
                    obs, _ = envs[i].reset()
                    frame_buffers[i].clear()
                    for _ in range(stack_size):
                        frame_buffers[i].append(obs.copy())
                    break
                frame_buffers[i].append(obs.copy())
            obs_list[i] = obs
    episode_returns = [0.0] * num_envs
    episode_steps_list = [0] * num_envs

    if getattr(args, "debug_single_episode", False):
        done = [False] * num_envs
        while not all(done):
            for i in range(num_envs):
                if done[i]:
                    continue
                obs, reward, terminated, truncated, info = envs[i].step(0)
                if terminated or truncated:
                    done[i] = True
        print("debug-single-episode: 1 episode per env completed. Exiting.")
        sys.exit(0)

    entropy_coef = getattr(args, "entropy_coef", 0.01)
    value_loss_coef = getattr(args, "value_loss_coef", 0.5)
    entropy_floor = getattr(args, "entropy_floor", 0.3)
    stuck_updates = getattr(args, "stuck_updates", 20)

    for update in range(start_update, args.epochs):
        model.train()
        roll_obs, roll_acts, roll_logp, roll_vals, roll_rews, roll_dones = [], [], [], [], [], []
        roll_components: list[dict[str, float]] = []
        episode_stats: list[tuple] = []  # (..., steps_after_key_total, steps_after_key_until_exit, steps_after_key_until_death)
        roll_h: list[torch.Tensor] = []
        roll_c: list[torch.Tensor] = []

        if use_recurrent:
            hidden_states = [model.zero_hidden(1, device) for _ in range(num_envs)]
        else:
            hidden_states = None
        stuck_nudge_remaining = [0] * num_envs

        t_roll_start = time.perf_counter()
        # Rollout: по каждому шагу проходим все env по очереди (env0, env1, env0, ...)
        for _ in range(args.rollout_steps):
            for i in range(num_envs):
                x = torch.from_numpy(
                    build_stacked_obs_single(frame_buffers[i], stack_size)
                ).unsqueeze(0).to(device)
                # stuck-nudge: при застревании пробовать RIGHT/LEFT/UP/DOWN по очереди
                use_stuck_nudge = stuck_nudge_remaining[i] > 0 and stuck_nudge_steps > 0
                if use_stuck_nudge:
                    idx = (stuck_nudge_steps - stuck_nudge_remaining[i]) % len(stuck_nudge_dirs)
                    nudge_action = stuck_nudge_dirs[idx]
                    stuck_nudge_remaining[i] -= 1
                with torch.no_grad():
                    if use_recurrent:
                        h_in, c_in = hidden_states[i]
                        action, log_prob, value, next_hidden = model.get_action(x, deterministic=False, hidden=(h_in, c_in))
                        roll_h.append(h_in)
                        roll_c.append(c_in)
                    else:
                        action, log_prob, value, _ = model.get_action(x, deterministic=False)
                if use_stuck_nudge:
                    action = nudge_action

                obs, reward, terminated, truncated, info = envs[i].step(action)
                if info.get("reward_stuck_event") and stuck_nudge_steps > 0 and stuck_nudge_remaining[i] == 0:
                    stuck_nudge_remaining[i] = stuck_nudge_steps
                if "reward_components" not in info:
                    reward = reward + args.step_penalty
                done = terminated or truncated

                if use_recurrent:
                    if done:
                        next_hidden = model.zero_hidden(1, device)
                    hidden_states[i] = next_hidden

                roll_components.append(info.get("reward_components", {}))

                frame_buffers[i].append(obs.copy())
                obs_list[i] = obs

                roll_obs.append(x.squeeze(0).cpu().numpy())
                roll_acts.append(action)
                roll_logp.append(log_prob.cpu().item())
                roll_vals.append(value.cpu().item())
                roll_rews.append(reward)
                roll_dones.append(done)

                episode_returns[i] += reward
                episode_steps_list[i] += 1

                if done:
                    ur = info.get("reward_unique_rooms", 0)
                    if not isinstance(ur, (int, float)):
                        ur = 0
                    room_trans = int(info.get("reward_room_transition_count", 0))
                    s00_steps = int(info.get("reward_stage00_exit_steps", -1))
                    s00_ok = int(info.get("reward_stage00_exit_success", 0))
                    s00_rt = int(info.get("reward_stage00_room_transitions", 0))
                    backt = int(info.get("reward_backtrack_count", 0))
                    room_dwell = float(info.get("reward_room_dwell_steps", -1.0))
                    door_enc = int(info.get("reward_door_encounter_count", 0))
                    loop_len = int(info.get("reward_loop_len_max", 0))
                    # episode-metrics-fix: новые эпизодные метрики (debounced)
                    ur_ep = int(info.get("reward_unique_rooms_ep", 0))
                    room_trans_ep = int(info.get("reward_room_transitions_ep", 0))
                    dwell_ep_mean = float(info.get("reward_room_dwell_steps_ep_mean", -1.0))
                    s00_exit_recorded_ep = int(info.get("reward_stage00_exit_recorded_ep", 0))
                    s00_exit_steps_ep = int(info.get("reward_stage00_exit_steps_ep", -1))
                    backtrack_rate_ep = float(info.get("reward_backtrack_rate_ep", 0.0))
                    stable_stage_ep = int(info.get("reward_stable_stage_ep", 0))
                    episode_stats.append((
                        float(episode_returns[i]),
                        episode_steps_list[i],
                        int(ur),
                        bool(terminated),
                        bool(info.get("reward_stuck_event", False)),
                        int(info.get("stage", 0)),
                        i,
                        room_trans,
                        s00_steps,
                        s00_ok,
                        s00_rt,
                        backt,
                        int(info.get("reward_steps_after_key_total", -1)),
                        int(info.get("reward_steps_after_key_until_exit", -1)),
                        int(info.get("reward_steps_after_key_until_death", -1)),
                        room_dwell,
                        door_enc,
                        loop_len,
                        ur_ep,
                        room_trans_ep,
                        dwell_ep_mean,
                        s00_exit_recorded_ep,
                        s00_exit_steps_ep,
                        backtrack_rate_ep,
                        stable_stage_ep,
                    ))
                    # soft_reset=True: не перезапускаем процесс openMSX, только «продолжить» клавишами
                    obs_list[i], _ = envs[i].reset()
                    frame_buffers[i].clear()
                    for _ in range(stack_size):
                        frame_buffers[i].append(obs_list[i].copy())
                    # nudge-right: в начале эпизода N шагов RIGHT
                    if nudge_right_steps > 0:
                        for _ in range(nudge_right_steps):
                            obs_list[i], _, term, trunc, _ = envs[i].step(ACTION_RIGHT)
                            if term or trunc:
                                obs_list[i], _ = envs[i].reset()
                                frame_buffers[i].clear()
                                for _ in range(stack_size):
                                    frame_buffers[i].append(obs_list[i].copy())
                                break
                            frame_buffers[i].append(obs_list[i].copy())
                    stuck_nudge_remaining[i] = 0
                    if not quiet and update % 5 == 0:
                        logger.info(f"Update {update}  env{i} return={episode_returns[i]:.2f}  steps={episode_steps_list[i]}")
                    episode_returns[i] = 0.0
                    episode_steps_list[i] = 0
        t_roll_end = time.perf_counter()

        n_roll = len(roll_obs)
        assert n_roll == num_envs * args.rollout_steps
        # GAE по траекториям каждого env отдельно (индексы i, i+num_envs, i+2*num_envs, ...)
        adv = [0.0] * n_roll
        ret = [0.0] * n_roll
        for i in range(num_envs):
            inds = list(range(i, n_roll, num_envs))
            rews_i = [roll_rews[j] for j in inds]
            vals_i = [roll_vals[j] for j in inds]
            dones_i = [roll_dones[j] for j in inds]
            if dones_i[-1]:
                last_val, last_done = 0.0, True
            else:
                with torch.no_grad():
                    x_last = torch.from_numpy(
                        build_stacked_obs_single(frame_buffers[i], stack_size)
                    ).unsqueeze(0).to(device)
                    if use_recurrent:
                        _, _, last_val, _ = model.get_action(x_last, deterministic=True, hidden=hidden_states[i])
                    else:
                        _, _, last_val, _ = model.get_action(x_last, deterministic=True)
                    last_val = last_val.cpu().item()
                last_done = False
            adv_i, ret_i = compute_gae(
                rews_i, vals_i, dones_i,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                last_value=last_val, last_done=last_done,
            )
            for j, (a, r) in zip(inds, zip(adv_i, ret_i)):
                adv[j], ret[j] = a, r

        obs_t = torch.from_numpy(np.stack(roll_obs)).to(device)
        acts_t = torch.tensor(roll_acts, dtype=torch.long, device=device)
        logp_old_t = torch.tensor(roll_logp, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        assert obs_t.shape[0] == n_roll

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Агрегация метрик по rollout
        comp_agg: dict[str, list[float]] = defaultdict(list)
        for c in roll_components:
            for k, v in c.items():
                comp_agg[k].append(float(v))
        components_avg = {k: (sum(v) / len(v)) if v else 0.0 for k, v in comp_agg.items()}
        reward_mean = float(np.mean(roll_rews))
        ep_return_mean = float(np.mean([e[0] for e in episode_stats])) if episode_stats else 0.0
        ep_steps_mean = float(np.mean([e[1] for e in episode_stats])) if episode_stats else 0.0
        ep_returns = [e[0] for e in episode_stats]
        ep_steps_list = [e[1] for e in episode_stats]
        unique_rooms_list = [e[2] for e in episode_stats]
        unique_rooms_mean = float(np.mean(unique_rooms_list)) if episode_stats else 0.0
        unique_rooms_min = min(unique_rooms_list, default=0)
        unique_rooms_max = max(unique_rooms_list, default=0)
        ep_return_min = min(ep_returns, default=0.0)
        ep_return_max = max(ep_returns, default=0.0)
        ep_len_min = min(ep_steps_list, default=0.0)
        ep_len_max = max(ep_steps_list, default=0.0)
        unique_rooms_per_env = [
            float(np.mean([e[2] for e in episode_stats if e[6] == j])) if [e for e in episode_stats if e[6] == j] else 0.0
            for j in range(num_envs)
        ]
        room_trans_per_env = [
            sum(e[7] for e in episode_stats if e[6] == j) for j in range(num_envs)
        ]
        s00_exit_steps_list = [e[8] for e in episode_stats if e[8] >= 0]
        stage00_exit_steps_mean = float(np.mean(s00_exit_steps_list)) if s00_exit_steps_list else -1.0
        stage00_exit_rate = float(np.mean([e[9] for e in episode_stats])) if episode_stats else 0.0
        stage00_room_trans_mean = float(np.mean([e[10] for e in episode_stats])) if episode_stats else 0.0
        backtrack_per_env = [sum(e[11] for e in episode_stats if e[6] == j) for j in range(num_envs)]
        total_trans = sum(room_trans_per_env)
        total_backtracks = sum(backtrack_per_env)
        backtrack_rate_mean = total_backtracks / max(1, total_trans) if total_trans > 0 else 0.0

        sak_total_list = [e[12] for e in episode_stats if e[12] >= 0]
        sak_exit_list = [e[13] for e in episode_stats if e[13] >= 0]
        sak_death_list = [e[14] for e in episode_stats if e[14] >= 0]
        steps_after_key_total_mean = float(np.mean(sak_total_list)) if sak_total_list else -1.0
        steps_after_key_until_exit_mean = float(np.mean(sak_exit_list)) if sak_exit_list else -1.0
        steps_after_key_until_death_mean = float(np.mean(sak_death_list)) if sak_death_list else -1.0

        room_dwell_list = [e[15] for e in episode_stats if e[15] >= 0]
        room_dwell_steps_mean = float(np.mean(room_dwell_list)) if room_dwell_list else -1.0
        door_enc_list = [e[16] for e in episode_stats]
        door_encounter_count_mean = float(np.mean(door_enc_list)) if door_enc_list else 0.0
        loop_len_list = [e[17] for e in episode_stats if e[17] > 0]
        loop_len_max_mean = float(np.mean(loop_len_list)) if loop_len_list else 0.0
        s00_exit_steps_list_valid = [e[8] for e in episode_stats if e[8] >= 0]
        stage00_time_to_exit_steps = float(np.mean(s00_exit_steps_list_valid)) if s00_exit_steps_list_valid else -1.0

        # episode-metrics-fix, fix-room-metrics-stability: агрегация эпизодных метрик (indices 18-24)
        ur_ep_list = [e[18] for e in episode_stats]
        room_trans_ep_list = [e[19] for e in episode_stats]
        dwell_ep_list = [e[20] for e in episode_stats if e[20] >= 0]
        s00_exit_recorded_list = [e[21] for e in episode_stats]
        s00_exit_steps_ep_list = [e[22] for e in episode_stats if e[22] >= 0]
        backtrack_rate_ep_list = [e[23] for e in episode_stats]
        unique_rooms_ep_mean_val = float(np.mean(ur_ep_list)) if ur_ep_list else 0.0
        unique_rooms_ep_min_val = min(ur_ep_list, default=0)
        unique_rooms_ep_max_val = max(ur_ep_list, default=0)
        room_transitions_ep_mean_val = float(np.mean(room_trans_ep_list)) if room_trans_ep_list else 0.0
        room_dwell_steps_ep_mean_val = float(np.mean(dwell_ep_list)) if dwell_ep_list else -1.0
        stage00_exit_rate_ep_val = float(np.mean(s00_exit_recorded_list)) if s00_exit_recorded_list else 0.0
        stage00_time_to_exit_ep_mean_val = float(np.mean(s00_exit_steps_ep_list)) if s00_exit_steps_ep_list else -1.0
        stage00_time_to_exit_ep_min_val = min(s00_exit_steps_ep_list) if s00_exit_steps_ep_list else -1.0
        stage00_time_to_exit_ep_max_val = max(s00_exit_steps_ep_list) if s00_exit_steps_ep_list else -1.0
        backtrack_rate_ep_mean_val = float(np.mean(backtrack_rate_ep_list)) if backtrack_rate_ep_list else 0.0
        backtrack_rate_ep_min_val = min(backtrack_rate_ep_list, default=0.0)
        backtrack_rate_ep_max_val = max(backtrack_rate_ep_list, default=0.0)
        unique_rooms_ep_per_env = [
            float(np.mean([e[18] for e in episode_stats if e[6] == j])) if [e for e in episode_stats if e[6] == j] else 0.0
            for j in range(num_envs)
        ]
        room_transitions_ep_per_env = [
            sum(e[19] for e in episode_stats if e[6] == j) for j in range(num_envs)
        ]
        stage00_time_to_exit_ep_per_env = []
        for j in range(num_envs):
            lst = [e[22] for e in episode_stats if e[6] == j and e[22] >= 0]
            stage00_time_to_exit_ep_per_env.append(float(np.mean(lst)) if lst else -1.0)
        backtrack_rate_ep_per_env = [
            float(np.mean([e[23] for e in episode_stats if e[6] == j])) if [e for e in episode_stats if e[6] == j] else 0.0
            for j in range(num_envs)
        ]
        stable_stage_ep_per_env = [
            float(np.mean([e[24] for e in episode_stats if e[6] == j])) if [e for e in episode_stats if e[6] == j] else 0.0
            for j in range(num_envs)
        ]

        deaths = sum(1 for e in episode_stats if e[3])
        stuck_events = sum(1 for e in episode_stats if e[4])
        deaths_per_env = [sum(1 for e in episode_stats if e[3] and e[6] == j) for j in range(num_envs)]
        ep_steps_per_env = []
        ep_return_per_env = []
        stage_per_env = []
        for j in range(num_envs):
            steps_j = [e[1] for e in episode_stats if e[6] == j]
            ep_steps_per_env.append(float(np.mean(steps_j)) if steps_j else 0.0)
            ret_j = [e[0] for e in episode_stats if e[6] == j]
            ep_return_per_env.append(float(np.mean(ret_j)) if ret_j else 0.0)
            st_j = [e[5] for e in episode_stats if e[6] == j]
            stage_per_env.append(float(np.mean(st_j)) if st_j else 0.0)
        recent_unique_rooms.append(unique_rooms_mean)

        h_norm_mean = h_norm_std = h_delta_mean = 0.0
        if use_recurrent and roll_h:
            with torch.no_grad():
                norms = [h.norm().item() for h in roll_h]
                h_norm_mean = float(np.mean(norms))
                h_norm_std = float(np.std(norms)) if len(norms) > 1 else 0.0
                deltas = []
                for j in range(1, len(roll_h)):
                    d = (roll_h[j] - roll_h[j - 1]).norm().item()
                    deltas.append(d)
                h_delta_mean = float(np.mean(deltas)) if deltas else 0.0

        # PPO update: несколько эпох по мини-батчам с перемешиванием
        n = len(roll_obs)
        inds = np.arange(n)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        t_ppo_start = time.perf_counter()
        for _ in range(args.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                mb = inds[start:end]
                x_mb = obs_t[mb]
                acts_mb = acts_t[mb]
                logp_old_mb = logp_old_t[mb]
                adv_mb = adv_t[mb]
                ret_mb = ret_t[mb]

                if use_recurrent:
                    logits_list, values_list = [], []
                    for j in mb:
                        out = model(obs_t[j : j + 1], (roll_h[j], roll_c[j]))
                        logits_list.append(out[0])
                        values_list.append(out[1])
                    logits = torch.cat(logits_list, dim=0)
                    values = torch.cat(values_list, dim=0)
                else:
                    logits, values = model(x_mb)
                dist = torch.distributions.Categorical(logits=logits)
                logp_new = dist.log_prob(acts_mb)
                entropy = dist.entropy().mean()
                approx_kl = (logp_old_mb - logp_new).mean().item()

                ratio = (logp_new - logp_old_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(values.squeeze(-1), ret_mb)
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                approx_kls.append(approx_kl)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        t_ppo_end = time.perf_counter()

        # Explained variance (текущая модель по rollout)
        with torch.no_grad():
            out = model(obs_t)
            new_vals = out[1].squeeze(-1).cpu().numpy()  # value всегда второй; recurrent возвращает (logits, val, h, c)
        ret_np = ret_t.cpu().numpy()
        var_ret = np.var(ret_np)
        explained_var = float(1.0 - np.var(ret_np - new_vals) / (var_ret + 1e-8)) if var_ret > 1e-8 else 0.0

        policy_loss_avg = float(np.mean(policy_losses))
        value_loss_avg = float(np.mean(value_losses))
        entropy_avg = float(np.mean(entropies))
        approx_kl_avg = float(np.mean(approx_kls))

        roll_time = t_roll_end - t_roll_start
        ppo_time = t_ppo_end - t_ppo_start
        steps_per_sec = n_roll / (roll_time + ppo_time) if (roll_time + ppo_time) > 0 else 0.0
        steps_per_sec_per_env_val = steps_per_sec / max(1, num_envs)
        rollout_fps_val = n_roll / roll_time if roll_time > 0 else 0.0
        sample_throughput = n_roll / roll_time if roll_time > 0 else 0.0

        # Guardrails (always to console as warnings)
        if np.isnan(policy_loss_avg) or np.isnan(value_loss_avg) or np.isnan(entropy_avg):
            logger.warning(f"NaN в loss/entropy на update {update} — проверьте lr и reward scale.")
        if entropy_avg < entropy_floor:
            logger.warning(f"Низкая энтропия {entropy_avg:.4f} < {entropy_floor} — риск коллапса политики; рассмотрите увеличение --entropy-coef.")
        if value_loss_avg > 5.0 or explained_var < -0.5:
            logger.warning(f"Возможная расходимость критика: value_loss={value_loss_avg:.2f} expl_var={explained_var:.2f}; проверьте масштаб наград.")
        if len(recent_unique_rooms) >= stuck_updates and max(recent_unique_rooms) == min(recent_unique_rooms) and max(recent_unique_rooms) <= 1:
            logger.warning(f"Нет роста unique_rooms за последние {stuck_updates} обновлений — обучение может застрять; проверьте reward/exploration.")

        total_steps_so_far += n_roll
        uptime_sec = time.perf_counter() - training_start_time
        steps_per_update = n_roll
        next_ckpt = (checkpoint_every - (update + 1) % checkpoint_every) % max(1, checkpoint_every) if checkpoint_every > 0 else 0
        stage_mean_val = _stage_mean_from_episode_stats(episode_stats, components_avg)

        # Compact summary: every summary_every updates or every summary_interval_sec
        now = time.perf_counter()
        if (update + 1 - last_summary_update >= summary_every) or (now - last_summary_time >= summary_interval_sec) or (update + 1 == args.epochs):
            _print_training_summary(
                uptime_sec=uptime_sec,
                total_steps=total_steps_so_far,
                steps_per_sec=steps_per_sec,
                update=update + 1,
                steps_per_update=steps_per_update,
                rollout_steps=args.rollout_steps,
                num_envs=num_envs,
                checkpoint_every=checkpoint_every,
                next_checkpoint_in=next_ckpt,
                ckpt_path=ckpt_dir / "last.pt",
                reward_mean=reward_mean,
                ep_return_mean=ep_return_mean,
                ep_steps_mean=ep_steps_mean,
                components_avg=components_avg,
                unique_rooms_mean=unique_rooms_mean,
                unique_rooms_max=unique_rooms_max,
                stage_mean=stage_mean_val,
                deaths=deaths,
                stuck_events=stuck_events,
                stage00_exit_rate=stage00_exit_rate,
                backtrack_rate_mean=backtrack_rate_mean,
                entropy_avg=entropy_avg,
                approx_kl_avg=approx_kl_avg,
                value_loss_avg=value_loss_avg,
                policy_loss_avg=policy_loss_avg,
                explained_var=explained_var,
                use_recurrent=use_recurrent,
                h_norm_mean=h_norm_mean,
                prev_metrics=last_summary_metrics,
                deaths_per_env=deaths_per_env if getattr(args, "debug", False) else None,
                ep_steps_per_env=ep_steps_per_env if getattr(args, "debug", False) else None,
            )
            last_summary_time = now
            last_summary_update = update + 1
            last_summary_metrics = {
                "policy_loss": policy_loss_avg,
                "value_loss": value_loss_avg,
                "entropy": entropy_avg,
                "unique_rooms": unique_rooms_mean,
            }
        _append_metrics(
            metrics_file,
            update + 1,
            steps_per_sec,
            sample_throughput,
            policy_loss_avg,
            value_loss_avg,
            entropy_avg,
            approx_kl_avg,
            explained_var,
            reward_mean,
            ep_return_mean,
            ep_steps_mean,
            unique_rooms_mean,
            deaths,
            stuck_events,
            components_avg,
            steps_per_sec_per_env=steps_per_sec_per_env_val,
            rollout_fps=rollout_fps_val,
            unique_rooms_max=unique_rooms_max,
            unique_rooms_min=unique_rooms_min,
            ep_return_min=ep_return_min,
            ep_return_max=ep_return_max,
            ep_len_min=ep_len_min,
            ep_len_max=ep_len_max,
            run_id=_run_id,
            hostname=_hostname,
            pid=_pid,
            last_checkpoint_path=str(ckpt_dir / "last.pt"),
            checkpoint_update=update + 1,
            ep_return_env0=ep_return_per_env[0] if num_envs >= 1 else 0.0,
            ep_return_env1=ep_return_per_env[1] if num_envs >= 2 else 0.0,
            ep_len_env0=ep_steps_per_env[0] if num_envs >= 1 else 0.0,
            ep_len_env1=ep_steps_per_env[1] if num_envs >= 2 else 0.0,
            unique_rooms_env0=unique_rooms_per_env[0] if num_envs >= 1 else 0.0,
            unique_rooms_env1=unique_rooms_per_env[1] if num_envs >= 2 else 0.0,
            room_transition_env0=room_trans_per_env[0] if num_envs >= 1 else 0,
            room_transition_env1=room_trans_per_env[1] if num_envs >= 2 else 0,
            deaths_env0=deaths_per_env[0] if num_envs >= 1 else 0,
            deaths_env1=deaths_per_env[1] if num_envs >= 2 else 0,
            stage_env0=stage_per_env[0] if num_envs >= 1 else 0.0,
            stage_env1=stage_per_env[1] if num_envs >= 2 else 0.0,
            stage00_exit_rate=stage00_exit_rate,
            stage00_exit_steps_mean=stage00_exit_steps_mean,
            stage00_room_trans_mean=stage00_room_trans_mean,
            stage00_time_to_exit_steps=stage00_time_to_exit_steps,
            candles_broken_stage00=-1.0,
            backtrack_rate_mean=backtrack_rate_mean,
            room_dwell_steps_mean=room_dwell_steps_mean,
            door_encounter_count_mean=door_encounter_count_mean,
            loop_len_max_mean=loop_len_max_mean,
            steps_after_key_total_mean=steps_after_key_total_mean,
            steps_after_key_until_exit_mean=steps_after_key_until_exit_mean,
            steps_after_key_until_death_mean=steps_after_key_until_death_mean,
            recurrent_hidden_norm_mean=h_norm_mean,
            recurrent_hidden_norm_std=h_norm_std,
            recurrent_hidden_delta_mean=h_delta_mean,
            resume_count=resume_count_val,
            last_resume_update=last_resume_update_val,
            unique_rooms_ep_env0=unique_rooms_ep_per_env[0] if num_envs >= 1 else 0.0,
            unique_rooms_ep_env1=unique_rooms_ep_per_env[1] if num_envs >= 2 else 0.0,
            unique_rooms_ep_mean=unique_rooms_ep_mean_val,
            unique_rooms_ep_min=unique_rooms_ep_min_val,
            unique_rooms_ep_max=unique_rooms_ep_max_val,
            room_transitions_ep_env0=room_transitions_ep_per_env[0] if num_envs >= 1 else 0,
            room_transitions_ep_env1=room_transitions_ep_per_env[1] if num_envs >= 2 else 0,
            room_transitions_ep_mean=room_transitions_ep_mean_val,
            room_dwell_steps_ep_mean=room_dwell_steps_ep_mean_val,
            stage00_exit_rate_ep=stage00_exit_rate_ep_val,
            stage00_time_to_exit_steps_ep_env0=stage00_time_to_exit_ep_per_env[0] if num_envs >= 1 else -1.0,
            stage00_time_to_exit_steps_ep_env1=stage00_time_to_exit_ep_per_env[1] if num_envs >= 2 else -1.0,
            stage00_time_to_exit_steps_ep_mean=stage00_time_to_exit_ep_mean_val,
            stage00_time_to_exit_steps_ep_min=stage00_time_to_exit_ep_min_val,
            stage00_time_to_exit_steps_ep_max=stage00_time_to_exit_ep_max_val,
            backtrack_rate_ep_env0=backtrack_rate_ep_per_env[0] if num_envs >= 1 else 0.0,
            backtrack_rate_ep_env1=backtrack_rate_ep_per_env[1] if num_envs >= 2 else 0.0,
            backtrack_rate_ep_mean=backtrack_rate_ep_mean_val,
            backtrack_rate_ep_min=backtrack_rate_ep_min_val,
            backtrack_rate_ep_max=backtrack_rate_ep_max_val,
            stage_stable_env0=stable_stage_ep_per_env[0] if num_envs >= 1 else 0.0,
            stage_stable_env1=stable_stage_ep_per_env[1] if num_envs >= 2 else 0.0,
        )

        _save_checkpoint(ckpt_dir, model, opt, update + 1, stack_size, args.arch)
        if checkpoint_every > 0 and (update + 1) % checkpoint_every == 0:
            _rotate_backups(ckpt_dir, model, opt, update + 1, stack_size, args.arch)
            if not quiet:
                logger.info(f"Update {update+1}  backup saved to {ckpt_dir}")
        if (update + 1) % 10 == 0:
            torch.save({
                "state_dict": model.state_dict(),
                "frame_stack": stack_size,
                "arch": args.arch,
            }, ckpt_dir / f"epoch_{update+1}.pt")
            if not quiet:
                logger.info(f"Update {update+1}  saved to {ckpt_dir}")

        # Dry run: после N секунд вывести steps/sec, updates/hour и выйти
        if dry_run_seconds > 0 and dry_run_start is not None:
            elapsed = time.perf_counter() - dry_run_start
            if elapsed >= dry_run_seconds:
                total_steps = (update - start_update + 1) * n_roll
                steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
                updates_done = update - start_update + 1
                updates_per_hour = (updates_done / elapsed) * 3600 if elapsed > 0 else 0
                print(
                    f"\n[DRY RUN] {elapsed:.1f}s  total_steps={total_steps}  updates={updates_done}\n"
                    f"  steps/sec = {steps_per_sec:.2f}  steps/hour ≈ {steps_per_sec * 3600:,.0f}\n"
                    f"  updates/hour = {updates_per_hour:.1f}\n"
                    f"  Estimated max_total_steps for 8h at this speed: ≈ {int(steps_per_sec * 3600 * 8):,}"
                )
                break

    for e in envs:
        e.close()

    print("Обучение PPO завершено.")
    print(f"LAST_RUN_PATH={run_dir}")

    if getattr(args, "export_metrics", None):
        import shutil
        dest = Path(args.export_metrics).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_file, dest / "metrics.csv")
        print(f"Метрики экспортированы в {dest / 'metrics.csv'}")


if __name__ == "__main__":
    main()
