"""
Ночной супервизор PPO: запуск train_ppo в 2 env, автовозобновление при падении,
rollback при NaN, watchdog по таймауту. Читает configs/night_training.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from config_loader import load_night_training_config

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "night_training.json"


def load_config() -> dict:
    """Load night training config via central loader (single source of truth)."""
    # CONFIG_PATH is kept for clarity; config_loader already defaults to this path.
    return load_night_training_config()


def run_dir_from_config(cfg: dict) -> Path:
    log_dir = cfg.get("log_dir") or cfg.get("checkpoint_dir", "checkpoints/ppo")
    return ROOT / log_dir / cfg["run_name"]


def metrics_file_from_config(cfg: dict) -> Path:
    return run_dir_from_config(cfg) / "metrics.csv"


def ckpt_dir_from_config(cfg: dict) -> Path:
    return ROOT / cfg["checkpoint_dir"]


def last_line_contains_nan(metrics_path: Path) -> bool:
    if not metrics_path.exists():
        return False
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if not lines:
            return False
        last = lines[-1].lower()
        return "nan" in last or "inf" in last
    except Exception:
        return False


def rollback_to_safe_checkpoint(ckpt_dir: Path) -> bool:
    """Скопировать последний backup поверх last.pt. Возвращает True, если откат выполнен."""
    for i in range(5):
        backup = ckpt_dir / f"backup_{i}.pt"
        if backup.exists():
            last = ckpt_dir / "last.pt"
            shutil.copy2(backup, last)
            return True
    return False


def build_argv(cfg: dict, *, with_resume: bool = True, run_dir_override: Path | None = None) -> list[str]:
    # Primary: if run_dir has config_snapshot.json, spawn with --config + --resume only (same config for all restarts)
    snapshot = (run_dir_override / "config_snapshot.json") if run_dir_override else None
    if snapshot and snapshot.exists():
        argv = [
            sys.executable,
            str(ROOT / "train_ppo.py"),
            "--config", str(snapshot.resolve()),
        ]
        if with_resume:
            argv.append("--resume")
        return argv
    # Legacy: first start of this run_dir (snapshot not yet written) — do NOT pass --resume, checkpoint doesn't exist
    argv = [
        sys.executable,
        str(ROOT / "train_ppo.py"),
        "--num-envs", str(cfg["num_envs"]),
        "--run-name", cfg["run_name"],
        "--epochs", str(cfg["max_updates"]),
        "--checkpoint-dir", cfg["checkpoint_dir"],
        "--entropy-floor", str(cfg["entropy_floor"]),
    ]
    # --resume only when we have a snapshot (restart); first run has no checkpoint yet
    if with_resume and snapshot and snapshot.exists():
        argv.append("--resume")
    if cfg.get("checkpoint_every", 0) > 0:
        argv.extend(["--checkpoint-every", str(cfg["checkpoint_every"])])
    if cfg.get("nudge_right_steps", 0) > 0:
        argv.extend(["--nudge-right-steps", str(cfg["nudge_right_steps"])])
    if cfg.get("stuck_nudge_steps", 0) > 0:
        argv.extend(["--stuck-nudge-steps", str(cfg["stuck_nudge_steps"])])
    if "novelty_reward" in cfg:
        argv.extend(["--novelty-reward", str(cfg["novelty_reward"])])
    if "rollout_steps" in cfg:
        argv.extend(["--rollout-steps", str(cfg["rollout_steps"])])
    if "entropy_coef" in cfg:
        argv.extend(["--entropy-coef", str(cfg["entropy_coef"])])
    if "max_episode_steps" in cfg:
        argv.extend(["--max-episode-steps", str(cfg["max_episode_steps"])])
    if cfg.get("recurrent", False):
        argv.append("--recurrent")
    if cfg.get("use_runs_dir", False):
        argv.append("--use-runs-dir")
    if cfg.get("num_envs", 1) > 1:
        argv.append("--no-reset-handshake")
    if cfg.get("reward_config"):
        argv.extend(["--reward-config", str(cfg["reward_config"])])
    return argv


def run_training(
    cfg: dict,
    restart_count: int,
    crash_flag: int,
    process_holder: list,
    watchdog_stop: threading.Event,
    last_metric_mtime: list,
    run_dir_override: Path | None = None,
) -> int:
    """Запуск train_ppo; возвращает exit code. process_holder[0] = subprocess.Popen."""
    if run_dir_override is not None:
        run_dir = run_dir_override
        ckpt_dir = run_dir / "checkpoints" / "ppo"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.csv"
    else:
        run_dir = run_dir_from_config(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = ckpt_dir_from_config(cfg)
        metrics_path = metrics_file_from_config(cfg)

    env = os.environ.copy()
    env["SUPERVISOR_RESTART_COUNT"] = str(restart_count)
    env["SUPERVISOR_CRASH_FLAG"] = str(crash_flag)
    # SUPERVISOR_UPTIME_SECONDS считается внутри train_ppo от старта процесса

    argv = build_argv(cfg, run_dir_override=run_dir_override)
    if run_dir_override is not None and not (run_dir_override / "config_snapshot.json").exists():
        argv.extend(["--run-dir", str(run_dir)])
    last_metric_mtime[0] = time.time()
    if metrics_path.exists():
        last_metric_mtime[0] = metrics_path.stat().st_mtime

    # stdout/stderr=None (inherit) — чтобы train_ppo писал в консоль напрямую,
    # как при прямом запуске. Иначе PIPE может блокировать и ломать тайминг (кнопки в env не нажимаются).
    proc = subprocess.Popen(
        argv,
        cwd=str(ROOT),
        env=env,
        stdout=None,
        stderr=None,
    )
    process_holder[0] = proc

    while proc.poll() is None:
        time.sleep(5)
        if metrics_path.exists():
            last_metric_mtime[0] = max(last_metric_mtime[0], metrics_path.stat().st_mtime)
        if watchdog_stop.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            return -1
    return proc.returncode or 0


def watchdog_loop(
    cfg: dict,
    process_holder: list,
    last_metric_mtime: list,
    stop_event: threading.Event,
    run_dir_for_log: Path | None = None,
) -> None:
    """Если вывод/heartbeat не было watchdog_timeout_minutes минут — выставить stop_event и убить процесс."""
    timeout_s = cfg["watchdog_timeout_minutes"] * 60
    while not stop_event.is_set():
        stop_event.wait(min(60, timeout_s // 2))
        if stop_event.is_set():
            break
        proc = process_holder[0] if process_holder else None
        if proc is None or proc.poll() is not None:
            continue
        now = time.time()
        if now - last_metric_mtime[0] > timeout_s:
            log_path = (run_dir_for_log / "train.log") if run_dir_for_log else (run_dir_from_config(cfg) / "supervisor.log")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[watchdog] No output/heartbeat for {cfg['watchdog_timeout_minutes']} min, killing process\n")
            except Exception:
                pass
            stop_event.set()


def _supervisor_main() -> None:
    cfg = load_config()
    run_dir_override = None
    if cfg.get("use_runs_dir", False):
        from scripts.run_utils import make_run_dir
        run_dir_override = make_run_dir(cfg["run_name"], runs_base="runs")
        run_dir_override.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir_override / "checkpoints" / "ppo"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir_override / "metrics.csv"
    else:
        run_dir_override = run_dir_from_config(cfg)
        run_dir_override.mkdir(parents=True, exist_ok=True)
        ckpt_dir = ckpt_dir_from_config(cfg)
        metrics_path = metrics_file_from_config(cfg)
    log_path = run_dir_override / "supervisor.log"
    restart_count = 0
    restart_delay = cfg.get("restart_delay_seconds", 30)
    restart_limit = cfg.get("restart_limit", 20)

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n--- Supervisor started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    while restart_count <= restart_limit:
        crash_flag = 1 if restart_count > 0 else 0
        process_holder = [None]
        last_metric_mtime = [time.time()]
        watchdog_stop = threading.Event()
        wd_thread = threading.Thread(
            target=watchdog_loop,
            args=(cfg, process_holder, last_metric_mtime, watchdog_stop, run_dir_override),
            daemon=True,
        )
        wd_thread.start()

        exit_code = run_training(cfg, restart_count, crash_flag, process_holder, watchdog_stop, last_metric_mtime, run_dir_override=run_dir_override)
        watchdog_stop.set()

        if exit_code != 0:
            crash_flag = 1

        # NaN rollback: если в последней строке метрик есть nan — откатить last.pt из backup
        if last_line_contains_nan(metrics_path):
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] NaN detected in metrics, rolling back to backup\n")
            if rollback_to_safe_checkpoint(ckpt_dir):
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write("Rollback: restored last.pt from backup\n")
            crash_flag = 1

        # Нормальное завершение (все обновления отработаны)
        if exit_code == 0 and crash_flag == 0:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write("Training completed successfully.\n")
            break

        restart_count += 1
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Restart #{restart_count} exit_code={exit_code} crash_flag={crash_flag}\n"
            )

        # Сохранить копию last.pt при падении (для разбора)
        if exit_code != 0 and (ckpt_dir / "last.pt").exists():
            crash_ckpt = ckpt_dir / f"crash_restart_{restart_count}.pt"
            shutil.copy2(ckpt_dir / "last.pt", crash_ckpt)

        if restart_count > restart_limit:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"Restart limit {restart_limit} reached, stopping.\n")
            break

        time.sleep(restart_delay)

    print("Supervisor finished.")


def _print_metrics_tail(metrics_path: Path, lines: int = 3) -> None:
    if not metrics_path.exists():
        print(f"[preflight] metrics.csv not found at {metrics_path}")
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        rows = [l.rstrip("\n") for l in f if l.strip()]
    if not rows:
        print("[preflight] metrics.csv is empty")
        return
    header = rows[0]
    tail = rows[1:][-lines:]
    print("\n[preflight] metrics.csv header:")
    print(header)
    print(f"[preflight] last {len(tail)} rows:")
    for r in tail:
        print(r)


def _preflight() -> None:
    """60s dry-run с num_envs=1, печать путей и проверки last.pt/metrics."""
    cfg = load_config().copy()
    # Для префлайта принудительно без use_runs_dir, чтобы пути были стабильными и совпадали с выводом.
    cfg["num_envs"] = 1
    cfg["use_runs_dir"] = False
    run_dir = run_dir_from_config(cfg)
    ckpt_dir = ckpt_dir_from_config(cfg)
    metrics_path = metrics_file_from_config(cfg)
    last_ckpt = ckpt_dir / "last.pt"
    last_mtime_before = last_ckpt.stat().st_mtime if last_ckpt.exists() else None
    print(f"[preflight] run_dir={run_dir}")
    print(f"[preflight] ckpt_dir={ckpt_dir}")
    print(f"[preflight] metrics_path={metrics_path}")

    argv = build_argv(cfg)
    argv.extend(["--dry-run-seconds", "60"])
    print(f"[preflight] launching: {' '.join(argv)}")
    subprocess.run(argv, cwd=str(ROOT), check=False)

    if last_ckpt.exists():
        mtime_after = last_ckpt.stat().st_mtime
        updated = last_mtime_before is None or mtime_after > last_mtime_before
        print(f"[preflight] last.pt exists: {last_ckpt}")
        print(f"[preflight] last.pt updated: {updated}")
    else:
        print(f"[preflight] WARNING: last.pt not found in {ckpt_dir}")

    _print_metrics_tail(metrics_path, lines=3)


def _resume_smoke_test() -> None:
    """
    Двухпроходный тест resume:
    - первый запуск без --resume c dry-run ~45s,
    - второй запуск с --resume c dry-run ~45s,
    - проверка, что номер update продолжает расти.
    """
    base_cfg = load_config()
    cfg = base_cfg.copy()
    cfg["num_envs"] = 1
    cfg["use_runs_dir"] = False  # стабилизируем пути для smoke‑теста
    cfg["run_name"] = f"{base_cfg.get('run_name', 'auto_night')}_resume_smoke"
    run_dir = run_dir_from_config(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_file_from_config(cfg)
    ckpt_dir = ckpt_dir_from_config(cfg)

    def max_update(path: Path) -> int:
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            rows = [l.strip() for l in f if l.strip()]
        if len(rows) <= 1:
            return 0
        updates: list[int] = []
        for row in rows[1:]:
            parts = row.split(",")
            try:
                updates.append(int(parts[0]))
            except Exception:
                continue
        return max(updates) if updates else 0

    # Первый запуск: без --resume
    argv1 = build_argv(cfg, with_resume=False)
    argv1.extend(["--dry-run-seconds", "45"])
    print(f"[resume-smoke] first run (no resume): {' '.join(argv1)}")
    subprocess.run(argv1, cwd=str(ROOT), check=False)
    u1 = max_update(metrics_path)
    print(f"[resume-smoke] max update after first run: {u1}")

    # Второй запуск: с --resume
    argv2 = build_argv(cfg, with_resume=True)
    argv2.extend(["--dry-run-seconds", "45"])
    print(f"[resume-smoke] second run (with resume): {' '.join(argv2)}")
    subprocess.run(argv2, cwd=str(ROOT), check=False)
    u2 = max_update(metrics_path)
    print(f"[resume-smoke] max update after second run: {u2}")

    ok = u2 > u1 and u1 > 0
    report_path = run_dir / "preflight_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("## Resume smoke test\n\n")
        f.write(f"- metrics_path: `{metrics_path}`\n")
        f.write(f"- checkpoint_dir: `{ckpt_dir}`\n")
        f.write(f"- max_update_first_run: {u1}\n")
        f.write(f"- max_update_second_run: {u2}\n")
        f.write(f"- result: {'OK (updates increased)' if ok else 'FAIL (updates did not increase as expected)'}\n")
    print(f"[resume-smoke] report written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Night supervisor and preflight for PPO training")
    parser.add_argument("--preflight", action="store_true", help="run 60s dry-run preflight (no supervision loop)")
    parser.add_argument("--resume-smoke-test", action="store_true", help="run short two-stage resume smoke test")
    args = parser.parse_args()

    if args.preflight:
        _preflight()
        return
    if args.resume_smoke_test:
        _resume_smoke_test()
        return

    _supervisor_main()


if __name__ == "__main__":
    main()
