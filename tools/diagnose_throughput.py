#!/usr/bin/env python3
"""
Диагностика throughput: почему при num_envs=2 steps/sec не ~2×.
Запускает матрицу микробенчмарков и выводит VERDICT с первопричиной.

Usage:
  python tools/diagnose_throughput.py --minutes 3 --rollout-steps 128 --envs 1,2
  python tools/diagnose_throughput.py --minutes 2 --modes capture_on,capture_off --policy random --train off

Output: diagnostics/<timestamp>/report.md, results.json, config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from msx_env.env import EnvConfig, VampireKillerEnv, ACTION_RIGHT, NUM_ACTIONS
from msx_env.make_env import make_env
from msx_env.reward import default_v1_config


def _percentile(lst: list[float], p: float) -> float:
    if not lst:
        return 0.0
    s = sorted(lst)
    i = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[i]


@dataclass
class TestResult:
    """Результат одного теста."""
    num_envs: int
    capture_on: bool
    policy_mode: str  # "random" | "policy"
    train_on: bool
    total_env_steps: int
    wall_time_sec: float
    env_steps_per_sec_total: float
    env_steps_per_sec_per_env: float
    update_wall_sec_p50: float
    update_wall_sec_p95: float
    update_wall_sec_mean: float
    updates_per_minute: float
    t_action_p50: float
    t_action_p95: float
    t_capture_p50: float
    t_capture_p95: float
    t_reward_p50: float
    t_reward_p95: float
    raw_perf: dict = field(default_factory=dict)


def run_single_test(
    *,
    num_envs: int,
    capture_on: bool,
    policy_mode: str,
    train_on: bool,
    minutes: float,
    rollout_steps: int,
    rom_path: str,
    tmp_root: str,
    device: str = "cpu",
    bc_checkpoint: str | None = None,
) -> TestResult | None:
    """Запустить один тест. Возвращает None при ошибке."""
    from dataclasses import replace

    base_cfg = EnvConfig(
        rom_path=rom_path,
        workdir=str(Path(tmp_root) / "0"),  # make_env переопределит per-rank
        frame_size=(84, 84),
        terminated_on_death=True,
        max_episode_steps=1500,
        reward_config=default_v1_config(),
        soft_reset=True,
        quiet=True,
        perf_profile=True,
        capture_off=not capture_on,
        post_action_delay_ms=50.0 if num_envs > 1 else 0.0,
        tmp_root=tmp_root,
    )
    if num_envs > 1:
        base_cfg = replace(
            base_cfg,
            reset_handshake_stable_frames=3,
            reset_handshake_conf_min=0.5,
            reset_handshake_timeout_s=15.0,
        )

    try:
        base_cfg = replace(base_cfg, tmp_root=tmp_root)
        env_fns = [make_env(i, base_cfg) for i in range(num_envs)]
        envs = [fn() for fn in env_fns]
    except Exception as e:
        print(f"  [SKIP] env init failed: {e}")
        return None

    model = None
    if policy_mode == "policy" or train_on:
        import torch
        from msx_env.ppo_model import ActorCritic, FRAME_STACK, init_from_bc
        arch = "deep"
        use_recurrent = False
        if bc_checkpoint:
            cp = Path(bc_checkpoint)
            if cp.exists():
                try:
                    ckpt = torch.load(str(cp), map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(str(cp), map_location="cpu")
                if isinstance(ckpt, dict) and "arch" in ckpt:
                    arch = ckpt["arch"]
        model = ActorCritic(
            num_actions=NUM_ACTIONS,
            in_channels=FRAME_STACK,
            arch=arch,
            recurrent=use_recurrent,
        ).to(device)
        if bc_checkpoint and Path(bc_checkpoint).exists():
            init_from_bc(model, bc_checkpoint)

    rng = np.random.default_rng(42)
    stack_size = 4
    frame_buffers = [deque(maxlen=stack_size) for _ in range(num_envs)]
    hidden_states = None

    def build_stacked_obs(buf: deque) -> np.ndarray:
        stacked = np.stack(list(buf), axis=0).astype(np.float32) / 255.0
        return stacked

    # Warmup: reset all envs
    try:
        for i in range(num_envs):
            if num_envs > 1 and i > 0:
                time.sleep(3.0)
            obs, _ = envs[i].reset()
            for _ in range(stack_size):
                frame_buffers[i].append(obs.copy())
    except Exception as e:
        print(f"  [SKIP] reset failed: {e}")
        for e in envs:
            try:
                e.close()
            except Exception:
                pass
        return None

    target_wall_sec = minutes * 60.0
    total_env_steps = 0
    update_times: list[float] = []
    steps_per_update = rollout_steps * num_envs

    t_start = time.perf_counter()
    update_count = 0

    while time.perf_counter() - t_start < target_wall_sec:
        t_up_start = time.perf_counter()

        for _ in range(rollout_steps):
            for i in range(num_envs):
                x = np.expand_dims(build_stacked_obs(frame_buffers[i]), axis=0)
                if policy_mode == "random":
                    action = int(rng.integers(0, NUM_ACTIONS))
                else:
                    import torch
                    x_t = torch.from_numpy(x).float().to(device)
                    with torch.no_grad():
                        action, _, _, _ = model.get_action(x_t, deterministic=False)
                    action = int(action.cpu().item())

                obs, reward, term, trunc, info = envs[i].step(action)
                done = term or trunc
                frame_buffers[i].append(obs.copy())
                total_env_steps += 1

                if done:
                    obs, _ = envs[i].reset()
                    frame_buffers[i].clear()
                    for _ in range(stack_size):
                        frame_buffers[i].append(obs.copy())

        update_count += 1
        update_times.append(time.perf_counter() - t_up_start)

        if train_on and model is not None:
            # Минимальный train pass: один batch (не полный PPO)
            import torch
            batch_obs = []
            for _ in range(min(rollout_steps * num_envs, 64)):
                i = rng.integers(0, num_envs)
                x = build_stacked_obs(frame_buffers[i])
                batch_obs.append(x)
            obs_t = torch.from_numpy(np.stack(batch_obs)).float().to(device)
            with torch.enable_grad():
                _, logp, v, _ = model.get_action(obs_t, deterministic=False)
                loss = -logp.mean() + 0.5 * (v ** 2).mean()
                loss.backward()
            # не делаем opt.step чтобы не менять модель

    wall_sec = time.perf_counter() - t_start

    # Агрегация perf stats (до close)
    all_perf: dict = {}
    for i, e in enumerate(envs):
        try:
            st = e.get_perf_stats()
            if st:
                for k, v in st.items():
                    if k not in all_perf:
                        all_perf[k] = []
                    if isinstance(v, (int, float)):
                        all_perf[k].append(v)
        except Exception:
            pass

    def get_p(key: str) -> float:
        lst = all_perf.get(key, [])
        return _percentile(lst, 95) if "_p95" in key else _percentile(lst, 50) if lst else 0.0

    t_action_p50 = get_p("t_action_send_ms_env0_p50") or get_p("t_action_send_ms_p50")
    t_action_p95 = get_p("t_action_send_ms_env0_p95") or get_p("t_action_send_ms_p95")
    t_capture_p50 = get_p("t_capture_ms_env0_p50") or get_p("t_capture_ms_p50")
    t_capture_p95 = get_p("t_capture_ms_env0_p95") or get_p("t_capture_ms_p95")
    t_reward_p50 = get_p("t_reward_ms_env0_p50") or get_p("t_reward_ms_p50")
    t_reward_p95 = get_p("t_reward_ms_env0_p95") or get_p("t_reward_ms_p95")

    up_p50 = _percentile(update_times, 50) if update_times else 0
    up_p95 = _percentile(update_times, 95) if update_times else 0
    up_mean = sum(update_times) / len(update_times) if update_times else 0
    up_per_min = 60.0 / up_mean if up_mean > 0 else 0

    return TestResult(
        num_envs=num_envs,
        capture_on=capture_on,
        policy_mode=policy_mode,
        train_on=train_on,
        total_env_steps=total_env_steps,
        wall_time_sec=wall_sec,
        env_steps_per_sec_total=total_env_steps / wall_sec if wall_sec > 0 else 0,
        env_steps_per_sec_per_env=(total_env_steps / num_envs) / wall_sec if wall_sec > 0 else 0,
        update_wall_sec_p50=up_p50,
        update_wall_sec_p95=up_p95,
        update_wall_sec_mean=up_mean,
        updates_per_minute=up_per_min,
        t_action_p50=t_action_p50,
        t_action_p95=t_action_p95,
        t_capture_p50=t_capture_p50,
        t_capture_p95=t_capture_p95,
        t_reward_p50=t_reward_p50,
        t_reward_p95=t_reward_p95,
        raw_perf=all_perf,
    )

    for e in envs:
        try:
            e.close()
        except Exception:
            pass


def compute_verdict(results: list[TestResult]) -> tuple[str, str, list[str]]:
    """Классификация первопричины. Возвращает (code, description, next_actions)."""
    by_key = {}
    for r in results:
        k = (r.num_envs, r.capture_on, r.policy_mode, r.train_on)
        by_key[k] = r

    r1 = by_key.get((1, True, "random", False))
    r2 = by_key.get((2, True, "random", False))
    r1_cap_off = by_key.get((1, False, "random", False))
    r2_cap_off = by_key.get((2, False, "random", False))
    r1_policy = by_key.get((1, True, "policy", False))
    r2_policy = by_key.get((2, True, "policy", False))
    r1_train = by_key.get((1, True, "policy", True))
    r2_train = by_key.get((2, True, "policy", True))

    next_actions: list[str] = []
    scaling_1_to_2 = 0.0
    if r1 and r2 and r1.env_steps_per_sec_total > 0:
        scaling_1_to_2 = r2.env_steps_per_sec_total / r1.env_steps_per_sec_total

    # S7: METRICS MISCOUNT — total_env_steps считает только env0?
    # Проверка: при num_envs=2 total должен быть 2*rollout_steps*updates
    # Эвристика: если r2.steps ≈ r1.steps при равном wall time — miscount
    if r1 and r2 and r1.wall_time_sec > 10 and r2.wall_time_sec > 10:
        steps_ratio = r2.total_env_steps / max(1, r1.total_env_steps)
        if steps_ratio < 1.2:  # ожидаем ~2x
            return (
                "S7_METRICS_MISCOUNT",
                "total_env_steps не учитывает все env (steps envs=2 ≈ steps envs=1)",
                ["Проверить: episode_steps_list и total_steps считают sum по всем env, не только env0"],
            )

    # S1: SEQUENTIAL STEPPING
    if scaling_1_to_2 > 0 and scaling_1_to_2 < 1.4:
        # Почти нет масштабирования
        if r2 and r2.t_action_p95 > 0 and r2.t_capture_p95 > 0:
            step_total_ms = (r2.t_action_p95 + r2.t_capture_p95 + r2.t_reward_p95) * 2  # env0+env1
            update_steps = 256  # rollout_steps*2
            expected_update_sec = (step_total_ms / 1000) * update_steps / 2  # sequential
            if r2.update_wall_sec_mean > 0 and r2.update_wall_sec_mean >= expected_update_sec * 0.8:
                return (
                    "S1_SEQUENTIAL_STEPPING",
                    f"envs=2 не даёт 2× throughput (scaling={scaling_1_to_2:.2f}). Шаги env выполняются последовательно.",
                    [
                        "Вынести env в subprocess (multiprocessing) или AsyncVectorEnv",
                        "Рассмотреть параллельный stepping: env0.step и env1.step в разных потоках",
                    ],
                )

    # S2: CAPTURE BOTTLENECK
    if r1 and r1_cap_off and r1.env_steps_per_sec_total > 0 and r1_cap_off.env_steps_per_sec_total > 0:
        cap_ratio = r1_cap_off.env_steps_per_sec_total / r1.env_steps_per_sec_total
        if cap_ratio > 1.6 and r1.t_capture_p95 > max(r1.t_action_p95, r1.t_reward_p95):
            return (
                "S2_CAPTURE_BOTTLENECK",
                f"capture_off даёт {cap_ratio:.1f}× ускорение. t_capture dominates.",
                [
                    "Перейти на window capture вместо PNG (dxcam/mss)",
                    "Уменьшить разрешение скриншота",
                    "Кэшировать кадр при быстрых step (если допустимо)",
                ],
            )

    # S3: LOGGING/IO BOTTLENECK — косвенно: если t_logging доминирует (у нас нет t_logging в env)
    # Пропускаем, т.к. в diagnose мы не пишем metrics.csv

    # S4: INFERENCE BOTTLENECK
    if r1 and r1_policy and r1.env_steps_per_sec_total > 0 and r1_policy.env_steps_per_sec_total > 0:
        inf_ratio = r1.env_steps_per_sec_total / r1_policy.env_steps_per_sec_total
        if inf_ratio > 1.5:
            return (
                "S4_INFERENCE_BOTTLENECK",
                f"random policy в {inf_ratio:.1f}× быстрее policy. Forward pass доминирует.",
                [
                    "Батчить inference (не по 1 obs, а по batch)",
                    "Квантизация модели, torch.compile",
                    "GPU для inference",
                ],
            )

    # S5: TRAIN BOTTLENECK
    if r1_policy and r1_train and r1_policy.env_steps_per_sec_total > 0 and r1_train.env_steps_per_sec_total > 0:
        train_ratio = r1_policy.env_steps_per_sec_total / r1_train.env_steps_per_sec_total
        if train_ratio > 1.5:
            return (
                "S5_TRAIN_BOTTLENECK",
                f"train off в {train_ratio:.1f}× быстрее. PPO update доминирует.",
                [
                    "Уменьшить ppo_epochs, batch_size",
                    "GPU для train",
                    "Увеличить rollout_steps относительно train time",
                ],
            )

    # S6: THROTTLE/REALTIME SYNC
    if r1 and r1.env_steps_per_sec_total > 0:
        # 60Hz limit = 60 steps/sec на 1 env
        if r1.env_steps_per_sec_per_env < 70 and r1.env_steps_per_sec_per_env > 50:
            return (
                "S6_THROTTLE_REALTIME_SYNC",
                f"Throughput ~{r1.env_steps_per_sec_per_env:.0f}/sec per env — похоже на 60Hz limit.",
                [
                    "Проверить decision_fps, throttle в openMSX",
                    "Убедиться, что openMSX не в realtime mode",
                ],
            )

    # Default
    if scaling_1_to_2 > 0 and scaling_1_to_2 < 1.5:
        return (
            "S1_LIKELY_SEQUENTIAL",
            f"Слабое масштабирование (scaling={scaling_1_to_2:.2f}). Требуется детальный разбор таймеров.",
            [
                "Проверить Top 3 time sinks в report",
                "Запустить с --modes capture_on,capture_off для проверки S2",
            ],
        )

    return (
        "UNKNOWN",
        "Недостаточно данных для однозначной классификации.",
        ["Запустите полную матрицу: --envs 1,2 --modes capture_on,capture_off --policy random,policy --train off,on"],
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose throughput scaling")
    p.add_argument("--minutes", type=float, default=2.0, help="Minutes per test")
    p.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout")
    p.add_argument("--envs", type=str, default="1,2", help="Comma-separated: 1,2")
    p.add_argument("--modes", type=str, default="capture_on", help="capture_on,capture_off")
    p.add_argument("--policy", type=str, default="random", help="random,policy")
    p.add_argument("--train", type=str, default="off", help="off,on")
    p.add_argument("--rom", type=str, default=None)
    p.add_argument("--bc-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    rom = Path(args.rom) if args.rom else ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        print(f"ERROR: ROM not found: {rom}")
        return 2

    env_list = [int(x.strip()) for x in args.envs.split(",") if x.strip()]
    modes_list = [x.strip() for x in args.modes.split(",") if x.strip()]
    capture_modes = []
    for m in modes_list:
        if m == "capture_on":
            capture_modes.append(True)
        elif m == "capture_off":
            capture_modes.append(False)
        else:
            print(f"Unknown mode: {m}")
    if not capture_modes:
        capture_modes = [True]

    policy_modes = [x.strip() for x in args.policy.split(",") if x.strip()]
    train_modes = [x.strip() for x in args.train.split(",") if x.strip()]
    train_bools = [t == "on" for t in train_modes]

    out_dir = ROOT / "diagnostics" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = str((out_dir / "tmp").resolve())
    Path(tmp_root).mkdir(parents=True, exist_ok=True)

    print("=== Throughput diagnostics ===\n")
    print(f"Output: {out_dir}\n")

    results: list[TestResult] = []
    for num_envs in env_list:
        for cap_on in capture_modes:
            for pol in policy_modes:
                for tr_on in train_bools:
                    tag = f"envs={num_envs} capture={'on' if cap_on else 'off'} policy={pol} train={'on' if tr_on else 'off'}"
                    print(f"Running: {tag} ...", end=" ", flush=True)
                    r = run_single_test(
                        num_envs=num_envs,
                        capture_on=cap_on,
                        policy_mode=pol,
                        train_on=tr_on,
                        minutes=args.minutes,
                        rollout_steps=args.rollout_steps,
                        rom_path=str(rom.resolve()),
                        tmp_root=tmp_root,
                        device=args.device,
                        bc_checkpoint=args.bc_checkpoint,
                    )
                    if r:
                        results.append(r)
                        print(f"{r.env_steps_per_sec_total:.1f} steps/s")
                    else:
                        print("SKIP")

    # VERDICT
    verdict_code, verdict_desc, next_actions = compute_verdict(results)

    # Top 3 time sinks (from first result with perf data)
    top_sinks: list[str] = []
    for r in results:
        if r.raw_perf:
            candidates = []
            for k in ["t_action_send_ms_env0_p95", "t_capture_ms_env0_p95", "t_reward_ms_env0_p95"]:
                v = r.raw_perf.get(k, 0)
                if isinstance(v, (int, float)):
                    candidates.append((k.replace("_env0_p95", ""), float(v)))
            candidates.sort(key=lambda x: -x[1])
            top_sinks = [f"{c[0]}: {c[1]:.1f}ms" for c in candidates[:3]]
            break

    # Report
    config = {
        "minutes": args.minutes,
        "rollout_steps": args.rollout_steps,
        "envs": env_list,
        "modes": modes_list,
        "policy": policy_modes,
        "train": train_modes,
        "rom": str(rom),
    }

    results_ser = []
    for r in results:
        results_ser.append({
            "num_envs": r.num_envs,
            "capture_on": r.capture_on,
            "policy_mode": r.policy_mode,
            "train_on": r.train_on,
            "total_env_steps": r.total_env_steps,
            "wall_time_sec": r.wall_time_sec,
            "env_steps_per_sec_total": r.env_steps_per_sec_total,
            "env_steps_per_sec_per_env": r.env_steps_per_sec_per_env,
            "update_wall_sec_p50": r.update_wall_sec_p50,
            "update_wall_sec_p95": r.update_wall_sec_p95,
            "updates_per_minute": r.updates_per_minute,
            "t_action_p50": r.t_action_p50,
            "t_action_p95": r.t_action_p95,
            "t_capture_p50": r.t_capture_p50,
            "t_capture_p95": r.t_capture_p95,
            "t_reward_p50": r.t_reward_p50,
            "t_reward_p95": r.t_reward_p95,
        })

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": results_ser, "verdict": {"code": verdict_code, "description": verdict_desc, "next_actions": next_actions}}, f, indent=2)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Expected vs observed scaling
    r1 = next((x for x in results if x.num_envs == 1 and x.capture_on and x.policy_mode == "random" and not x.train_on), None)
    r2 = next((x for x in results if x.num_envs == 2 and x.capture_on and x.policy_mode == "random" and not x.train_on), None)
    scaling_str = "N/A"
    if r1 and r2 and r1.env_steps_per_sec_total > 0:
        obs = r2.env_steps_per_sec_total / r1.env_steps_per_sec_total
        scaling_str = f"Expected 2.0×, observed {obs:.2f}×"

    report = f"""# Throughput diagnostics report

## Config
- Minutes per test: {args.minutes}
- Rollout steps: {args.rollout_steps}
- Envs: {env_list}
- Modes: {modes_list}

## Results

| num_envs | capture | policy | train | steps/s total | steps/s per_env | update_sec p50 | update_sec p95 |
|----------|---------|--------|-------|---------------|-----------------|----------------|----------------|
"""
    for r in results:
        report += f"| {r.num_envs} | {'on' if r.capture_on else 'off'} | {r.policy_mode} | {'on' if r.train_on else 'off'} | {r.env_steps_per_sec_total:.1f} | {r.env_steps_per_sec_per_env:.1f} | {r.update_wall_sec_p50:.2f} | {r.update_wall_sec_p95:.2f} |\n"

    report += f"""
## Expected vs observed scaling (envs 1→2, random, capture_on, train off)
{scaling_str}

## Top 3 time sinks (p95 ms)
{chr(10).join('- ' + s for s in top_sinks) if top_sinks else '- (no data)'}

## VERDICT
**{verdict_code}**

{verdict_desc}

### Next action
{chr(10).join('- ' + a for a in next_actions)}
"""
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("VERDICT:", verdict_code)
    print(verdict_desc)
    print("\nNext action:")
    for a in next_actions:
        print("  -", a)
    print(f"\nReport: {out_dir / 'report.md'}")
    print(f"Results: {out_dir / 'results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
