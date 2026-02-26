import argparse
import time
from pathlib import Path
import sys

import numpy as np

# Добавляем корень проекта в sys.path, чтобы импортировать msx_env
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.dataset import DemoMetadata, save_demo_run, validate_demo_run
from msx_env.env import EnvConfig, VampireKillerEnv, ACTION_SET_VERSION
from msx_env.human_controller import HumanController, HumanControllerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record human demo for Vampire Killer (openMSX).")
    p.add_argument("--mode", choices=["record"], default="record")
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--max-minutes", type=float, default=3.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--save-raw-every", type=int, default=0, help="зарезервировано, пока не используется")
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    run_dir = base_dir / "demos" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rom = base_dir / "VAMPIRE.ROM"
    if not rom.exists():
        raise FileNotFoundError(f"ROM not found: {rom}")

    # Важно: workdir = run_dir, чтобы все служебные файлы openMSX (логи, tcl, скриншоты)
    # и step_frame.png складывались в каталог текущего рана.
    env = VampireKillerEnv(
        EnvConfig(
            rom_path=str(rom),
            workdir=str(run_dir),
            frame_size=(84, 84),
        )
    )
    controller = HumanController(HumanControllerConfig(fps=args.fps))

    obs_list = []
    next_obs_list = []
    act_list = []
    rew_list = []
    done_list = []
    ts_list = []

    try:
        obs, info = env.reset()
        start = time.time()
        step = 0

        while True:
            now = time.time()
            if step >= args.max_steps:
                print("Max steps reached.")
                break
            if (now - start) / 60.0 >= args.max_minutes:
                print("Max minutes reached.")
                break

            action = controller.poll_action()
            if action is None:
                print("Human requested exit (Esc).")
                break

            ts = time.time()
            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
            except TimeoutError as e:
                # Типичный кейс: пользователь закрыл окно openMSX, а мы пытаемся сделать screenshot.
                print(f"Env step TimeoutError (вероятно, openMSX закрыт): {e}")
                break
            except RuntimeError as e:
                print(f"Env step RuntimeError, останавливаем запись: {e}")
                break
            done = bool(terminated or truncated)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            act_list.append(action)
            rew_list.append(reward)
            done_list.append(done)
            ts_list.append(ts)

            obs = next_obs
            step += 1

            if done:
                print("Episode finished (terminated/truncated).")
                break

    finally:
        controller.close()
        env.close()

    if not obs_list:
        print("No data collected, skipping save.")
        return

    obs_arr = np.stack(obs_list, axis=0)
    next_obs_arr = np.stack(next_obs_list, axis=0)
    actions_arr = np.asarray(act_list, dtype=np.int64)
    rewards_arr = np.asarray(rew_list, dtype=np.float32)
    dones_arr = np.asarray(done_list, dtype=np.bool_)
    ts_arr = np.asarray(ts_list, dtype=np.float64)

    meta = DemoMetadata(
        version=1,
        game_id="vampire_killer_msx2",
        machine_config="default",
        rom_path=str(rom),
        action_set_version=ACTION_SET_VERSION,
        obs_shape=tuple(obs_arr.shape[1:]),
        obs_dtype=str(obs_arr.dtype),
        step_rate_hz=float(args.fps),
        created_at=time.time(),
        total_steps=int(len(actions_arr)),
        max_minutes=float(args.max_minutes),
        run_id=args.run_id,
    )

    save_demo_run(
        run_dir,
        obs=obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        next_obs=next_obs_arr,
        dones=dones_arr,
        timestamps=ts_arr,
        meta=meta,
        extras=None,
    )

    print(f"Saved demo to {run_dir}")
    ok = validate_demo_run(run_dir, preview=args.preview)
    if ok:
        print("Demo run VALIDATION: PASS")
    else:
        print("Demo run VALIDATION: FAIL")


if __name__ == "__main__":
    main()

