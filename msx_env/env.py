"""
Окружение Vampire Killer (MSX2) для RL.

VampireKillerEnv — Gym-подобный env: reset() (интро/продолжение), step(action) (клавиши → скриншот → obs, reward, done, info).
Управление openMSX через openmsx_bridge (commands.tcl / reply.txt). Захват кадра — msx_env.capture (png/single/window).
Поддержка нескольких инстансов (num_envs): у каждого свой workdir, при reset() по умолчанию мягкий сброс (soft_reset),
без перезапуска процесса openMSX. Награды: legacy (HUD + death) или модульная RewardPolicy (reward_config); см. docs/REWARD.md и docs/TRAINING.md.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
from PIL import Image

from openmsx_bridge import OpenMSXFileControl

from msx_env.capture import FrameCaptureBackend, make_capture_backend
from msx_env.hud_parser import HudState, compute_pickup_reward, parse_hud, parse_stage
from msx_env.life_bar import get_life_estimate
from msx_env.reward import EpisodeRewardLogger, RewardPolicy
from msx_env.reward.config import RewardConfig
from msx_env.env_diagnostics import (
    print_reset_block,
    print_step_control,
    print_termination_block,
    register_and_assert_resources,
)
from msx_env.reward.hashers import frame_diff_metric, block_mean_hash


# Дискретный экшн‑спейс для Vampire Killer
ACTION_NOOP = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_ATTACK = 5
ACTION_RIGHT_JUMP = 6
ACTION_LEFT_JUMP = 7
ACTION_RIGHT_JUMP_ATTACK = 8
ACTION_LEFT_JUMP_ATTACK = 9

ACTION_SET_VERSION = "vkiller_v1"
NUM_ACTIONS = 10

_DEBUG_FORCE_ACTION_MAP = {
    "RIGHT": ACTION_RIGHT,
    "LEFT": ACTION_LEFT,
    "UP": ACTION_UP,
    "DOWN": ACTION_DOWN,
    "SPACE": ACTION_ATTACK,
    "RIGHT_JUMP": ACTION_RIGHT_JUMP,
    "RIGHT+UP": ACTION_RIGHT_JUMP,
    "LEFT_JUMP": ACTION_LEFT_JUMP,
    "LEFT+UP": ACTION_LEFT_JUMP,
}


def _debug_force_action_to_id(force_name: str, fallback: int) -> int:
    return _DEBUG_FORCE_ACTION_MAP.get(force_name, fallback)


def _action_id_to_name(aid: int) -> str:
    names = {
        0: "NOOP", 1: "RIGHT", 2: "LEFT", 3: "UP", 4: "DOWN", 5: "SPACE",
        6: "RIGHT_JUMP", 7: "LEFT_JUMP", 8: "RIGHT_JUMP_ATTACK", 9: "LEFT_JUMP_ATTACK",
    }
    return names.get(aid, f"A{aid}")


@dataclass
class EnvConfig:
    rom_path: str
    workdir: str
    frame_size: Tuple[int, int] = (84, 84)
    poll_ms: int = 15  # интервал опроса openMSX (меньше = быстрее реакция)
    hold_keys: bool = True  # держать клавиши нажатыми до смены действия (непрерывное движение)
    hud_reward: bool = True  # reward за подбор предметов (оружие, ключи, прочее) по HUD
    # Режим RL: terminated при смерти, truncated при max_episode_steps (0 = без лимита)
    terminated_on_death: bool = False  # True: при падении жизни — terminated
    max_episode_steps: int = 0  # >0: truncated при достижении (для RL)
    # Performance: action_repeat=1 (backward compat), decision_fps=None (no throttle)
    action_repeat: int = 1  # N внутренних шагов эмулятора на 1 захват кадра
    decision_fps: float | None = None  # фиксированная частота решений (10–15 Hz)
    capture_backend: str = "png"  # png | single | window
    window_crop: Tuple[int, int, int, int] | None = None  # (x, y, w, h) для window backend
    window_title: str | None = None  # подстрока заголовка окна (по умолчанию openMSX)
    capture_lag_ms: float = 0  # задержка перед grab (уменьшение tearing)
    post_action_delay_ms: float = 0  # задержка после нажатия клавиш перед grab (дать эмулятору отрисовать кадр; при num_envs>1 полезно 40–50)
    reward_config: RewardConfig | None = None  # None = legacy (hud + death в env, step_penalty в скрипте)
    # Multi-instance (Phase 3): уникальный workdir на инстанс
    instance_id: int | str | None = None  # для логов и путей
    tmp_root: str = "runs/tmp"  # база для workdir при num_envs > 1
    soft_reset: bool = True  # при reset() не убивать процесс openMSX, а «продолжить» клавишами (нужно при num_envs>1)
    perf_log_interval: int = 0  # каждые N шагов печатать [perf] (capture/prep/step); 0 = отключено
    quiet: bool = False  # не печатать отчёт эпизода и лишний вывод (ночной режим)
    openmsx_log_dir: str | None = None  # каталог для openmsx_<instance>.log; None = workdir
    # Debug (opt-in): диагностика room_hash, stage, stuck, действия
    debug: bool = False
    debug_every: int = 10  # печатать компактную строку каждые N шагов
    debug_episode_max_steps: int = 0  # 0 = не менять; >0 = ограничить эпизод для короткого прогона
    debug_dump_frames: int = 0  # сохранить первые N кадров в debug_frames/
    debug_force_action: str = ""  # "RIGHT", "LEFT", ... — подменять действие для проверки
    debug_dump_dir: str | None = None  # каталог для debug_frames; None = workdir
    debug_room_change: bool = False  # печатать [ENV i] room change prev -> new при смене комнаты
    ignore_death: bool = False  # для теста: не ставить terminated=True при смерти, только логировать
    death_warmup_steps: int = 50  # в первых N шагах не объявлять смерть (защита от ложных срабатываний по кадру)
    # Reset handshake (multi-env): ждать стабильного stage перед возвратом из reset()
    reset_handshake_stable_frames: int = 0  # 0 = выкл; N = ждать N подряд кадров с stage_conf >= conf_min
    reset_handshake_conf_min: float = 0.5  # минимальная уверенность stage для «стабильного» кадра
    reset_handshake_timeout_s: float = 15.0  # макс. время ожидания стабильного кадра
    # fix-room-metrics-stability: dump HUD crop для отладки stage detector
    dump_hud_every_n_steps: int = 0  # 0 = выкл; N = сохранять HUD crop каждые N шагов
    dump_hud_dir: str | None = None  # каталог для hud_env{X}_step{t}.png; None = не сохранять
    # Throughput diagnostics (diagnose_throughput.py): включаются только при явном флаге
    perf_profile: bool = False  # собирать t_action_ms, t_capture_ms, t_preproc_ms, t_reward_ms
    capture_off: bool = False  # при True: не вызывать grab(), возвращать кэш/zeros (для benchmark capture_on vs off)


class VampireKillerEnv:
    """
    Минимальный Gym‑подобный env вокруг OpenMSXFileControl.

    reset() -> obs, info
    step(action) -> obs, reward, terminated, truncated, info
    """

    # Маппинг action -> клавиши для удержания (stateful режим)
    _ACTION_KEYS: dict[int, tuple[str, ...]] = {
        0: (),  # NOOP
        1: ("RIGHT",),
        2: ("LEFT",),
        3: ("UP",),
        4: ("DOWN",),
        5: ("SPACE",),
        6: ("RIGHT", "UP"),
        7: ("LEFT", "UP"),
        8: ("RIGHT", "UP", "SPACE"),
        9: ("LEFT", "UP", "SPACE"),
    }

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self._emu: OpenMSXFileControl | None = None
        self._capture: FrameCaptureBackend | None = None
        self._keys_down: set[str] = set()
        self._prev_hud = None
        self._life_prev: float = 1.0
        self._step_count: int = 0
        self._perf_timings: deque[Dict[str, float]] = deque(maxlen=100)
        self._perf_log_interval: int = getattr(cfg, "perf_log_interval", 0)
        rc = getattr(cfg, "reward_config", None)
        self._reward_policy: RewardPolicy | None = RewardPolicy(rc) if rc is not None else None
        self._reward_logger: EpisodeRewardLogger | None = EpisodeRewardLogger() if self._reward_policy is not None else None
        self._prev_obs: np.ndarray | None = None
        # Debug state (сбрасывается в reset при debug)
        self._debug_prev_room_hash: str | None = None
        self._debug_room_changes: int = 0
        self._debug_stage_changes: int = 0
        self._debug_stuck_events: int = 0
        self._debug_steps_since_room: int = 0
        self._debug_dumped_count: int = 0
        self._death_low_life_steps: int = 0  # гистерезис: смерть только после 2 подряд кадров с life < 0.15
        # perf_profile: diagnose_throughput.py
        self._perf_acc = None
        if getattr(cfg, "perf_profile", False):
            from msx_env.perf_timers import PerfTimerAccumulator
            self._perf_acc = PerfTimerAccumulator()
        self._capture_off_cache_rgb: np.ndarray | None = None
        self._capture_off_cache_obs: np.ndarray | None = None

    def get_perf_stats(self) -> dict | None:
        """Возвращает агрегированную статистику таймеров (только при perf_profile=True)."""
        if self._perf_acc is None:
            return None
        return self._perf_acc.get_stats()

    # --------- низкоуровневые helpers ---------

    def _ensure_emu(self) -> OpenMSXFileControl:
        if self._emu is None:
            self._emu = OpenMSXFileControl(
                rom_path=self.cfg.rom_path,
                workdir=self.cfg.workdir,
                poll_ms=getattr(self.cfg, "poll_ms", 15),
                boot_timeout_s=30.0,
                log_dir=getattr(self.cfg, "openmsx_log_dir", None),
                instance_id=getattr(self.cfg, "instance_id", None),
            )
        return self._emu

    def _ensure_capture(self) -> FrameCaptureBackend:
        if self._capture is None and self._emu is not None:
            backend_name = getattr(self.cfg, "capture_backend", "png")
            kwargs = {}
            if backend_name in ("window", "dxcam"):
                kwargs["window_title"] = getattr(self.cfg, "window_title", None)
                kwargs["capture_lag_ms"] = getattr(self.cfg, "capture_lag_ms", 0)
            if backend_name == "window":
                kwargs["window_crop"] = getattr(self.cfg, "window_crop", None)
            self._capture = make_capture_backend(
                backend_name,
                self._emu,
                self.cfg.workdir,
                "step_frame.png",
                **kwargs,
            )
            self._capture.start()
        if self._capture is None:
            raise RuntimeError("emulator not initialized")
        return self._capture

    def _skip_intro(self, emu: OpenMSXFileControl) -> None:
        # Ждём ухода BIOS‑экрана (in-memory через capture — без скриншотов на диск)
        get_frame = None
        if self._capture is not None:
            get_frame = lambda: self._capture.grab()
        emu.wait_for_nonblue_screen(
            timeout_s=30.0, check_every_s=0.5, get_frame=get_frame
        )
        time.sleep(1.0)  # дать заголовку/меню стабилизироваться
        # Пропустить заставку Konami
        emu.press_type("SPACE")
        time.sleep(2.0)
        # PRESS START — запуск игры
        emu.press_type("SPACE")
        time.sleep(3.0)  # загрузка уровня

    def _soft_reset(self, emu: OpenMSXFileControl) -> None:
        """Продолжить после Game Over без перезапуска процесса: отпустить клавиши, SPACE для continue, загрузка уровня."""
        if self._keys_down:
            emu.keyup(list(self._keys_down))
            self._keys_down = set()
        time.sleep(1.0)
        emu.press_type("SPACE")
        time.sleep(2.0)
        emu.press_type("SPACE")
        time.sleep(3.0)

    def _grab_frame_and_obs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Захват через бэкенд: RGB (H,W,3) и obs (84,84) uint8 grayscale.
        Один вызов grab — без двойного чтения PNG.
        capture_off=True: пропуск grab, возврат кэша или zeros.
        """
        capture_off = getattr(self.cfg, "capture_off", False)
        if capture_off:
            if self._capture_off_cache_rgb is not None:
                return self._capture_off_cache_rgb.copy(), self._capture_off_cache_obs.copy()
            # первый раз: минимальный placeholder (256x192 типичный MSX)
            h, w = 192, 256
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            img = Image.fromarray(rgb).resize(self.cfg.frame_size).convert("L")
            obs = np.array(img, dtype=np.uint8)
            self._capture_off_cache_rgb = rgb
            self._capture_off_cache_obs = obs
            return rgb.copy(), obs.copy()

        capture = self._ensure_capture()
        t0 = time.perf_counter()
        rgb = capture.grab()
        capture_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        w, h = self.cfg.frame_size
        img = Image.fromarray(rgb).resize((w, h)).convert("L")
        obs = np.array(img, dtype=np.uint8)
        preprocess_time = time.perf_counter() - t1

        self._last_capture_time = capture_time
        self._last_preprocess_time = preprocess_time
        return rgb, obs

    def _apply_action(self, emu: OpenMSXFileControl, action: int) -> None:
        if getattr(self.cfg, "hold_keys", True):
            # Режим удержания: клавиши нажаты, пока действие то же
            needed = set(self._ACTION_KEYS.get(action, ()))
            to_release = self._keys_down - needed
            to_press = needed - self._keys_down
            if to_release:
                emu.keyup(list(to_release))
                self._keys_down -= to_release
            if to_press:
                emu.keydown(list(to_press))
                self._keys_down |= to_press
        else:
            # Импульсный режим (как раньше): press с hold_ms
            if action == ACTION_NOOP:
                return
            if self._keys_down:
                emu.keyup(list(self._keys_down))
                self._keys_down = set()
            if action == ACTION_RIGHT:
                emu.press("RIGHT", hold_ms=100)
            elif action == ACTION_LEFT:
                emu.press("LEFT", hold_ms=100)
            elif action == ACTION_UP:
                emu.press("UP", hold_ms=180)
            elif action == ACTION_DOWN:
                emu.press("DOWN", hold_ms=180)
            elif action == ACTION_ATTACK:
                emu.press("SPACE", hold_ms=100)
            elif action == ACTION_RIGHT_JUMP:
                emu.press_combo(["RIGHT", "UP"], hold_ms=200)
            elif action == ACTION_LEFT_JUMP:
                emu.press_combo(["LEFT", "UP"], hold_ms=200)
            elif action == ACTION_RIGHT_JUMP_ATTACK:
                emu.press_combo(["RIGHT", "UP", "SPACE"], hold_ms=200)
            elif action == ACTION_LEFT_JUMP_ATTACK:
                emu.press_combo(["LEFT", "UP", "SPACE"], hold_ms=200)
            else:
                raise ValueError(f"Unknown action id: {action}")

    # --------- публичный API ---------

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """Сброс эпизода. При soft_reset=True процесс openMSX не перезапускается — только «продолжить» клавишами."""
        soft = getattr(self.cfg, "soft_reset", True)
        if self._emu is None:
            # Первый запуск: поднять процесс, capture для in-memory проверки интро, пройти интро
            emu = self._ensure_emu()
            self._ensure_capture()
            self._skip_intro(emu)
        elif soft:
            # Мягкий сброс: не закрывать процесс, «продолжить» после Game Over
            emu = self._emu
            self._soft_reset(emu)
            if self._capture is not None:
                try:
                    self._capture.close()
                except Exception:
                    pass
                self._capture = None
        else:
            if self._capture is not None:
                try:
                    self._capture.close()
                except Exception:
                    pass
                self._capture = None
            if self._keys_down:
                self._emu.keyup(list(self._keys_down))
                self._keys_down = set()
            self._emu.close()
            self._emu = None
            emu = self._ensure_emu()
            self._ensure_capture()
            self._skip_intro(emu)
        rgb, obs = self._grab_frame_and_obs()
        # Optional reset handshake: wait for stable stage (multi-env)
        stable_n = getattr(self.cfg, "reset_handshake_stable_frames", 0)
        if stable_n > 0:
            conf_min = getattr(self.cfg, "reset_handshake_conf_min", 0.5)
            timeout_s = getattr(self.cfg, "reset_handshake_timeout_s", 15.0)
            t0 = time.perf_counter()
            stable_count = 0
            while stable_count < stable_n:
                if time.perf_counter() - t0 > timeout_s:
                    if getattr(self.cfg, "debug", False):
                        print(f"[debug] ENV instance_id={getattr(self.cfg, 'instance_id', '?')} reset handshake TIMEOUT (using last frame)")
                    break
                try:
                    stage_int, stage_conf = parse_stage(rgb)
                    if stage_conf >= conf_min:
                        stable_count += 1
                    else:
                        stable_count = 0
                except Exception:
                    stable_count = 0
                if stable_count < stable_n:
                    time.sleep(0.1)
                    rgb, obs = self._grab_frame_and_obs()
            if getattr(self.cfg, "debug", False):
                print(f"READY env_id={getattr(self.cfg, 'instance_id', '?')} (reset handshake done)")
        if getattr(self.cfg, "hud_reward", True):
            try:
                self._prev_hud = parse_hud(rgb)
            except Exception:
                self._prev_hud = None
        self._life_prev = get_life_estimate(obs)
        self._step_count = 0
        self._prev_obs = obs.copy()
        if getattr(self.cfg, "debug", False):
            self._debug_prev_room_hash = None
            self._debug_room_changes = 0
            self._debug_stage_changes = 0
            self._debug_stuck_events = 0
            self._debug_steps_since_room = 0
            self._debug_prev_stage = None
        self._death_low_life_steps = 0
        if self._reward_policy is not None:
            self._reward_policy.reset()
            self._reward_logger.reset()
        info: Dict[str, Any] = {}
        if getattr(self.cfg, "debug", False):
            register_and_assert_resources(self)
            print_reset_block(self, obs, rgb)
        return obs, info

    def step(self, action: int):
        """Применить действие (keydown/keyup), опционально post_action_delay_ms, захват кадра, награда, done."""
        t_step_start = time.perf_counter()
        perf = self._perf_acc is not None
        env_id = getattr(self.cfg, "instance_id", 0)

        force = (getattr(self.cfg, "debug_force_action", "") or "").strip().upper()
        if force:
            action = _debug_force_action_to_id(force, action)
        emu = self._ensure_emu()
        action_repeat = getattr(self.cfg, "action_repeat", 1)
        decision_fps = getattr(self.cfg, "decision_fps", None)
        step_duration_s = 1.0 / decision_fps if decision_fps and decision_fps > 0 else 0.0

        t_action_start = time.perf_counter() if perf else 0
        if getattr(self.cfg, "debug", False) and self._step_count < 3:
            print_step_control(self, emu)
        self._apply_action(emu, int(action))

        for _ in range(action_repeat - 1):
            if step_duration_s > 0:
                time.sleep(step_duration_s)
            self._apply_action(emu, int(action))

        delay_ms = getattr(self.cfg, "post_action_delay_ms", 0.0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        if perf:
            t_action_ms = (time.perf_counter() - t_action_start) * 1000
            self._perf_acc.record("t_action_send_ms", t_action_ms, env_id)

        t_grab_start = time.perf_counter() if perf else 0
        rgb, obs = self._grab_frame_and_obs()
        if perf:
            t_capture_preproc_ms = (time.perf_counter() - t_grab_start) * 1000
            self._perf_acc.record("t_capture_ms", t_capture_preproc_ms, env_id)
        # fix-room-metrics-stability: dump HUD crop для отладки stage detector
        dump_n = getattr(self.cfg, "dump_hud_every_n_steps", 0)
        dump_dir = getattr(self.cfg, "dump_hud_dir", None)
        next_step = self._step_count + 1
        if dump_n > 0 and dump_dir and next_step % dump_n == 0:
            try:
                from pathlib import Path
                d = Path(dump_dir)
                d.mkdir(parents=True, exist_ok=True)
                eid = getattr(self.cfg, "instance_id", 0)
                fn = d / f"hud_env{eid}_step{next_step}.png"
                hud_crop = rgb[: min(24, rgb.shape[0]), :, :]
                Image.fromarray(hud_crop).save(fn)
            except Exception:
                pass
        reward = 0.0
        info: Dict[str, Any] = {}
        env_id = getattr(self.cfg, "instance_id", None)
        if env_id is not None:
            info["env_id"] = env_id

        terminated_by_death = False
        life: float = 0.5
        life_prev_for_signal: float = self._life_prev
        if getattr(self.cfg, "terminated_on_death", False):
            life = get_life_estimate(obs)
            warmup = getattr(self.cfg, "death_warmup_steps", 50)
            if self._step_count < warmup:
                # В warmup не объявляем смерть — артефакты кадра/переходов дают ложные life≈0
                self._death_low_life_steps = 0
                self._life_prev = life
            else:
                if life < 0.15:
                    self._death_low_life_steps += 1
                    if self._death_low_life_steps >= 2 and self._life_prev > 0.3:
                        terminated_by_death = True
                else:
                    self._death_low_life_steps = 0
                self._life_prev = life

        t_reward_start = time.perf_counter() if perf else 0
        if self._reward_policy is not None:
            total_reward, components, extra = self._reward_policy.compute(
                self._prev_obs,
                obs,
                info,
                terminated=False,
                truncated=False,
                terminated_by_death=terminated_by_death,
                rgb_for_hud=rgb,
            )
            reward = total_reward
            info["reward_components"] = components
            self._reward_logger.add_step(reward, components, extra)
            info["episode_reward_components"] = self._reward_logger.get_episode_components()
            for k, v in extra.items():
                info[f"reward_{k}"] = v
            try:
                curr_hud = parse_hud(rgb)
                info["hud"] = {"weapon": curr_hud.weapon, "key_chest": curr_hud.key_chest, "key_door": curr_hud.key_door, "items": curr_hud.items}
            except Exception:
                pass
            try:
                stage_int, stage_conf = parse_stage(rgb)
                info["stage"] = stage_int
                info["stage_conf"] = stage_conf
            except Exception:
                info["stage"] = 0
                info["stage_conf"] = 0.0
        else:
            if getattr(self.cfg, "hud_reward", True):
                try:
                    curr_hud = parse_hud(rgb)
                    reward = compute_pickup_reward(self._prev_hud, curr_hud)
                    self._prev_hud = curr_hud
                    info["hud"] = {
                        "weapon": curr_hud.weapon,
                        "key_chest": curr_hud.key_chest,
                        "key_door": curr_hud.key_door,
                        "items": curr_hud.items,
                    }
                except Exception:
                    pass
            try:
                stage_int, stage_conf = parse_stage(rgb)
                info["stage"] = stage_int
                info["stage_conf"] = stage_conf
            except Exception:
                info["stage"] = 0
                info["stage_conf"] = 0.0
            if terminated_by_death:
                reward = reward - 1.0

        if perf:
            t_reward_ms = (time.perf_counter() - t_reward_start) * 1000
            self._perf_acc.record("t_reward_ms", t_reward_ms, env_id)

        terminated = terminated_by_death
        if getattr(self.cfg, "ignore_death", False) and terminated_by_death:
            terminated = False
            info["death_detected_but_ignored"] = True
            raw_signal = {"life": life, "life_prev": life_prev_for_signal, "death_threshold": 0.15, "prev_threshold": 0.3}
            info["termination_reason"] = "death_ignored"
            info["raw_death_signals"] = raw_signal
            info["hp_value"] = life
            info["lives_value"] = None
            info["gameover_flag"] = None
            info["stage"] = info.get("stage", 0)
            info["stage_conf"] = info.get("stage_conf", 0.0)
            if getattr(self.cfg, "debug", False):
                print(
                    f"[debug] death detected but ignored (ignore_death=True) step={self._step_count} hp={life} "
                    f"life_prev={life_prev_for_signal} raw_signals={raw_signal}"
                )
        truncated = False
        self._step_count += 1

        if self._reward_policy is not None and info.get("reward_stuck_truncate"):
            truncated = True
        max_steps = getattr(self.cfg, "max_episode_steps", 0)
        if max_steps > 0 and self._step_count >= max_steps:
            truncated = True

        # Termination diagnostics: normalize reason to death|stuck|timeout|reset_handshake_fail|unknown
        if terminated or truncated:
            reason_parts = []
            if terminated_by_death and not getattr(self.cfg, "ignore_death", False):
                reason_parts.append("death")
            if info.get("reward_stuck_truncate"):
                reason_parts.append("stuck")
            if max_steps > 0 and self._step_count >= max_steps:
                reason_parts.append("timeout")
            if reason_parts:
                info["termination_reason"] = reason_parts[0] if len(reason_parts) == 1 else "+".join(reason_parts)
            else:
                info["termination_reason"] = "unknown"
            info["raw_death_signals"] = {
                "life": life,
                "life_prev": life_prev_for_signal,
                "death_threshold": 0.15,
                "prev_threshold": 0.3,
            }
            info["hp_value"] = life
            info["lives_value"] = None
            info["gameover_flag"] = None
            info["stage"] = info.get("stage", 0)
            info["stage_conf"] = info.get("stage_conf", 0.0)
            info["death_detector"] = "life_bar_v1"  # get_life_estimate + thresholds 0.15 / 0.3

        frame_diff = frame_diff_metric(self._prev_obs, obs) if self._prev_obs is not None else 0.0
        room_hash = info.get("reward_room_hash")
        if getattr(self.cfg, "debug_room_change", False) and info.get("reward_room_transition_event"):
            prev_h = info.get("reward_prev_room_hash", "")
            prev_short = (prev_h or "")[:12] if prev_h else "-"
            new_short = (room_hash or "")[:12] if room_hash else "-"
            eid = env_id if env_id is not None else 0
            stage_curr = info.get("stage", 0)
            novelty_rooms = info.get("reward_unique_rooms", 0)
            episode_rooms = info.get("reward_unique_rooms_ep", 0)
            ep_room_id = info.get("reward_stable_room_id_ep", None) or ""
            ep_room_id_short = ep_room_id[:24] + "..." if ep_room_id and len(ep_room_id) > 24 else ep_room_id
            debounce_k = info.get("reward_room_debounce_counter_ep", 0)
            print(
                f"[ENV {eid}] room change stage={stage_curr} "
                f"novelty_hash={prev_short}->{new_short} "
                f"novelty_rooms={novelty_rooms} episode_rooms_ep={episode_rooms} "
                f"episode_room_id_ep={ep_room_id_short} debounce_counter_ep={debounce_k}"
            )
            # Optional debug frame dump on room change (under debug flags only)
            if getattr(self.cfg, "debug", False):
                dump_dir = Path(getattr(self.cfg, "debug_dump_dir", None) or self.cfg.workdir)
                frames_dir = dump_dir / "debug_frames" / "rooms"
                try:
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    fn = frames_dir / f"env{eid}_step{self._step_count:06d}_room_{new_short or 'none'}.png"
                    Image.fromarray(obs).save(fn)
                except Exception:
                    pass
        if getattr(self.cfg, "debug", False):
            if room_hash is not None:
                if room_hash != self._debug_prev_room_hash:
                    self._debug_room_changes += 1
                    self._debug_steps_since_room = 0
                    self._debug_prev_room_hash = room_hash
                else:
                    self._debug_steps_since_room += 1
            if info.get("reward_stuck_event"):
                self._debug_stuck_events += 1
            stage_prev = getattr(self, "_debug_prev_stage", None)
            stage_curr = info.get("stage", 0)
            if stage_prev is not None and stage_curr != stage_prev:
                self._debug_stage_changes += 1
            self._debug_prev_stage = stage_curr

            debug_every = max(1, getattr(self.cfg, "debug_every", 10))
            if self._step_count % debug_every == 0:
                comp = info.get("reward_components", {})
                top = sorted([(k, v) for k, v in comp.items() if v != 0], key=lambda x: abs(x[1]), reverse=True)[:3]
                rh_short = (room_hash or "")[:8] if room_hash else "—"
                act_name = _action_id_to_name(int(action))
                line = (
                    f"[debug] step={self._step_count} act={act_name}({int(action)}) "
                    f"stage={info.get('stage', 0)} conf={info.get('stage_conf', 0):.2f} "
                    f"rh={rh_short} room_changes={self._debug_room_changes} uq={info.get('reward_unique_rooms', 0)} "
                    f"fd={frame_diff:.3f} stuck_ev={self._debug_stuck_events} "
                    f"r={reward:.3f} top={top}"
                )
                if force:
                    hold_ms = "keydown_hold"
                    action_repeat_val = getattr(self.cfg, "action_repeat", 1)
                    line += f" | action_name={act_name} hold_ms={hold_ms} action_repeat={action_repeat_val}"
                print(line)
            dump_n = getattr(self.cfg, "debug_dump_frames", 0)
            if dump_n > 0 and self._debug_dumped_count < dump_n:
                dump_dir = Path(getattr(self.cfg, "debug_dump_dir", None) or self.cfg.workdir)
                frames_dir = dump_dir / "debug_frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                out = frames_dir / f"step_{self._step_count:05d}.png"
                Image.fromarray(obs).save(out)
                self._debug_dumped_count += 1

        self._prev_obs = obs.copy()

        if terminated or truncated:
            # Ensure termination_reason is single canonical value for logging
            reason = info.get("termination_reason", "unknown")
            if reason not in ("death", "stuck", "timeout", "reset_handshake_fail", "unknown"):
                reason = "death" if "death" in reason else ("stuck" if "stuck" in reason else ("timeout" if "timeout" in reason else "unknown"))
                info["termination_reason"] = reason
            if getattr(self.cfg, "debug", False):
                print_termination_block(
                    self, self._step_count, _action_id_to_name(int(action)), terminated, truncated, info, obs
                )
            # STAGE 00: в первых 50 шагах при уверенном stage=0 считаем смерть ложной (артефакт кадра/перехода)
            if (
                terminated_by_death
                and not getattr(self.cfg, "ignore_death", False)
                and self._step_count <= 50
            ):
                stage = info.get("stage", -1)
                stage_conf = info.get("stage_conf", 0.0)
                if stage == 0 and stage_conf >= 0.5:
                    if getattr(self.cfg, "debug", False):
                        print(
                            "[WARN] Impossible death in STAGE 00 (false-positive?): env_id=%s step=%s raw_death_signals=%s — overriding to not terminated"
                            % (getattr(self.cfg, "instance_id", "?"), self._step_count, info.get("raw_death_signals"))
                        )
                    terminated = False
                    terminated_by_death = False
                    self._death_low_life_steps = 0
                    info["termination_reason"] = "unknown"
                    info["death_false_positive_overridden"] = True

        total_step_time = time.perf_counter() - t_step_start
        self._perf_timings.append({
            "capture_time": getattr(self, "_last_capture_time", 0),
            "preprocessing_time": getattr(self, "_last_preprocess_time", 0),
            "env_step_time": total_step_time,
        })
        if self._perf_log_interval > 0 and self._step_count % self._perf_log_interval == 0 and self._perf_timings:
            N = len(self._perf_timings)
            avg_cap = sum(t["capture_time"] for t in self._perf_timings) / N
            avg_prep = sum(t["preprocessing_time"] for t in self._perf_timings) / N
            avg_step = sum(t["env_step_time"] for t in self._perf_timings) / N
            steps_per_min = 60.0 / avg_step if avg_step > 0 else 0
            print(
                f"[perf] step={self._step_count} capture={avg_cap*1000:.1f}ms prep={avg_prep*1000:.1f}ms "
                f"step_total={avg_step*1000:.1f}ms spm={steps_per_min:.1f}"
            )

        if (terminated or truncated) and self._reward_logger is not None and self._reward_logger.step_count > 0:
            if not getattr(self.cfg, "quiet", False):
                print(self._reward_logger.format_episode_report(self._reward_logger.last_extra))

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.close()
            except Exception:
                pass
            self._capture = None
        if self._emu is not None:
            if self._keys_down:
                try:
                    self._emu.keyup(list(self._keys_down))
                except Exception:
                    pass
                self._keys_down = set()
            self._emu.close()
            self._emu = None

