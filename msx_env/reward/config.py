"""
Конфигурация политики наград (версионирование, веса, пороги).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Конфиг политики наград v1. Все веса и пороги в одном месте."""

    version: str = "v1"

    # Step
    step_penalty: float = -0.001

    # Death (terminated in env)
    death_penalty: float = -1.0

    # Pickup (HUD) — нормализованные, с возможным cap
    pickup_weapon: float = 0.2
    pickup_key_chest: float = 0.2
    pickup_key_door: float = 0.2
    pickup_item: float = 0.1
    pickup_cooldown_steps: int = 30  # не давать повторную награду за тот же слот раньше

    # Room novelty
    novelty_reward: float = 0.35
    novelty_persistence_frames: int = 3  # K подряд одинаковых хэшей = новая комната
    novelty_saturation_cap: int = 0  # 0 = без ограничения; >0 = после N комнат decay
    novelty_saturation_decay: float = 0.95  # множитель после cap

    # Anti ping-pong
    pingpong_window: int = 50
    pingpong_min_alternations: int = 4  # A-B-A-B минимум для штрафа
    pingpong_penalty: float = -0.05

    # Stuck
    stuck_no_room_change_steps: int = 200
    stuck_frame_diff_threshold: float = 0.02  # mean abs diff нормализованный
    stuck_penalty: float = -0.3  # базовый штраф; при progressive умножается по severity
    stuck_truncate: bool = True  # при stuck можно truncated
    stuck_penalty_severity_levels: list[float] | None = None  # [-0.3, -0.5, -0.8]; None = formula

    # Progress proxy (disabled by default)
    progress_reward_per_pixel: float = 0.0  # >0 чтобы включить, если будет детектор
    progress_backtrack_penalty: float = 0.0

    # Backtrack penalty (optional, no cheating): штраф за A->B->A переходы
    backtrack_penalty_enabled: bool = False  # по умолчанию выкл
    backtrack_penalty_value: float = -0.01  # награда за каждый backtrack при включении

    # room_hash: исключить HUD (верх кадра) для стабильного хэша геймплея
    room_hash_crop_top: int = 14  # для 84x84: пропустить верхние 14 строк (HUD)
    room_hash_crop_bottom: int = 0

    # v3: novelty/detectors/stuck
    novelty_stability_frames: int = 0  # 0 = использовать novelty_persistence_frames
    pingpong_threshold: int = 0  # 0 = использовать pingpong_min_alternations
    key_reward: float = 0.0  # >0: награда за детекцию ключа (раз за эпизод)
    door_reward: float = 0.0  # >0: награда за детекцию двери (раз за эпизод)
    stuck_progressive: bool = True  # нарастающий штраф при залипании
    stuck_position_variance_steps: int = 60  # окно для дисперсии позиции (proxy)
    stuck_position_variance_threshold: float = 0.001  # порог низкой дисперсии X (норм. 0..1)

    # Stage (STAGE 00, 01, ... из HUD): ускорение прохождения, бонус за переход
    enable_stage_reward: bool = False
    stage_step_penalty: float = -0.002  # штраф за шаг только в stage_only_for (напр. STAGE 00)
    stage_advance_bonus: float = 2.0  # бонус за переход на следующий stage (один раз за переход)
    stage_only_for: list[int] | None = None  # [0] = только STAGE 00; None = все этапы
    stage_conf_threshold: float = 0.6  # ниже — не применять stage reward (anti-exploit)
    stage_stability_frames: int = 3  # гистерезис: stage должен быть стабилен K кадров

    # fix-room-metrics-stability: эпизодные метрики комнат (stable_room_id, playfield hash)
    episode_room_debounce_k: int = 7  # K кадров для debounce room_hash (6–8)
    episode_stage_stable_frames: int = 5  # N кадров для debounce stage
    episode_playfield_crop_top: int = 20  # playfield: без HUD сверху
    episode_playfield_crop_bottom: int = 4
    episode_playfield_crop_right: int = 36  # playfield: без weapon/key/items справа
    episode_unique_rooms_sanity_warn: int = 20  # WARNING если unique_rooms_ep > N

    # Door approach shaping (disabled by default; requires reliable door distance detector)
    enable_door_distance_reward: bool = False
    door_distance_reward: float = 0.01
    door_distance_min_delta: float = 0.0
    door_distance_clip: float = 0.05
    door_distance_requires_key: bool = True

    # Block break reward (requires reliable breakable-block detector; disabled by default)
    enable_block_break_reward: bool = False
    block_break_reward: float = 0.1
    block_break_debounce_frames: int = 8

    # Небольшая награда за использование ATTACK (с cooldown), чтобы стимулировать ломать блоки/свечи
    attack_use_reward: float = 0.0  # >0: бонус за шаг с action=ATTACK (раз в cooldown шагов)
    attack_use_cooldown_steps: int = 15  # минимум шагов между наградами за ATTACK

    # Exploration reduction after key found (multiplier for novelty reward)
    novelty_after_key_multiplier: float = 0.25

    # Position novelty: reward for visiting new (quantized) grid cells per episode
    enable_position_novelty: bool = False
    position_novelty_reward: float = 0.005
    position_novelty_quantize: int = 8  # cell size in pixels (obs coords 0..W-1, 0..H-1)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        """Загрузить конфиг из словаря (например из JSON). Неизвестные ключи игнорируются. Обратная совместимость с v1."""
        known = {
            "version", "step_penalty", "death_penalty",
            "pickup_weapon", "pickup_key_chest", "pickup_key_door", "pickup_item", "pickup_cooldown_steps",
            "novelty_reward", "novelty_persistence_frames", "novelty_saturation_cap", "novelty_saturation_decay",
            "pingpong_window", "pingpong_min_alternations", "pingpong_penalty", "pingpong_threshold",
            "stuck_no_room_change_steps", "stuck_frame_diff_threshold", "stuck_penalty", "stuck_truncate",
            "stuck_penalty_severity_levels",
            "room_hash_crop_top", "room_hash_crop_bottom",
            "progress_reward_per_pixel", "progress_backtrack_penalty",
            "backtrack_penalty_enabled", "backtrack_penalty_value",
            "novelty_stability_frames", "key_reward", "door_reward",
            "stuck_progressive", "stuck_position_variance_steps", "stuck_position_variance_threshold",
            "enable_stage_reward", "stage_step_penalty", "stage_advance_bonus", "stage_only_for",
            "stage_conf_threshold", "stage_stability_frames",
            "episode_room_debounce_k", "episode_stage_stable_frames",
            "episode_playfield_crop_top", "episode_playfield_crop_bottom", "episode_playfield_crop_right",
            "episode_unique_rooms_sanity_warn",
            "enable_door_distance_reward", "door_distance_reward",
            "door_distance_min_delta", "door_distance_clip", "door_distance_requires_key",
            "enable_block_break_reward", "block_break_reward", "block_break_debounce_frames",
            "attack_use_reward", "attack_use_cooldown_steps",
            "novelty_after_key_multiplier",
            "enable_position_novelty", "position_novelty_reward", "position_novelty_quantize",
        }
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "step_penalty": self.step_penalty,
            "death_penalty": self.death_penalty,
            "pickup_weapon": self.pickup_weapon,
            "pickup_key_chest": self.pickup_key_chest,
            "pickup_key_door": self.pickup_key_door,
            "pickup_item": self.pickup_item,
            "pickup_cooldown_steps": self.pickup_cooldown_steps,
            "novelty_reward": self.novelty_reward,
            "novelty_persistence_frames": self.novelty_persistence_frames,
            "novelty_saturation_cap": self.novelty_saturation_cap,
            "novelty_saturation_decay": self.novelty_saturation_decay,
            "pingpong_window": self.pingpong_window,
            "pingpong_min_alternations": self.pingpong_min_alternations,
            "pingpong_penalty": self.pingpong_penalty,
            "stuck_no_room_change_steps": self.stuck_no_room_change_steps,
            "stuck_frame_diff_threshold": self.stuck_frame_diff_threshold,
            "stuck_penalty": self.stuck_penalty,
            "stuck_truncate": self.stuck_truncate,
            "stuck_penalty_severity_levels": self.stuck_penalty_severity_levels,
            "room_hash_crop_top": self.room_hash_crop_top,
            "room_hash_crop_bottom": self.room_hash_crop_bottom,
            "progress_reward_per_pixel": self.progress_reward_per_pixel,
            "progress_backtrack_penalty": self.progress_backtrack_penalty,
            "backtrack_penalty_enabled": self.backtrack_penalty_enabled,
            "backtrack_penalty_value": self.backtrack_penalty_value,
            "novelty_stability_frames": self.novelty_stability_frames,
            "pingpong_threshold": self.pingpong_threshold,
            "key_reward": self.key_reward,
            "door_reward": self.door_reward,
            "stuck_progressive": self.stuck_progressive,
            "stuck_position_variance_steps": self.stuck_position_variance_steps,
            "stuck_position_variance_threshold": self.stuck_position_variance_threshold,
            "enable_stage_reward": self.enable_stage_reward,
            "stage_step_penalty": self.stage_step_penalty,
            "stage_advance_bonus": self.stage_advance_bonus,
            "stage_only_for": self.stage_only_for,
            "stage_conf_threshold": self.stage_conf_threshold,
            "stage_stability_frames": self.stage_stability_frames,
            "episode_room_debounce_k": self.episode_room_debounce_k,
            "episode_stage_stable_frames": self.episode_stage_stable_frames,
            "episode_playfield_crop_top": self.episode_playfield_crop_top,
            "episode_playfield_crop_bottom": self.episode_playfield_crop_bottom,
            "episode_playfield_crop_right": self.episode_playfield_crop_right,
            "episode_unique_rooms_sanity_warn": self.episode_unique_rooms_sanity_warn,
            "enable_door_distance_reward": self.enable_door_distance_reward,
            "door_distance_reward": self.door_distance_reward,
            "door_distance_min_delta": self.door_distance_min_delta,
            "door_distance_clip": self.door_distance_clip,
            "door_distance_requires_key": self.door_distance_requires_key,
            "enable_block_break_reward": self.enable_block_break_reward,
            "block_break_reward": self.block_break_reward,
            "block_break_debounce_frames": self.block_break_debounce_frames,
            "attack_use_reward": self.attack_use_reward,
            "attack_use_cooldown_steps": self.attack_use_cooldown_steps,
            "novelty_after_key_multiplier": self.novelty_after_key_multiplier,
            "enable_position_novelty": self.enable_position_novelty,
            "position_novelty_reward": self.position_novelty_reward,
            "position_novelty_quantize": self.position_novelty_quantize,
        }


def default_v1_config() -> RewardConfig:
    """Дефолтный конфиг v1 для обратной совместимости по масштабам."""
    return RewardConfig()


def default_v3_config() -> RewardConfig:
    """Конфиг v3: улучшенная novelty, key/door детекторы, прогрессивный stuck, диагностика."""
    return RewardConfig(
        version="v3",
        novelty_stability_frames=3,
        pingpong_threshold=4,
        key_reward=0.3,
        door_reward=0.5,
        stuck_progressive=True,
        stuck_position_variance_steps=60,
        stuck_position_variance_threshold=0.001,
        stuck_penalty_severity_levels=[-0.3, -0.5, -0.8],
        room_hash_crop_top=14,
    )
