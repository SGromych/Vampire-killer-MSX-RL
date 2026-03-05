# Система наград Vampire Killer RL

Список предметов игры и соответствие слотам HUD: **`docs/ITEMS_AND_REWARDS.md`**. Спецификация игры (уровни, смерть, HUD): **`docs/VAMPIRE_KILLER_SPEC.md`**.

## Обзор

Награда оптимизирует поведение агента для прохождения уровня: исследование комнат, подбор предметов (оружие, ключи), избегание смерти и залипания. Чтобы давать **ранний плотный сигнал** (даже до выхода с уровня 1), введены компоненты исследования (novelty по комнатам) и анти-эксплойты (stuck, ping-pong).

## Версионирование политики

- **v1** — базовая политика: step penalty, death, pickup (нормализованные), novelty, ping-pong, stuck.
- **v3** — улучшенная novelty (окно 50 хэшей, novelty_rate, pingpong_count), KeyDetector/DoorDetector (награда раз за эпизод), прогрессивный stuck (дисперсия X-позиции, stuck_severity), диагностика доминирования (death > 50% — предупреждение). Конфиг: `default_v3_config()` или JSON с полями `novelty_stability_frames`, `key_reward`, `door_reward`, `stuck_progressive` и др. Обратная совместимость с v1.
- Конфиг задаётся через `RewardConfig` в `msx_env.reward.config`; версия хранится в `config.version`.

Изменить политику: создать свой `RewardConfig` (или подкласс), подставить нужные веса/пороги и передать в `EnvConfig.reward_config`. При `reward_config=None` используется **legacy**-режим (только HUD pickup + death в env; step_penalty добавляется в `train_ppo.py`).

### Как подключить reward v3

- **Вручную (train_ppo):**  
  `python train_ppo.py --reward-config configs/reward_v3.json ...`  
  Готовый пресет v3: **`configs/reward_v3.json`** (key_reward=0.3, door_reward=0.5, novelty_stability_frames=3, прогрессивный stuck и др.).
- **Ночной супервизор:** в `configs/night_training.json` добавь поле `"reward_config": "configs/reward_v3.json"` — супервизор передаст `--reward-config` при первом запуске (при рестартах используется снимок из run dir).

---

## Компоненты

### 1. Step penalty

- **Назначение:** не затягивать эпизод без прогресса.
- **Логика:** каждый шаг даёт `step_penalty` (отрицательное).
- **Формула:** `r_step = step_penalty` (например −0.001).
- **Пороги:** `RewardConfig.step_penalty`.
- **Риски:** при слишком большом по модулю значении агент стремится закончить эпизод быстрее (в т.ч. смертью); при малом — игнорирует.

### 2. Death penalty

- **Назначение:** штраф за смерть (падение HP по полоске жизни).
- **Логика:** при `terminated_on_death` и падении жизни (life < 0.15, ранее > 0.3) эпизод завершается и даётся штраф.
- **Формула:** `r_death = death_penalty` (например −1.0) в момент терминации.
- **Пороги:** `RewardConfig.death_penalty`; пороги жизни — в `life_bar.get_life_estimate` и в env.
- **Риски:** слишком большой по модулю штраф доминирует над остальными наградами.

### 3. Pickup (HUD)

- **Назначение:** награда за подбор оружия, ключей, предметов по HUD.
- **Логика:** переход пусто → заполнено по слотам (weapon, key_chest, key_door, items). Внутренний cooldown шагов, чтобы не фармить один и тот же слот из-за мерцания.
- **Формула:** за каждый тип подбора разовая награда (weapon/key_chest/key_door/item), с учётом cooldown.
- **Пороги:** `pickup_weapon`, `pickup_key_chest`, `pickup_key_door`, `pickup_item`, `pickup_cooldown_steps`.
- **Риски:** без cooldown — эксплуатация мерцания HUD; слишком большие веса — доминирование над исследованием.

### 3.1. Stage (STAGE 00, 01, … из HUD)

- **Назначение:** ускорить прохождение (меньше залипания в STAGE 00) и поощрить переход на следующий этап.
- **Детектор:** номер этапа читается из HUD (два знака «00», «01», …) в `msx_env.hud_parser.parse_stage`; в `info` попадают `stage` (int) и `stage_conf` (0..1). При низкой уверенности награды за stage не применяются.
- **Штраф за шаг:** пока этап не меняется, каждый шаг в выбранных этапах даёт `stage_step_penalty` (например −0.002). По умолчанию только для STAGE 00 (`stage_only_for: [0]` или `null` = только 00).
- **Бонус за переход:** при увеличении stage (например 0→1) один раз даётся `stage_advance_bonus` (например +2.0).
- **Гистерезис:** смена stage принимается только после `stage_stability_frames` подряд одинаковых значений (анти-дрожание).
- **Конфиг:** `enable_stage_reward`, `stage_step_penalty`, `stage_advance_bonus`, `stage_only_for`, `stage_conf_threshold`, `stage_stability_frames`. Компоненты в разбивке: `reward_stage_step`, `reward_stage_advance`.

---

### 4. Room novelty (награда за «ходить туда, где ещё не был»)

- **Назначение:** поощрение за исследование новых комнат/сцен — для лабиринта критично, иначе агент залипает на одном экране.
- **Логика:** по obs строится стабильный хэш (block-mean по downscale); гистерезис: «новая комната» засчитывается, только если один и тот же хэш держится K кадров подряд. Награда `novelty_reward` (+0.2 по умолчанию) за **первый визит** комнаты в эпизоде.
- **Псевдокод:**
  - `h = block_mean_hash(obs)`
  - буфер последних K хэшей; если все K совпадают → `stable_hash`
  - если `stable_hash` ещё не в `seen_rooms` → `r_novelty = novelty_reward`, добавить в `seen_rooms`
- **Пороги:** `novelty_reward`, `novelty_persistence_frames` (K), `novelty_saturation_cap`, `novelty_saturation_decay`.
- **Риски:** при малом K — ложные срабатывания из-за анимации; при большом K — запаздывание. Слишком большая награда — агент бегает по комнатам без прогресса.
- **Тюнинг лабиринта:** если агент залипает — увеличить `novelty_reward` (например 0.3–0.4) или уменьшить `novelty_persistence_frames` (K) при шуме. Настройка через `--reward-config` JSON.

### 5. Anti ping-pong

- **Назначение:** снизить бессмысленное хождение туда-обратно между двумя комнатами.
- **Логика:** в окне последних W шагов (по стабильным хэшам комнат) считается число чередований A↔B; при превышении порога — штраф.
- **Пороги:** `pingpong_window`, `pingpong_min_alternations`, `pingpong_penalty`.
- **Риски:** при слишком агрессивном штрафе — подавление легитимного возврата (например за ключом).

### 6. Stuck

- **Назначение:** детекция залипания (нет смены комнаты и почти нет изменения кадра).
- **Логика:** если дольше N шагов нет смены стабильного хэша комнаты и `frame_diff(prev_obs, obs) < threshold` → штраф и опционально `truncated=True`.
- **Пороги:** `stuck_no_room_change_steps`, `stuck_frame_diff_threshold`, `stuck_penalty`, `stuck_truncate`.
- **Риски:** в статичных сценах (меню, диалог) возможны ложные срабатывания; порог diff и N нужно подбирать.

### 7. Progress proxy (опционально)

- **Назначение:** малая награда за движение вперёд (например вправо), штраф за сильный откат.
- **Состояние:** по умолчанию выключено (`progress_reward_per_pixel = 0`); требует детекции позиции (спрайт/шаблон), пока не реализовано надёжно.

---

## Масштабирование

- **Death = −1.0, step = −0.001:** за 1000 шагов без событий накопленный step penalty ≈ −1.0 (как одна смерть). Это выравнивает «ничего не делать» и «умереть».
- **Pickup:** в v1 нормализованы до ~0.1–0.2 за событие, чтобы не доминировали над novelty и не перевешивали смерть.
- **Ключ/дверь в будущем:** рекомендуется держать ключ/дверь в том же порядке величины, что и текущие pickup (0.2), или чуть выше (0.3–0.5), с учётом редкости события.

---

## Отладка

| Симптом | Что проверить |
|--------|----------------|
| Агент крутится между двумя комнатами | Увеличить `pingpong_penalty` или уменьшить `pingpong_min_alternations`; смотреть `reward_components["pingpong"]` и `reward_pingpong_event` в info. |
| Фармит подборы (мерцание HUD) | Увеличить `pickup_cooldown_steps`; уменьшить веса pickup. |
| Не исследует | Увеличить `novelty_reward`; проверить, что хэш стабилен (увеличить `novelty_persistence_frames` при шуме). |
| Ложные stuck | Увеличить `stuck_no_room_change_steps` или `stuck_frame_diff_threshold`; временно отключить `stuck_truncate`. |
| Слишком быстрая смерть | Уменьшить по модулю `death_penalty` или усилить награды за прогресс/novelty. |

Зависимости от препроцессинга:
- Хэш комнаты считается по obs (84×84 grayscale); смена crop/resize изменит хэши.
- HUD pickup зависит от полного кадра (rgb) и ROI в `hud_parser`; смена разрешения/кропа влияет на парсинг.

---

## Воспроизведение

- **Legacy (без политики):** `EnvConfig(reward_config=None)`; в `train_ppo.py` к reward добавляется `args.step_penalty`.
- **v1:** `EnvConfig(reward_config=default_v1_config())`; step penalty уже входит в reward из env; в `train_ppo` при наличии `info["reward_components"]` step_penalty не добавляется повторно.

Пример конфига v1 по умолчанию (таблица):

| Параметр | Значение по умолчанию |
|----------|------------------------|
| version | "v1" |
| step_penalty | -0.001 |
| death_penalty | -1.0 |
| pickup_weapon | 0.2 |
| pickup_key_chest | 0.2 |
| pickup_key_door | 0.2 |
| pickup_item | 0.1 |
| pickup_cooldown_steps | 30 |
| novelty_reward | 0.35 |
| novelty_persistence_frames | 3 |
| novelty_saturation_cap | 0 |
| novelty_saturation_decay | 0.95 |
| pingpong_window | 50 |
| pingpong_min_alternations | 4 |
| pingpong_penalty | -0.05 |
| stuck_no_room_change_steps | 200 |
| stuck_frame_diff_threshold | 0.02 |
| stuck_penalty | -0.2 |
| stuck_truncate | True |
| progress_reward_per_pixel | 0.0 |
| progress_backtrack_penalty | 0.0 |
| novelty_stability_frames (v3) | 0 (= persistence) |
| pingpong_threshold (v3) | 0 (= min_alternations) |
| key_reward (v3) | 0.0 |
| door_reward (v3) | 0.0 |
| stuck_progressive (v3) | True |
| stuck_position_variance_steps (v3) | 60 |
| stuck_position_variance_threshold (v3) | 0.001 |

---

## Метрики в info и для TensorBoard

В `info` при использовании политики:
- `info["reward_components"]` — вклад по компонентам за текущий шаг.
- `info["episode_reward_components"]` — накопленные за эпизод суммы по компонентам.
- `info["reward_unique_rooms"]`, `info["reward_pingpong_event"]`, `info["reward_stuck_event"]`, `info["reward_stuck_truncate"]` — вспомогательные флаги/значения.

Рекомендуемые метрики для TensorBoard (логировать из `episode_reward_components` и флагов по окончании эпизода):
- reward/total, reward/novelty, reward/pickup, reward/penalties_step, reward/penalties_death, reward/penalties_stuck, reward/penalties_pingpong, reward/key, reward/door (v3);
- env/unique_rooms, env/novelty_rate, env/pingpong_count, env/stuck_events, env/stuck_severity (v3);
- env/key_detected, env/door_detected, env/detection_confidence (v3);
- reward_dominance_ratio, reward_dominant_component, death_dominance_warning (v3).
