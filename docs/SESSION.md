# Саммари сессии: Vampire Killer MSX — RL (BC → PPO, multi-env)

**Назначение:** контекст для перехода в новый чат. Состояние проекта, недавние правки, где что лежит, как запускать. Для нового чата достаточно открыть этот файл и при необходимости смотреть **`docs/MODULES_AND_FLAGS.md`** и **`docs/CONTEXT.md`**.

**Копировать в другой чат:** готовый блок для вставки — **`docs/PROJECT_SUMMARY.md`** (краткое саммари проекта, пути, метрики, конфиги).

---

## 1. Что за проект

- **Цель:** обучить агента играть в Vampire Killer (Konami, MSX2) через эмулятор openMSX.
- **Цепочка:** запись демо человеком → **BC** (имитация) → **PPO** (RL по reward).
- **Управление:** file-based: Python пишет `commands.tcl`, openMSX опрашивает и выполняет, ответ в `reply.txt`. Клавиши — keymatrix/keydown/keyup.
- **Окружение:** `msx_env.env.VampireKillerEnv` — Gym-подобный API: `reset()`, `step(action)` → obs (84×84 grayscale), reward, terminated, truncated, info.

---

## 2. Ключевые пути

| Назначение | Путь |
|------------|------|
| **Саммари для чата** | **`docs/SESSION.md`** (этот файл) |
| Обзор проекта, run dir, метрики | `docs/PROJECT_OVERVIEW.md` |
| Контекст, архитектура | `docs/CONTEXT.md` |
| Модули, флаги, выходы | `docs/MODULES_AND_FLAGS.md` |
| **PPO: обучение, аудит, эксперименты** | **`docs/TRAINING.md`** |
| Награды (v1, компоненты) | `docs/REWARD.md` |
| Захват кадра (png/window) | `docs/CAPTURE.md` |
| Запись демо | `demos/record_demo.py` → `demos/runs/<run_id>/` |
| Обучение BC | `train_bc.py` → `checkpoints/bc/` |
| Обучение PPO | `train_ppo.py` → `checkpoints/ppo/` |
| Ночной запуск | `train_supervisor.py` + `configs/night_training.json` |
| Тест BC | `test_policy.py` |
| Тест PPO | `test_ppo.py` (в т.ч. `--diagnose-policy`) |
| Env + конфиг | `msx_env/env.py` (EnvConfig, VampireKillerEnv) |
| Фабрика env для num_envs | `msx_env/make_env.py` |
| Модели BC/PPO | `msx_env/bc_model.py`, `msx_env/ppo_model.py` |
| Мост openMSX | `openmsx_bridge.py` |

### Где искать последний ночной run и метрики

- При **`use_runs_dir: true`** в `configs/night_training.json` каждый запуск супервизора создаёт папку  
  **`runs/<YYYYMMDD>_<HHMMSS>_<git>_<run_name>/`** (например `runs/20260304_181828_58548f2_auto_night/`).
- **«Сегодняшний» прогон** — папка с самой свежей датой изменения (LastWriteTime); запись может идти до утра следующего дня.
- Внутри run dir: **`metrics.csv`** или **`metrics1.csv`**, **`train.log`**, **`config_snapshot.json`**, **`supervisor.log`** или **`supervisor1.log`**. При поиске данных проверять оба варианта имён.

---

## 3. Недавние доработки (эта сессия)

### 3.1. PPO multi-env (num_envs > 1)

- **Проблема:** при `--num-envs 2` оба openMSX запускались, но герои почти не двигались (1–2 шага и стоп).
- **Причина:** после отправки клавиш скриншот делался сразу; эмулятор не успевал отрисовать кадр → в буфер попадал старый кадр → политика уходила в NOOP.
- **Решение:** параметр **`post_action_delay_ms`** (в EnvConfig и `--post-action-delay-ms` в train_ppo). При `num_envs > 1` по умолчанию **50 мс** задержки после нажатия клавиш перед захватом кадра. Задаётся в `msx_env/env.py` в `step()` перед `_grab_frame_and_obs()`.

### 3.2. Одно окно openMSX постоянно перезапускалось

- **Проблема:** в одном окне всё играло, в другом окно то открывалось, то закрывалось (перезапуск самого процесса openMSX).
- **Причина:** при каждом **done** (смерть или конец эпизода) вызывался `reset()`, который делал `_emu.close()` и заново запускал openMSX. Тот env, у которого эпизоды заканчивались часто, постоянно убивал и поднимал процесс.
- **Решение:** **мягкий сброс (soft reset)**. Параметр **`soft_reset: bool = True`** в EnvConfig. При `reset()` и уже запущенном эмуляторе процесс **не закрывается**; вызывается `_soft_reset(emu)`: отпускаем клавиши, затем SPACE дважды с паузами (как «Continue» после Game Over), ждём загрузки уровня. Окно остаётся открытым. Полный перезапуск только при первом запуске (`_emu is None`) или при `soft_reset=False`.

### 3.3. Синтаксическая ошибка в train_ppo.py

- В f-строке использовался set comprehension `{e.cfg.workdir for e in envs}` — внутренние `{}` конфликтовали с плейсхолдерами. Исправлено: список workdir формируется отдельно, затем подставляется в строку.

### 3.4. PPO: аудит, эксперименты, guardrails (docs/TRAINING.md)

- **Аудит:** проанализирован цикл обучения; выявлено отсутствие логов value_loss, entropy, explained variance, разбивки наград и эпизодных статистик; сформулированы топ-3 узких места (слабый/шумный reward, коллапс энтропии, расходимость критика).
- **Эксперименты:** добавлены `--run-name`, `--log-dir`, `--config` (JSON), `--reward-config` (JSON), `--entropy-coef`, `--value-loss-coef`. Логирование: steps/sec, sample throughput, policy/value loss, entropy, approx_kl, explained_var, reward components, unique_rooms, deaths, stuck. Снимок конфига в `<log-dir>/<run_name>/config_snapshot.json`, метрики в `metrics.csv`.
- **Guardrails:** предупреждения при NaN в loss, низкой энтропии (коллапс политики), расходимости критика (value_loss > 5 или explained_var < −0.5), отсутствии роста unique_rooms за N обновлений.
- **Документация:** создан `docs/TRAINING.md` (архитектура, аудит, рекомендации по стабильности, система экспериментов, отладка). Конфиг наград: `RewardConfig.from_dict()` в `msx_env/reward/config.py`.

---

## 4. Как запускать

- **Демо:** `python demos/record_demo.py --run-id test01 --max-steps 2000 --preview`
- **BC:** `python train_bc.py --epochs 40` или с `--deep --move-weight 1.6 --rare-weight 1.5`
- **PPO (один env):** `python train_ppo.py --bc-checkpoint checkpoints/bc/best.pt --epochs 100`
- **PPO (два env, короткий прогон):**  
  `python train_ppo.py --num-envs 2 --epochs 2 --rollout-steps 32`  
  При `num_envs > 1` автоматически: `post_action_delay_ms=50`, `soft_reset=True`; workdir для каждого env: `runs/tmp/0`, `runs/tmp/1` (или `--tmp-root`).
- **PPO с экспериментом:**  
  `python train_ppo.py --run-name exp_01 --config configs/ppo_default.json --entropy-coef 0.01 --num-envs 4`  
  Логи и метрики: `<checkpoint-dir>/exp_01/metrics.csv`, `config_snapshot.json`. Подробнее: `docs/TRAINING.md`.
- **Тест PPO:** `python test_ppo.py --checkpoint checkpoints/ppo/last.pt --deterministic`
- **Проверка скорости перед ночью (2 мин):**  
  `python train_ppo.py --bc-checkpoint checkpoints/bc/best.pt --num-envs 2 --dry-run-seconds 120`  
  В конце выводится steps/sec, updates/hour и подсказка для max_total_steps за 8 ч. Подробнее: **docs/TRAINING.md** §7.

---

## 4.1. Night run recipe (один env)

Минимальный чек‑лист перед ночным запуском PPO (num_envs=1):

1. **Префлайт:**  
   ```bash
   python train_supervisor.py --preflight
   ```  
   Убедиться, что:
   - напечатаны `run_dir`, `ckpt_dir`, `metrics_path`;
   - `last.pt` существует и помечен как updated;
   - заголовок `metrics.csv` и последние строки выглядят разумно (нет NaN/inf).
2. **Smoke‑тест резюме (по желанию):**  
   ```bash
   python train_supervisor.py --resume-smoke-test
   ```  
   Проверить `preflight_report.md` в соответствующем `run_dir`: `max_update_second_run` должен быть > `max_update_first_run`.
3. **Ночной запуск (1 env):**  
   - убедиться, что в `configs/night_training.json` выставлены `num_envs: 1`, `run_name`, `checkpoint_dir`;
   - запустить:  
     ```bash
     python train_supervisor.py
     ```  
   - мониторить `supervisor.log` и `train.log` в `run_dir`.

> **Важно про `--deterministic` на ранних чекпоинтах.**  
> На очень ранних PPO‑чекпоинтах (малый `update`) детерминированный режим (`--deterministic`) может
> “залипнуть” на одном действии (часто NOOP), поэтому герой выглядит как стоящий на месте, в то время как
> стохастический режим даёт хоть какое‑то движение. Не интерпретируйте это как баг env/политики —
> это ожидаемое поведение для почти не обученной модели. Для диагностики коллапса можно использовать
> `test_ppo.py --diagnose-policy`.

---

## 5. Важные детали по коду

- **Rollout при num_envs > 1:** в цикле по `rollout_steps` внутренний цикл по `num_envs`: для каждого env свой `frame_buffers[i]`, свой `envs[i].step(action)`. Данные складываются в общие списки `roll_obs`, `roll_acts`, … в порядке env0, env1, env0, env1, … При подсчёте GAE траектории разбиваются по env (индексы `i, i+num_envs, i+2*num_envs, ...`).
- **EnvConfig.soft_reset:** по умолчанию True; при False поведение reset() как раньше (close + новый процесс + _skip_intro).
- **EnvConfig.post_action_delay_ms:** 0 по умолчанию; при num_envs > 1 в train_ppo выставляется 50, если не передан `--post-action-delay-ms`.
- **Тихий режим:** по умолчанию не печатаются сообщения `[perf]` (capture/prep/step); `EnvConfig.perf_log_interval = 0`. OpenMSX stdout/stderr пишутся в workdir (`openmsx_stdout.log`, `openmsx_stderr.log`), не в терминал. См. docs/CAPTURE.md.

---

## 6. Документация

- **CONTEXT.md** — цели, архитектура, ключевые решения (frame stacking, полоска жизни, лаги).
- **MODULES_AND_FLAGS.md** — все точки входа, флаги, каталоги вывода, таблица EnvConfig (включая `post_action_delay_ms`, `soft_reset`, `num_envs`, `tmp_root`).
- **TRAINING.md** — PPO: аудит, стабильность, эксперименты (--run-name, --config, --reward-config), guardrails, отладка.
- **REWARD.md** — система наград (v1, компоненты, отладка).
- **CAPTURE.md** — бэкенды захвата (png, single, window), калибровка окна.

---

*Саммари обновлено для перехода в новый чат.*
