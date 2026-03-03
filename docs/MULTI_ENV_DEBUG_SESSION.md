# Multi-env отладка: саммари сессии и рекомендации

**Назначение:** полная документация изменений, найденных ошибок и текущего состояния при `num_envs > 1`. Использовать как входную точку для нового чата или рефакторинга.

Связанные документы: `docs/CAPTURE.md`, `docs/MODULES_AND_FLAGS.md`, `docs/TRAINING.md`.

---

## 1. Исходная проблема

- **Симптом:** при `num_envs=2` и `--debug-force-action RIGHT` эпизоды завершаются почти сразу: `ep_steps=2`, `ep_return=-1`, `deaths=1`, `unique_rooms=0`, `stage_mean=0`.
- **Гипотеза:** ложное срабатывание детектора смерти и/или коллизия ресурсов между двумя env (захват кадра, канал управления, окно).

---

## 2. Что сделали (хронологически)

### 2.1. Диагностика завершения эпизода (env.py)

- В **info** при любом `terminated`/`truncated` добавлены поля:
  - `termination_reason`, `raw_death_signals`, `hp_value`, `lives_value`, `gameover_flag`, `stage`, `stage_conf`
- В режиме **--debug** при завершении выводится блок **TERMINATION EVENT** (env_id, step, reason, hp, lives, stage, raw_signals).
- Добавлен флаг **--ignore-death**: смерть не ставит `terminated=True`, только логируется (для проверки ложных срабатываний).

### 2.2. Принудительное действие и диагностика шага (env.py, train_ppo.py)

- **--debug-force-action** уже был; в debug-строке при принудительном действии добавлен вывод: `action_name`, `hold_ms` (keydown_hold), `action_repeat`.
- **--debug-single-episode**: ровно один эпизод на каждый env, вывод reset/termination, затем выход (для быстрой отладки).

### 2.3. Диагностика reset и идентичность инстанса (env.py)

- При **--debug** на каждом reset выводится блок **ENV RESET** в формате:
  - `[ENV RESET] env_id=... pid=... channel=... capture=...`
  - `workdir=...`, `screenshot_path=...` (для file) или `window_rect=...` (для window)
  - `stage=... stage_conf=...`, `room_hash=...`, `frame_hash=...`
- Добавлен **реестр ресурсов** `_ENV_RESOURCE_REGISTRY`: при debug проверяется уникальность workdir, screenshot_path и (для window) window_rect; при коллизии — RuntimeError с понятным текстом.
- В начале запуска обучения вызывается **clear_env_resource_registry()**.

### 2.4. Безопасность multi-env для capture (train_ppo.py, make_env.py)

- При **num_envs > 1** и **capture_backend=window** без явных per-env rects:
  - автоматический **fallback на file (png)** и одноразовый warning.
- Добавлены опции: **--window-title-pattern**, **--window-rects-json** (формат: `{"0": {"title": "...", "crop": [x,y,w,h]}, "1": ...}`).
- **make_env(rank, base_cfg, per_env_window=...)** подставляет для каждого ранга свой `window_crop` и `window_title` из JSON.

### 2.5. Reset handshake (env.py, train_ppo.py)

- В **EnvConfig** добавлены: `reset_handshake_stable_frames`, `reset_handshake_conf_min`, `reset_handshake_timeout_s`.
- В **reset()** при `reset_handshake_stable_frames > 0`: ожидание нескольких подряд кадров с `stage_conf >= conf_min` (или таймаут), затем возврат obs.
- При **num_envs > 1** в train_ppo по умолчанию включается handshake (3 стабильных кадра, conf_min=0.5, timeout 15 с).
- В debug после handshake выводится строка **READY env_id=...**.

### 2.6. Задержка между запусками openMSX (train_ppo.py)

- Перед вторым и последующим **reset()** при num_envs > 1 добавлена пауза **3 с**, чтобы первый инстанс успел полностью подняться и занять порт.

### 2.7. Per-env статистика и TERMINATION (env.py, train_ppo.py)

- **termination_reason** нормализован к одному из: `death`, `stuck`, `timeout`, `reset_handshake_fail`, `unknown`.
- В info добавлено поле **death_detector** (например `life_bar_v1`).
- При **--debug** при любом terminated/truncated выводится блок **TERMINATION** с полями: env_id, step_in_episode, reason, death_detector, raw_death_signals, hp, lives, gameover, stage, room_hash, frame_hash, last_action, hold_ms, repeat.
- В **episode_stats** добавлен **env_id**; считаются **deaths_per_env** и **ep_steps_per_env**; при **--debug** в summary выводится строка `per_env: deaths_env0=... ep_steps_env0=... deaths_env1=... ep_steps_env1=...`.

### 2.8. Детектор смерти: гистерезис и STAGE 00 (env.py)

- **Гистерезис:** смерть объявляется только после **двух подряд** кадров с `life < 0.15` (и `life_prev > 0.3`). Один артефактный/чёрный кадр больше не даёт мгновенного terminated.
- В **reset()** сбрасывается счётчик **\_death_low_life_steps**.
- **STAGE 00:** при debug + debug_force_action + смерть в первых 50 шагах при stage=0 и stage_conf>=0.5 вместо RuntimeError выводится **предупреждение** и для этого шага **не** ставится terminated (эпизод продолжается); в info добавляется `death_false_positive_overridden=True`.

### 2.9. Диагностика канала управления (env.py)

- При **--debug** на первых трёх шагах выводится строка: `[debug] env_id=... sending action to workdir=... pid=... commands_tcl=...`.

### 2.10. Проверка процесса после загрузки (openmsx_bridge.py)

- В конце **\_wait_boot()** добавлена проверка: если процесс уже завершился (`proc.poll() is not None`), выбрасывается RuntimeError с workdir и путём к логу.

### 2.11. Захват скриншота (capture.py, openmsx_bridge.py)

- **FileCapturePNG.grab()**: после вызова `screenshot()` — до 5 попыток чтения; перед каждой попыткой пауза (0.05*(attempt+1)) с; чтение файла через **read_bytes()** и открытие через **io.BytesIO(data)** для согласованного снимка и избежания гонок/блокировок.
- При повторной ошибке в сообщении выводятся **path**, **size**, **magic_hex** (первые байты файла).
- В **screenshot()** в мосте после получения ответа добавлена пауза **50 ms** перед возвратом.
- Убран некорректный вызов `Image.open(..., format=...)` (в текущей версии Pillow нет такого аргумента).

---

## 3. Выявленные ошибки и причины

### 3.1. Коллизия при window capture (num_envs=2)

- **Симптом:** оба env видят один и тот же кадр (или один получает кадр другого).
- **Причина:** при `capture_backend=window` поиск окна по заголовку (`_get_window_rect_by_title("openMSX")`) возвращает **первое** найденное окно — оба env привязываются к одному окну.
- **Что сделано:** при num_envs>1 и window без **--window-rects-json** принудительный переход на file (png) и предупреждение.

### 3.2. Ложное срабатывание детектора смерти (один кадр)

- **Симптом:** `life` падает с ~0.9 до 0.0 за один шаг, эпизод завершается по «смерти».
- **Причина:** один артефактный кадр (чёрный экран, переход, недописанный PNG) даёт `life=0` в ROI полоски жизни.
- **Что сделано:** гистерезис (2 подряд кадра с life<0.15); в STAGE 00 при первых 50 шагах с debug_force_action — только предупреждение и отмена terminated.

### 3.3. Ошибка чтения скриншота (OSError: unrecognized data stream contents)

- **Симптом:** `Image.open(img_path).convert("RGB")` падает при чтении step_frame.png (размер файла мог быть ненулевой, например ~171 KB).
- **Возможные причины:** файл ещё дописывается/заблокирован; другой формат; повреждённая запись.
- **Что сделано:** чтение через `read_bytes()` + `BytesIO`, повторные попытки с паузой, пауза после screenshot в мосте, улучшенное сообщение об ошибке (path, size, magic_hex).

### 3.4. Старые процессы openMSX

- **Симптом:** кнопки перестают нажиматься во всех env; возможны глюки со скриншотами.
- **Причина:** оставались запущенные экземпляры openMSX от предыдущих запусков; они продолжали использовать порты/файлы/окна и мешали новым инстансам.
- **Решение:** пользователь закрыл старые openMSX; после этого кнопки снова заработали (по крайней мере в одном env).

### 3.5. Текущее состояние: env 0 vs env 1

- **Env 0 (первый):** кнопки нажимаются; герой идёт вправо и упирается в стену (ожидаемое поведение при постоянном RIGHT).
- **Env 1 (второй):** кнопки **не нажимаются**; игра, по сути, в демо-режиме (нет ввода).
- **Гипотеза:** для второго env либо команды (keydown/keyup) не доходят до своего процесса openMSX, либо второй процесс не читает свой `commands.tcl` (не тот workdir, не тот процесс, или порядок/тайминг опроса). Ресурсы изолированы по workdir (runs/tmp/0 и runs/tmp/1), PID у процессов разные (по логам), поэтому вероятны тайминг опроса, блокировка файла или особенность второго инстанса openMSX при двух окнах.

---

## 4. Текущее состояние кода (что оставить / что откатить)

- **Оставить:**  
  - гистерезис детектора смерти;  
  - чтение скриншота через read_bytes + BytesIO и повторные попытки;  
  - per-env диагностика (ENV RESET, TERMINATION, deaths_per_env, ep_steps_per_env) при --debug;  
  - при num_envs>1 и window — fallback на png без per-env rects;  
  - задержка 3 с между запусками инстансов;  
  - --debug-single-episode, --ignore-death;  
  - проверка процесса после _wait_boot;  
  - смягчение STAGE 00 (warning + override, без падения).

- **Рассмотреть откат/упрощение (по желанию):**  
  - избыточная детализация ENV RESET (можно сократить до одной строки при не-debug);  
  - реестр ресурсов и проверки коллизий — оставить только в debug или при num_envs>1;  
  - reset handshake по умолчанию при num_envs>1 — можно сделать опциональным флагом, если мешает старту.

- **Не откатывать:** логику смерти (гистерезис), чтение скриншота (BytesIO + retries), изоляцию workdir и fallback window→png.

---

## 4.1. Рефакторинг (выполнено)

- **Детектор смерти:** добавлен `death_warmup_steps` (50): в первых N шагах смерть не объявляется. STAGE 00 override безусловный (step≤50, stage=0, stage_conf≥0.5 → не прерывать эпизод).
- **Диагностика:** вынесена в `msx_env/env_diagnostics.py` (реестр, ENV RESET/TERMINATION при debug). `clear_env_resource_registry` импортируется из `msx_env.env_diagnostics` в train_ppo.
- **train_ppo:** флаг `--no-reset-handshake` — при num_envs>1 отключить handshake для проверки env 1.

---

## 5. Рекомендации для следующего шага / рефакторинга

1. **Проверить второй инстанс openMSX**
   - Убедиться, что у второго процесса свой workdir (runs/tmp/1) и он реально опрашивает `commands.tcl` из этого каталога (логи openMSX, bootstrap_status.txt, содержимое commands.tcl в момент step для env 1).
   - При необходимости увеличить задержку между запуском первого и второго процесса или добавить явную проверку «второй процесс запущен и ответил на ping» перед началом шагов.

2. **Канал управления**
   - Рассмотреть переход на сокет/pipe для управления (по документации openMSX), чтобы каждый инстанс имел явный канал, а не только свой файл commands.tcl (если при двух процессах обнаружатся гонки или блокировки файлов).

3. **Захват**
   - Для multi-env по умолчанию держать **png** (file) backend; window только с явным **--window-rects-json** и проверкой уникальности rect per env.
   - Документировать в CAPTURE.md: при нескольких env обязательно свой workdir и при window — свой rect/окно.

4. **Жизненный цикл процессов**
   - В скрипте или в env: при старте проверять, что нет «забытых» openMSX (по PID или по lock-файлу в runs/tmp), и выводить предупреждение или опцию «убить старые инстансы» перед запуском новых.

5. **Рефакторинг**
   - Вынести диагностику (ENV RESET, TERMINATION, per-env stats) в отдельный модуль или класс (например `EnvDiagnostics`) и включать только при debug.
   - Чётко разделить: «контроль openMSX» (один процесс на один workdir, один канал), «захват кадра» (один источник на env), «детектор смерти» (гистерезис, пороги, переопределение в STAGE 00).
   - Добавить минимальный интеграционный тест: два env, по 10 шагов с forced RIGHT, проверка что ep_steps и room_hash различаются и что нет смертей в STAGE 00.

---

## 6. Команды для воспроизведения

Один эпизод на env, принудительно RIGHT, png:

```bash
python train_ppo.py --num-envs 2 --debug --debug-single-episode --debug-force-action RIGHT --capture-backend png
```

С игнорированием смерти (проверка детектора):

```bash
python train_ppo.py --num-envs 2 --debug --debug-single-episode --debug-force-action RIGHT --ignore-death --capture-backend png
```

Dry-run 30 с, два env:

```bash
python train_ppo.py --num-envs 2 --dry-run-seconds 30 --debug --debug-force-action RIGHT --capture-backend png
```

---

## 7. Затронутые файлы (кратко)

| Файл | Изменения |
|------|-----------|
| `msx_env/env.py` | Реестр ресурсов, ENV RESET/TERMINATION вывод, гистерезис смерти, STAGE 00 override, debug вывод workdir/pid при step, сброс _death_low_life_steps в reset |
| `msx_env/capture.py` | FileCapturePNG.grab(): retries, read_bytes+BytesIO, удалён неверный format= |
| `msx_env/make_env.py` | per_env_window, передача window_crop/window_title по рангу |
| `openmsx_bridge.py` | Проверка proc после _wait_boot, пауза 50ms после screenshot |
| `train_ppo.py` | clear_env_resource_registry, fallback window→png при num_envs>1, задержка 3s между reset, per_env_window, deaths_per_env/ep_steps_per_env, --window-rects-json, --window-title-pattern, --debug-single-episode, reset handshake по умолчанию для num_envs>1 |

---

*Документ создан по итогам сессии отладки multi-env; при рефакторинге или новом чате использовать как саммари и чеклист.*
