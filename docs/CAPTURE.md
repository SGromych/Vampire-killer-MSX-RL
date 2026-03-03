# Захват кадра (Capture backends)

## Обзор

Env поддерживает три бэкенда захвата кадра для наблюдения (obs):

| Бэкенд   | Описание | Когда использовать |
|----------|----------|--------------------|
| **png**  | openMSX пишет PNG в файл (step_frame.png), Python читает с диска | По умолчанию, надёжно |
| **single** | То же, один и тот же файл перезаписывается | Меньше операций ФС |
| **window** | Захват области экрана (окно openMSX) через dxcam/mss, без файлов | Максимальная скорость, только Windows |

Формат после препроцессинга одинаковый: **(84, 84) uint8 grayscale**. Все бэкенды возвращают RGB (H,W,3) до resize/grayscale.

---

## Тихий режим (вывод в консоль и OpenMSX)

По умолчанию **сообщения о захвате кадра и производительности не выводятся**:

- **Терминал (Python):** в env каждые N шагов можно печатать строку `[perf] step=... capture=... prep=...` — по умолчанию **отключено** (`EnvConfig.perf_log_interval = 0`). Включить: передать `perf_log_interval=100` (или другой период) в конфиг окружения.
- **OpenMSX:** stdout и stderr процесса openMSX перенаправлены в файлы в workdir: **`openmsx_stdout.log`** и **`openmsx_stderr.log`**. В терминал они не попадают. Если в сборке openMSX команда `screenshot` пишет что-то в консоль, эти сообщения окажутся только в этих логах. Уменьшить уровень детализации openMSX (если поддерживается вашей сборкой) можно опцией при запуске эмулятора — см. документацию openMSX.

Итог: при обычном запуске train_ppo/test_ppo/benchmark_env в терминале не идёт постоянный поток сообщений о скриншотах или времени захвата; при необходимости диагностики можно включить `perf_log_interval` или просмотреть логи в workdir.

---

## Window backend

### Требования

- **Windows** (dxcam — только Windows; mss кросс-платформен, но калибровка по окну через win32).
- Установка: `pip install dxcam` (предпочтительно) или `pip install mss`.
- Окно openMSX должно быть видимым (не свёрнуто).

### Режимы выбора области

1. **Ручной crop (рекомендуется)**  
   Задать прямоугольник в координатах экрана: `(x, y, w, h)`.
   - Через конфиг: `EnvConfig(window_crop=(x, y, w, h), capture_backend="window")`.
   - Через CLI: `--window-crop x y w h` (если скрипт поддерживает).

2. **По заголовку окна**  
   Поиск окна по подстроке в заголовке (например `"openMSX"`), затем использование его rect.
   - `EnvConfig(window_title="openMSX", capture_backend="window")`.
   - Если окно не найдено, backend автоматически переключается на **file (png)** и в консоль выводится предупреждение.

### Калибровка

Скрипт **`scripts/calibrate_window_capture.py`**:

- Выводит список окон с подстрокой в заголовке (`--title openMSX`).
- Показывает текущий rect и замер FPS захвата.
- Опционально сохраняет несколько кадров на диск (`--dump-frames N`, `--out-dir DIR`).
- Выводит JSON-фрагмент конфига: `window_crop`, `window_title`, `capture_backend`, `capture_lag_ms`.

Пример:

```bash
python scripts/calibrate_window_capture.py --title openMSX --dump-frames 3 --out-dir calibrate_out
```

Полученный `window_crop` подставьте в `EnvConfig` или в CLI (если есть `--window-crop`).

### Задержка перед захватом (capture_lag_ms)

Для уменьшения tearing можно добавить небольшую задержку перед grab:

- `EnvConfig(capture_lag_ms=5)` или `--capture-lag-ms 5`.
- По умолчанию 0.

### Fallback

Если окно не найдено или dxcam/mss недоступны, при `fallback_to_file=True` (по умолчанию) используется file backend (png), в консоль пишется предупреждение.

---

## Бенчмарк

Сравнение бэкендов по времени шага и захвата:

```bash
python benchmark_env.py --backend png --steps 500
python benchmark_env.py --backend single --steps 500
python benchmark_env.py --backend window --steps 500
```

При `--backend window` можно задать область вручную:

```bash
python benchmark_env.py --backend window --window-crop 100 50 512 384 --steps 500
```

В выводе: steps/sec, steps/min, а также **avg** и **p95** (мс) для `capture_time`, `preprocessing_time`, `env_step_time`.

---

## Ограничения window backend при нескольких env (Phase 3)

- **Файловый backend (png/single):** несколько инстансов env безопасны: у каждого свой `workdir`, свой процесс openMSX, свои файлы.
- **Window backend:** параллельный запуск нескольких env с захватом из окна сложнее: у каждого процесса openMSX своё окно, и у каждого инстанса должна быть своя область захвата (свой `window_crop`). Режим с явным заданием crop по инстансу считается продвинутым и должен быть описан отдельно (per-instance crop). Сначала рекомендуется многопоточность/много-инстанс с **file** backend.

См. также **`docs/MODULES_AND_FLAGS.md`** (флаги `--capture-backend`, `--window-crop`, `--num-envs`).
