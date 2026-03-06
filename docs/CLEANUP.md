# Что можно удалить (лишние и устаревшие файлы)

Краткий список артефактов и документов, которые можно безопасно удалить или не коммитить.

---

## Артефакты в корне проекта

Если запускали openMSX или env из корня репозитория, в корне могли появиться:

| Файл | Описание | Действие |
|------|----------|----------|
| `reply.txt` | Ответ openMSX (commands.tcl) | Удалить, добавить в `.gitignore` |
| `bootstrap_status.txt` | Статус bootstrap от openMSX | Удалить, добавить в `.gitignore` |

Они создаются в workdir эмулятора; в корне оказываются только при workdir = корень. Рекомендуется добавить в `.gitignore`: `reply.txt`, `bootstrap_status.txt`.

---

## Каталог diagnostics/

Каталог **`diagnostics/`** — результат разовых прогонов диагностики (например, отладка multi-env): подкаталоги с датой/временем, внутри `tmp/0`, `tmp/1`, `report.md`, `config.json`, `results.json` и т.д.

- **Удалить целиком:** `diagnostics/` — если история диагностики не нужна.
- Рекомендуется добавить в `.gitignore`: `diagnostics/`, чтобы не коммитить будущие прогоны.

---

## Документация: что оставить, что можно убрать

### Основная (оставить)

- **README.md** — входная точка, быстрый старт.
- **docs/PROJECT_OVERVIEW.md** — обзор архитектуры и модулей.
- **docs/CONTEXT.md** — цели, решения, где что лежит.
- **docs/SESSION.md** — контекст сессии, пути, как искать последний run.
- **docs/MODULES_AND_FLAGS.md** — модули, флаги, выходы (справочник).
- **docs/TRAINING.md** — PPO: обучение, метрики, guardrails.
- **docs/REWARD.md** — система наград.
- **docs/CAPTURE.md** — бэкенды захвата (png/single/window/dxcam).
- **docs/CONFIG_SYSTEM.md** — система конфигурации.
- **docs/VAMPIRE_KILLER_SPEC.md**, **docs/ITEMS_AND_REWARDS.md** — спецификация игры и предметов.
- **docs/PPO_MODEL.md** — архитектура PPO (encoder, LSTM).

### Краткое саммари для чата (по желанию)

- **docs/PROJECT_SUMMARY.md** — компактный блок для копирования в новый чат. Можно оставить для быстрого контекста или удалить, если не используете.

### Аудит и разовые отчёты (можно удалить для чистоты)

Эти файлы полезны как история решений и отладки; если нужна минимальная документация — их можно удалить:

| Файл | Содержание |
|------|------------|
| **docs/CAPTURE_REFACTOR_ANALYSIS.md** | Анализ рефакторинга захвата (до введения dxcam). |
| **docs/PPO_RECURRENT_AUDIT.md** | Аудит рекуррентного PPO. |
| **docs/PPO_RECURRENT_BENCHMARK.md** | Бенчмарк recurrent. |
| **docs/DIAGNOSTICS_PR.md** | Описание PR диагностики. |
| **docs/EPISODE_METRICS_FIX.md** | Фиксация исправления эпизодных метрик. |
| **docs/STAGE00_DEBUG_AUDIT.md** | Аудит отладки STAGE 00. |
| **docs/SUPERVISOR_AUDIT.md** | Аудит супервизора. |
| **docs/REWARD_V3_AUDIT.md** | Аудит reward v3. |
| **docs/PROJECT_AUDIT.md** | Общий аудит проекта. |
| **docs/MULTI_ENV_DEBUG_SESSION.md** | Сессия отладки multi-env. |
| **docs/REWARD_IMPROVEMENTS_CHECKLIST.md** | Чеклист улучшений наград. |

### Справочники конфига (оставить при использовании tools)

- **docs/CONFIG_GRAPH.md**, **docs/CONFIG_INVENTORY.md** — граф и инвентарь опций конфига; имеют смысл при использовании `tools/config_inventory.py`.

---

## Шаблоны и прочее

| Файл | Описание | Действие |
|------|----------|----------|
| **scripts/run_readme_template.md** | Шаблон README для каталога run (метрики, запуск). | Оставить как шаблон или удалить, если не вставляете в run dir. |
| **docs/help.txt** | Справка (например, для openMSX или скриптов). | Оставить, если на него есть ссылки; иначе можно удалить. |

---

## Итог: минимум для «ничего лишнего»

1. Добавить в **.gitignore**: `reply.txt`, `bootstrap_status.txt`, `diagnostics/`.
2. Удалить из корня (если есть): `reply.txt`, `bootstrap_status.txt`.
3. Удалить каталог **diagnostics/** (если не нужна история диагностики).
4. По желанию: удалить из **docs/** перечисленные выше аудиты и чеклисты (CAPTURE_REFACTOR_ANALYSIS, PPO_RECURRENT_*, DIAGNOSTICS_PR, EPISODE_METRICS_FIX, STAGE00_DEBUG_AUDIT, SUPERVISOR_AUDIT, REWARD_V3_AUDIT, PROJECT_AUDIT, MULTI_ENV_DEBUG_SESSION, REWARD_IMPROVEMENTS_CHECKLIST), оставив только основную документацию и справочники.

После этого в репозитории останутся актуальная документация и код без разовых артефактов и устаревших отчётов.
