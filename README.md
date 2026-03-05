# RL_msx — PPO for Vampire Killer (MSX2)

Train a PPO agent to play Vampire Killer via the openMSX emulator (file-based control + screen capture). See `docs/PROJECT_OVERVIEW.md` and `docs/SESSION.md` for full context.

## Configuration system: how to run and how to override

- **Single source of truth:** `project_config.load_config(argv)` parses CLI, loads optional `--config` file, merges (defaults → file → CLI), resolves all paths to absolute, validates, and writes `config_snapshot.json` and `resolved_paths.json` into the run directory.
- **How to run:** Use `--config <path>` to point to a JSON config (e.g. `run_dir/config_snapshot.json`). The supervisor uses this on restarts so the same config is applied. Without `--config`, use legacy CLI flags; paths and snapshots are still resolved and written.
- **Override order:** defaults → config file → CLI. If you pass `--reward-config` or `--config` and the file is missing, the process fails immediately (no silent default).
- **Paths:** All run artifacts live under the run directory: `train.log`, `metrics.csv`, `config_snapshot.json`, `resolved_paths.json`, `checkpoints/ppo/`. See **docs/CONFIG_SYSTEM.md** for details and **tools/config_inventory.py** for a config inventory.
