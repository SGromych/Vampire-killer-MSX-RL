# Configuration graph

Which module consumes which configuration fields.

| Module | Consumes (sources) |
|--------|--------------------|
| benchmark_env.py | CLI flags |
| debug_env.py | CLI flags |
| demos\record_demo.py | CLI flags |
| demos\replay_demo.py | CLI flags |
| demos\validate_demo.py | CLI flags |
| msx_env\dataset.py | dataclass fields |
| msx_env\env.py | dataclass fields |
| msx_env\hud_parser.py | dataclass fields |
| msx_env\human_controller.py | dataclass fields |
| msx_env\perf_timers.py | dataclass fields |
| msx_env\reward\components.py | dataclass fields |
| msx_env\reward\config.py | dataclass fields |
| msx_env\reward\diagnostics.py | dataclass fields |
| msx_env\reward\episode_metrics.py | dataclass fields |
| msx_env\reward\logger.py | dataclass fields |
| msx_env\reward\policy.py | dataclass fields |
| project_config.py | CLI flags, dataclass fields |
| scripts\calibrate_window_capture.py | CLI flags |
| scripts\debug_stage_detector.py | CLI flags |
| test_policy.py | CLI flags |
| test_ppo.py | CLI flags |
| tools\diagnose_throughput.py | CLI flags, dataclass fields |
| tools\test_room_metrics.py | CLI flags |
| train_bc.py | CLI flags |
| train_ppo.py | CLI flags |
| train_supervisor.py | CLI flags |

## Flow

- `train_supervisor.py`: reads `configs/night_training.json`, spawns `train_ppo.py` with `--config run_dir/config_snapshot.json` or legacy flags.
- `train_ppo.py`: parses CLI; if `--config` → `project_config.load_config()` else `parse_args()` + `project_config.build_resolved_config_from_args()`.
- `project_config.py`: single `load_config(argv)` → ResolvedConfig (RunConfig, PPOConfig, EnvConfigSchema, RewardConfig, CaptureConfig, RunLayout).
- `msx_env.env`: receives EnvConfig built from ResolvedConfig.env_schema + reward_config.
- `msx_env.reward`: RewardConfig from ResolvedConfig.reward_config.
- `msx_env.capture`: backend from EnvConfig.capture_backend (resolved in project_config).
