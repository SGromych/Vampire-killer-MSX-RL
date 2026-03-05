# Throughput diagnostics report

## Config
- Minutes per test: 2.0
- Rollout steps: 128
- Envs: [1, 2]
- Modes: ['capture_on']

## Results

| num_envs | capture | policy | train | steps/s total | steps/s per_env | update_sec p50 | update_sec p95 |
|----------|---------|--------|-------|---------------|-----------------|----------------|----------------|
| 1 | on | random | off | 4.0 | 4.0 | 31.16 | 37.75 |
| 2 | on | random | off | 3.5 | 1.7 | 74.27 | 74.27 |

## Expected vs observed scaling (envs 1→2, random, capture_on, train off)
Expected 2.0×, observed 0.88×

## Top 3 time sinks (p95 ms)
- (no data)

## VERDICT
**S7_METRICS_MISCOUNT**

total_env_steps не учитывает все env (steps envs=2 ≈ steps envs=1)

### Next action
- Проверить: episode_steps_list и total_steps считают sum по всем env, не только env0
