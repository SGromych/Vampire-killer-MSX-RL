"""
Safety net tests for the central config system:
- Strict missing reward-config -> fails
- Resolved paths are absolute and under run_dir
- Supervisor spawn passes same config (--config snapshot)
- Resume architecture mismatch emits clear error
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestConfigSystem(unittest.TestCase):
    def test_strict_missing_reward_config_fails(self) -> None:
        """If --reward-config is provided and file missing -> fail (no silent default)."""
        from project_config import load_config
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nonexistent_reward.json"
            self.assertFalse(missing.exists())
            with self.assertRaises(FileNotFoundError) as ctx:
                load_config([
                    "--reward-config", str(missing), "--epochs", "1",
                    "--run-dir", td,
                ])
            self.assertIn("not found", str(ctx.exception).lower()) or self.assertIn("reward", str(ctx.exception).lower())

    def test_resolved_paths_absolute_and_under_run_dir(self) -> None:
        """Resolved paths are absolute and metrics/logs under run_dir (requires ROM)."""
        from project_config import load_config
        rom = ROOT / "VAMPIRE.ROM"
        if not rom.exists():
            self.skipTest("VAMPIRE.ROM not found")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td).resolve()
            config = load_config([
                "--run-dir", str(run_dir), "--epochs", "1", "--rollout-steps", "8",
            ])
            self.assertTrue(config.run.run_dir.is_absolute())
            self.assertTrue(config.layout.metrics_csv().resolve().is_absolute())
            run_abs = config.run.run_dir.resolve()
            self.assertTrue(
                str(config.layout.metrics_csv().resolve()).startswith(str(run_abs)),
                "metrics_csv should be under run_dir",
            )

    def test_supervisor_build_argv_uses_config_snapshot_when_exists(self) -> None:
        """When run_dir has config_snapshot.json, build_argv returns --config <path> --resume."""
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from train_supervisor import build_argv
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            snapshot = run_dir / "config_snapshot.json"
            snapshot.write_text("{}")
            argv = build_argv(
                {"num_envs": 1, "run_name": "x", "max_updates": 10, "checkpoint_dir": "ck", "entropy_floor": 0.3},
                run_dir_override=run_dir,
            )
            self.assertIn("--config", argv)
            self.assertIn("--resume", argv)
            self.assertTrue(any("config_snapshot" in a for a in argv))

    def test_resume_mismatch_emits_clear_error(self) -> None:
        """Resume with arch mismatch -> ValueError with diff."""
        if str(ROOT) not in __import__("sys").path:
            __import__("sys").path.insert(0, str(ROOT))
        from train_ppo import _validate_resume_signature
        ckpt = {"arch": "default", "frame_stack": 4, "recurrent": False, "lstm_hidden_size": 256}
        expected = {"arch": "deep", "frame_stack": 4, "recurrent": False, "lstm_hidden_size": 256}
        with self.assertRaises(ValueError) as ctx:
            _validate_resume_signature(ckpt, expected)
        self.assertIn("mismatch", str(ctx.exception).lower()) or self.assertIn("arch", str(ctx.exception).lower())
        expected_same = {"arch": "default", "frame_stack": 4, "recurrent": False, "lstm_hidden_size": 256}
        _validate_resume_signature(ckpt, expected_same)  # no raise
