"""
Configuration inventory: scan repo for argparse flags, dataclass fields, env vars.
Produces docs/CONFIG_INVENTORY.md (grouped list) and docs/CONFIG_GRAPH.md (which module consumes which).
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _scan_argparse(file_path: Path) -> list[tuple[str, str]]:
    """Return list of (dest_or_name, help_or_default) for add_argument calls."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    tree = ast.parse(text)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = getattr(node.func, "attr", None) or (getattr(node.func, "id", None) if hasattr(node.func, "id") else None)
            if f == "add_argument":
                names = []
                help_str = ""
                dest = ""
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        try:
                            dest = str(ast.literal_eval(kw.value))
                        except Exception:
                            pass
                    elif kw.arg == "help" and isinstance(kw.value, (ast.Constant,)):
                        help_str = str(ast.literal_eval(kw.value))
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        names.append(arg.value)
                name = names[0] if names else dest or "?"
                out.append((name, help_str[:80] if help_str else ""))
    return out


def _scan_dataclass_fields(file_path: Path) -> list[tuple[str, str]]:
    """Return list of (field_name, annotation_or_default) for @dataclass classes."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    tree = ast.parse(text)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass") or
                (isinstance(d, ast.Call) and getattr(d.func, "id", None) == "dataclass")
                for d in node.decorator_list
            )
            if is_dataclass:
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        name = stmt.target.id
                        try:
                            ann = ast.get_source_segment(text, stmt.annotation) or ""
                        except Exception:
                            ann = ""
                        out.append((name, ann))
    return out


def _scan_env_vars(file_path: Path) -> list[str]:
    """Return list of os.environ.get / os.getenv first argument strings."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return []
    # Simple regex for os.environ.get("X") or os.getenv("X")
    out = []
    for m in re.finditer(r'os\.(?:environ\.get|getenv)\s*\(\s*["\']([^"\']+)["\']', text):
        out.append(m.group(1))
    return out


def run_inventory() -> dict:
    """Collect all config sources into a structure."""
    inventory = {
        "argparse": {},
        "dataclasses": {},
        "env_vars": [],
    }
    for py_path in sorted(ROOT.rglob("*.py")):
        rel = py_path.relative_to(ROOT)
        if "venv" in str(rel) or "__pycache__" in str(rel) or ".git" in str(rel):
            continue
        args = _scan_argparse(py_path)
        if args:
            inventory["argparse"][str(rel)] = args
        dcs = _scan_dataclass_fields(py_path)
        if dcs:
            inventory["dataclasses"][str(rel)] = dcs
        ev = _scan_env_vars(py_path)
        if ev:
            inventory["env_vars"].extend([(str(rel), v) for v in ev])
    inventory["env_vars"] = list(dict.fromkeys([v for _, v in inventory["env_vars"]]))
    return inventory


def write_inventory_md(inv: dict, path: Path) -> None:
    """Write docs/CONFIG_INVENTORY.md."""
    lines = [
        "# Configuration inventory",
        "",
        "Grouped list of configuration sources (argparse flags, dataclass fields, env vars).",
        "",
        "## Argparse flags (by file)",
        "",
    ]
    for file_path, args in sorted(inv["argparse"].items()):
        lines.append(f"### {file_path}")
        lines.append("")
        for name, help_str in args:
            lines.append(f"- `{name}`: {help_str or '(no help)'}")
        lines.append("")
    lines.append("## Dataclass config fields (by file)")
    lines.append("")
    for file_path, fields in sorted(inv["dataclasses"].items()):
        if "config" in file_path.lower() or "env" in file_path or "reward" in file_path or file_path == "project_config.py":
            lines.append(f"### {file_path}")
            lines.append("")
            for name, ann in fields:
                lines.append(f"- `{name}`: `{ann}`")
            lines.append("")
    lines.append("## Environment variables")
    lines.append("")
    for v in sorted(inv["env_vars"]):
        lines.append(f"- `{v}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_graph_md(inv: dict, path: Path) -> None:
    """Write docs/CONFIG_GRAPH.md (which module consumes which fields)."""
    lines = [
        "# Configuration graph",
        "",
        "Which module consumes which configuration fields.",
        "",
        "| Module | Consumes (sources) |",
        "|--------|--------------------|",
    ]
    for file_path in sorted(set(inv["argparse"].keys()) | set(inv["dataclasses"].keys())):
        sources = []
        if file_path in inv["argparse"]:
            sources.append("CLI flags")
        if file_path in inv["dataclasses"]:
            sources.append("dataclass fields")
        if sources:
            lines.append(f"| {file_path} | {', '.join(sources)} |")
    lines.append("")
    lines.append("## Flow")
    lines.append("")
    lines.append("- `train_supervisor.py`: reads `configs/night_training.json`, spawns `train_ppo.py` with `--config run_dir/config_snapshot.json` or legacy flags.")
    lines.append("- `train_ppo.py`: parses CLI; if `--config` → `project_config.load_config()` else `parse_args()` + `project_config.build_resolved_config_from_args()`.")
    lines.append("- `project_config.py`: single `load_config(argv)` → ResolvedConfig (RunConfig, PPOConfig, EnvConfigSchema, RewardConfig, CaptureConfig, RunLayout).")
    lines.append("- `msx_env.env`: receives EnvConfig built from ResolvedConfig.env_schema + reward_config.")
    lines.append("- `msx_env.reward`: RewardConfig from ResolvedConfig.reward_config.")
    lines.append("- `msx_env.capture`: backend from EnvConfig.capture_backend (resolved in project_config).")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inv = run_inventory()
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    write_inventory_md(inv, docs / "CONFIG_INVENTORY.md")
    write_graph_md(inv, docs / "CONFIG_GRAPH.md")
    print("Wrote docs/CONFIG_INVENTORY.md and docs/CONFIG_GRAPH.md")


if __name__ == "__main__":
    main()
