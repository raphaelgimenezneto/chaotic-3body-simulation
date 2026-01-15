import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional


def _safe_run(cmd: list[str]) -> str:
    """Run a command and return stdout, or 'unknown' if it fails."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def get_git_commit() -> str:
    return _safe_run(["git", "rev-parse", "HEAD"])


def get_git_status_porcelain() -> str:
    # Empty string means clean working tree.
    return _safe_run(["git", "status", "--porcelain"])


def get_pip_freeze() -> str:
    # Uses current Python environment.
    return _safe_run(["python", "-m", "pip", "freeze"])


def snapshot_config(cfg_module) -> Dict[str, Any]:
    """
    Convert a config module into a JSON-serializable dict,
    keeping only ALL_CAPS variables.
    """
    snap: Dict[str, Any] = {}
    for k, v in vars(cfg_module).items():
        if not k.isupper():
            continue
        try:
            json.dumps(v)
            snap[k] = v
        except TypeError:
            snap[k] = str(v)
    return snap


def make_run_dir(base_dir: str = "outputs", prefix: Optional[str] = None) -> str:
    """
    Create a timestamped run directory like:
      outputs/2026-01-15_06-12-03
    or with prefix:
      outputs/vis_2026-01-15_06-12-03
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{prefix}_{ts}" if prefix else ts
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
