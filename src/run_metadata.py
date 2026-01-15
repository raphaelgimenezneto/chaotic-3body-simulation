import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict


def _safe_run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def get_git_commit() -> str:
    return _safe_run(["git", "rev-parse", "HEAD"])


def get_git_status_porcelain() -> str:
    # Empty string means clean working tree
    return _safe_run(["git", "status", "--porcelain"])


def get_pip_freeze() -> str:
    # Uses current interpreter environment
    return _safe_run(["python", "-m", "pip", "freeze"])


def snapshot_config(cfg_module) -> Dict[str, Any]:
    """
    Convert a config module into a JSON-serializable dict,
    keeping only ALL_CAPS variables.
    """
    d = {}
    for k, v in vars(cfg_module).items():
        if k.isupper():
            # try JSON serialization; fall back to string
            try:
                json.dumps(v)
                d[k] = v
            except TypeError:
                d[k] = str(v)
    return d


def make_run_dir(base_dir: str = "outputs") -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
