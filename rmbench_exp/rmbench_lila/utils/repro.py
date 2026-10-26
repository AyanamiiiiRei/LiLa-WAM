"""Reproducibility helpers: seeding and run metadata (git commits, command line)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct but reproducible numpy/random seed.

    Workers also ignore Ctrl+C: the main process then finishes the current step and saves
    a resume point instead of crashing because its workers died first."""
    import signal
    import torch
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_state(path: Path) -> dict:
    return {
        "commit": _git(["rev-parse", "HEAD"], path),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], path),
        "dirty": bool(_git(["status", "--porcelain", "--", "."], path)),
    }


def write_run_info(run_dir: Path, extra: dict | None = None) -> dict:
    from ..paths import SUBPROJECT_ROOT, lila_root
    info = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "subproject_git": git_state(SUBPROJECT_ROOT),
        "lila_wam_git": git_state(lila_root()),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    if extra:
        info.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")
