"""Locations of the sub-project, the LiLa-WAM main project and RMBench.

The main project is located (in order):
  1. env var ``LILA_WAM_ROOT``
  2. the parent directory of this sub-project (default layout: LiLa-WAM/rmbench_exp)

The main project is appended to ``sys.path`` (appended, not prepended), so that
nothing in it can shadow modules of this package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]


def lila_root() -> Path:
    env = os.environ.get("LILA_WAM_ROOT")
    root = Path(env).expanduser().resolve() if env else SUBPROJECT_ROOT.parent
    if not (root / "models" / "model_runner.py").is_file():
        raise RuntimeError(
            f"LiLa-WAM main project not found at {root} (expected models/model_runner.py). "
            "Set LILA_WAM_ROOT to the LiLa-WAM checkout."
        )
    return root


def ensure_lila_importable() -> Path:
    root = lila_root()
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def rmbench_root() -> Path | None:
    """RMBench checkout, if ``RMBENCH_ROOT`` is set (only needed for eval/adapter install)."""
    env = os.environ.get("RMBENCH_ROOT")
    return Path(env).expanduser().resolve() if env else None


def resolve(path: str | os.PathLike, base: Path = SUBPROJECT_ROOT) -> Path:
    """Resolve a config path: absolute stays absolute, relative is relative to ``base``."""
    p = Path(os.path.expandvars(str(path))).expanduser()
    return p if p.is_absolute() else (base / p).resolve()
