"""Shared bootstrap for the step scripts: makes ``rmbench_lila`` importable and parses
``--config`` plus trailing ``key=value`` overrides (OmegaConf dotlist)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

DEFAULT_CONFIG = SUBPROJECT_ROOT / "configs" / "exp" / "baseline.yaml"


def parser(description: str, config: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    if config:
        p.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment config")
        p.add_argument("overrides", nargs="*", help="config overrides, e.g. training.batch_size=16")
    return p


def load(args):
    from rmbench_lila.config import load_config
    return load_config(args.config, args.overrides)


def pick_device_dtype(device: str = "cuda"):
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU (slow)")
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    return device, dtype
