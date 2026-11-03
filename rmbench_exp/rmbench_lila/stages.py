"""Stage boundaries for the stage-memory experiment (docs/STAGE_MEMORY.md).

A stage is [start, end] in frame indices of the HDF5 episode, both on the 16-frame
action-chunk grid (the policy only observes a new frame every ``grid`` steps).
``end`` is the frame at which the stage's memory slot gets written; it is also the
next stage's ``start``. Only the first ``num_slots`` stages are ever written (the last
stage ends the task).

Boundaries file (boundaries_gt.json):
    {"episode_0000000.hdf5": [[0, 144], [144, 256], [256, 346]], ...}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

GRID = 16


def merge_annotation(segments: Sequence[Sequence]) -> list[tuple[str, int]]:
    """language_annotation.json entry [[desc, n_frames], ...] -> merged consecutive (desc, n)."""
    out: list[list] = []
    for desc, n in segments:
        if out and out[-1][0] == desc:
            out[-1][1] += int(n)
        else:
            out.append([desc, int(n)])
    return [(d, n) for d, n in out]


def raw_ends(merged: Sequence[tuple[str, int]]) -> list[int]:
    """Last frame index of each merged segment (annotation counts include one frame the
    HDF5 drops at the end; clipping to the episode length happens in ``grid_stages``)."""
    return [int(c) - 1 for c in np.cumsum([n for _, n in merged])]


def grid_stages(ends: Sequence[int], length: int, grid: int = GRID) -> list[list[int]]:
    """Round every stage end UP to the grid, clip to the last grid frame < length, and
    chain the stages. Raises if two stages collapse onto the same grid frame."""
    last_grid = ((length - 1) // grid) * grid
    stages, start = [], 0
    for i, e in enumerate(ends):
        if i == len(ends) - 1:
            g = last_grid                      # final stage: runs to the end (never written)
        else:
            g = min(int(np.ceil(e / grid) * grid), last_grid)
        if g <= start:
            raise ValueError(f"stage {i} collapses on the grid: start={start}, end={g} (raw ends {list(ends)})")
        stages.append([start, g])
        start = g
    return stages


def load_boundaries(path: str | Path) -> dict[str, list[list[int]]]:
    d = json.loads(Path(path).read_text())
    return {k: [list(map(int, s)) for s in v] for k, v in d.items()}


def jitter_ends(stages: Sequence[Sequence[int]], num_slots: int, length: int, p: float,
                grid: int = GRID, rng: random.Random | None = None) -> list[int]:
    """Ends of the first ``num_slots`` stages, each moved by ±1 grid step with prob ``p``
    (training only). Ends stay strictly increasing and inside the episode; a move that
    would break that is dropped."""
    rng = rng or random
    ends = [int(s[1]) for s in stages[:num_slots]]
    last_grid = ((length - 1) // grid) * grid
    out: list[int] = []
    for i, e in enumerate(ends):
        if p > 0 and rng.random() < p:
            cand = e + rng.choice((-grid, grid))
            lo = (out[-1] if out else 0) + grid
            hi = (ends[i + 1] - grid) if i + 1 < len(ends) else last_grid
            if lo <= cand <= hi:
                e = cand
        out.append(e)
    return out


def slots_at(t: int, ends: Sequence[int], num_slots: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Which slots are written at frame t (a slot is written once t >= its stage end) and
    the (first, last) frame of each slot. Never uses a stage that ends after t."""
    pairs, written, start = [], np.zeros(num_slots, dtype=bool), 0
    for n in range(num_slots):
        if n < len(ends) and t >= ends[n]:
            pairs.append((start, int(ends[n])))
            written[n] = True
            start = int(ends[n])
        else:
            pairs.append((0, 0))
    return pairs, written


def stage_index(t: int, stages: Sequence[Sequence[int]]) -> int:
    """0-based index of the stage that frame t belongs to (for per-stage loss logging)."""
    for i, (s, e) in enumerate(stages):
        if t < e:
            return i
    return len(stages) - 1
