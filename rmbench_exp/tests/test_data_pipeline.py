"""Tests for everything before the model: codec, schema, episodes, stats, config, dataset,
inference-time history indexing, and the evaluation statistics.

Run: cd rmbench_exp && python -m pytest tests -q
Needs numpy, h5py, opencv, omegaconf, scipy (+ torch for the dataset tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import N_TASKS, frame


# ------------------------------------------------------------------ codec
def test_codec_roundtrip_is_rgb():
    from rmbench_lila._vendor.xpl_image_codec import encode_image_bit
    from rmbench_lila.codec import decode_image_bit, has_rgb_marker
    img = frame(10)
    buf = encode_image_bit(img)
    assert has_rgb_marker(buf)
    dec = decode_image_bit(buf)
    assert dec[..., 0].mean() > 150 and dec[..., 2].mean() < 30, "red must stay in channel 0"


# ----------------------------------------------------------------- schema
def test_layout_dims():
    from rmbench_lila.schema import STATE_EE16, ACTION_JOINT14
    assert STATE_EE16.dim == 16 and ACTION_JOINT14.dim == 14
    assert [sl.start for _, sl in ACTION_JOINT14.slices()] == [0, 6, 7, 13]


# --------------------------------------------------------------- episodes
def test_scan_and_split(data_root):
    from rmbench_lila.episodes import scan_episodes, split_episodes
    refs = scan_episodes(data_root)
    assert {r.task for r in refs} == set(N_TASKS) and len(refs) == 8
    train, held = split_episodes(refs, 1)
    assert len(train) == 6 and len(held) == 2
    assert all(h.episode_id == 3 for h in held)          # last episode of each task held out


def test_alignment_next_vs_current(data_root):
    from rmbench_lila.episodes import EpisodeReader, scan_episodes
    from rmbench_lila.schema import ACTION_JOINT14
    ref = scan_episodes(data_root)[0]
    with EpisodeReader(ref.path) as r:
        nxt = r.read_action(ACTION_JOINT14, "next")
        cur = r.read_action(ACTION_JOINT14, "current")
    np.testing.assert_allclose(nxt[:-1], cur[1:])        # action[t] == state[t+1]


def test_frames_are_indexed_correctly(data_root):
    from rmbench_lila.episodes import EpisodeReader, scan_episodes
    ref = scan_episodes(data_root)[0]
    with EpisodeReader(ref.path) as r:
        f = r.read_frames("head_camera", [0, 5, 5])
    assert abs(int(f[1][..., 1].mean()) - 20) <= 3 and np.array_equal(f[1], f[2])


# ------------------------------------------------------------------ stats
def test_stats_never_touches_files(data_root, cfg):
    import os
    from rmbench_lila.episodes import select_episodes
    from rmbench_lila.schema import get_action_layout, get_state_layout
    from rmbench_lila.stats import compute_stats
    refs = select_episodes(cfg, "train")
    before = {r.path: os.stat(r.path).st_mtime for r in refs}
    st = compute_stats(refs, get_state_layout("ee16"), get_action_layout("joint14"), "next", num_workers=1)
    assert {r.path: os.stat(r.path).st_mtime for r in refs} == before
    s = st["robotwin2"]
    assert len(s["action"]["min"]) == 14 and len(s["state"]["max"]) == 16
    assert s["meta"]["dataset"] == "rmbench"


# ----------------------------------------------------------------- config
def test_config_validation_catches_mismatch(cfg):
    from omegaconf import OmegaConf
    from rmbench_lila.config import validate
    bad = OmegaConf.merge(cfg, {"common": {"proprio_len": 3}})
    with pytest.raises(ValueError, match="proprio_len"):
        validate(bad)
    bad = OmegaConf.merge(cfg, {"dataset": {"features": ["state", "action_chunk", "image", "task_cond"]}})
    with pytest.raises(ValueError, match="future_image"):
        validate(bad)


def test_exp_configs_load(data_root):
    from pathlib import Path
    from rmbench_lila.config import load_config
    for p in sorted((Path(__file__).parents[1] / "configs" / "exp").glob("*.yaml")):
        load_config(p, [f"rmbench.data_root={data_root}", "model.vision_encoder.checkpoint_path=/x"])


# ---------------------------------------------------------------- dataset
def test_dataset_sample(cfg):
    torch = pytest.importorskip("torch")
    from rmbench_lila.data import build_dataset, collate
    ds = build_dataset(cfg, "train")
    s = ds[0]
    assert s["state"].shape == (1, 16)
    assert s["action_sequence"].shape == (32, 14)
    assert s["pixel_values"].shape == (3, 48, 64)
    assert s["future_pixel_values"].shape == (3, 48, 64)
    assert s["task_cond"].shape == (8,)
    b = collate([ds[0], ds[1]])
    assert b["action_sequence"].shape == (2, 32, 14) and isinstance(b["task_name"], list)


def test_dataset_state_history_clips_at_start(cfg):
    pytest.importorskip("torch")
    from omegaconf import OmegaConf
    from rmbench_lila.data import build_dataset
    c = OmegaConf.merge(cfg, {"common": {"proprio_len": 3},
                              "dataset": {"indices_config": {"state_indices": [-10, -1, 0]}}})
    ds = build_dataset(c, "train")
    s = ds[0]                                    # anchor 0: all history clipped to frame 0
    assert np.allclose(s["state"][0], s["state"][2]) and s["state_mask"].tolist() == [False, False, True]


def test_action_chunk_pads_with_last(cfg):
    pytest.importorskip("torch")
    from rmbench_lila.data import build_dataset
    ds = build_dataset(cfg, "train")
    n = int(ds.lengths[0])
    s = ds[n - 1]                                # last anchor of episode 0
    a = s["action_sequence"].numpy()
    assert np.allclose(a[0], a[-1]) and s["action_mask"].sum() == 1


# ------------------------------------------------- inference history logic
def test_policy_history_matches_training_clip():
    pytest.importorskip("torch")
    from collections import deque
    from rmbench_lila.policy import LiLaWAMPolicy
    pol = object.__new__(LiLaWAMPolicy)          # no model needed for the indexing logic
    pol.state_offsets = [-10, -1, 0]
    pol.states = deque(maxlen=11)
    for t in range(3):
        pol.states.append(np.full(16, t, np.float32))
    h = pol._state_history()
    assert h[:, 0].tolist() == [0, 1, 2]         # -10 clipped to the first observed state
    for t in range(3, 30):
        pol.states.append(np.full(16, t, np.float32))
    assert pol._state_history()[:, 0].tolist() == [19, 28, 29]


# ------------------------------------------------------------- evalstats
def test_wilson_and_mcnemar():
    from rmbench_lila.evalstats import mcnemar_exact, wilson_ci
    lo, hi = wilson_ci(25, 50)
    assert 0.36 < lo < 0.37 and 0.63 < hi < 0.64
    a = {s: s < 10 for s in range(50)}
    b = {s: s < 30 for s in range(50)}
    m = mcnemar_exact(a, b)
    assert m["b_only"] == 20 and m["a_only"] == 0 and m["p_value"] < 1e-5
