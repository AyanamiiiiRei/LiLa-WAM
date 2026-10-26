"""Synthetic RMBench-format episodes, written exactly like rmbench's pkl2hdf5.py
(state = x[:-1], action = x[1:], frames JPEG-encoded with encode_image_bit), so the
whole data path can be tested without the real dataset, a GPU or DINOv3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_TASKS = ("cover_blocks", "swap_T")
W, H = 64, 48


def frame(t: int) -> np.ndarray:
    """Pure-red image whose green channel encodes t (lets tests check frame indexing)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 1] = min(t * 4, 255)
    return img


def write_episode(path: Path, n_rec: int, seed: int) -> None:
    import h5py
    from rmbench_lila._vendor.xpl_image_codec import images_encoding

    rng = np.random.default_rng(seed)
    t = np.arange(n_rec, dtype=np.float32)[:, None]
    arm_l = np.sin(t / 10 + rng.uniform(0, 1, 6)).astype(np.float32)
    arm_r = np.cos(t / 10 + rng.uniform(0, 1, 6)).astype(np.float32)
    grip_l = (np.sin(t / 7) * 0.5 + 0.5).astype(np.float32)
    grip_r = (np.cos(t / 7) * 0.5 + 0.5).astype(np.float32)
    quat = np.tile(np.array([1, 0, 0, 0], np.float32), (n_rec, 1))
    pose_l = np.concatenate([t / 100, t / 200, t / 300, quat], 1).astype(np.float32)
    pose_r = np.concatenate([-t / 100, t / 200, t / 300, quat], 1).astype(np.float32)
    rec = {"left_arm_joint_states": arm_l, "left_ee_joint_states": grip_l,
           "right_arm_joint_states": arm_r, "right_ee_joint_states": grip_r,
           "left_ee_poses": pose_l, "right_ee_poses": pose_r}
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_group("additional_info").create_dataset("frequency", data=np.int32(15))
        s, a = f.create_group("state"), f.create_group("action")
        for k, v in rec.items():
            s.create_dataset(k, data=v[:-1])
            a.create_dataset(k, data=v[1:])
        enc, max_len = images_encoding(np.stack([frame(i) for i in range(n_rec)])[:-1])
        cam = f.create_group("vision").create_group("cam_head")
        cam.create_dataset("colors", data=enc, dtype=f"S{max_len}")


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("rmbench") / "demo_clean"
    for ti, task in enumerate(N_TASKS):
        for e in range(4):
            write_episode(root / task / "aloha_agilex" / "data" / f"episode_{e:07d}.hdf5",
                          n_rec=40 + 5 * e, seed=100 * ti + e)
    return root


@pytest.fixture()
def cfg(data_root, tmp_path):
    """Base config pointed at the synthetic data, small images, fake VTTs and stats."""
    from rmbench_lila.config import load_config
    vtt = tmp_path / "vtt"
    for task in N_TASKS:
        (vtt / task).mkdir(parents=True)
        np.save(vtt / task / "task_cond.npy", np.ones(8, np.float32))
    return load_config(ROOT / "configs" / "exp" / "baseline.yaml", [
        f"rmbench.data_root={data_root}", f"dataset.task_cond_dir={vtt}",
        f"paths.norm_stats={tmp_path / 'stats.json'}", f"dataset.image_size=[{W},{H}]",
        "model.vision_encoder.checkpoint_path=/nonexistent",
    ])
