"""Visual Transition Token on RMBench (plan step 5).

Identical definition to the main project's utils/precompute_task_cond.py:
    diff_e = CLS_lastlayer(last frame) - CLS_lastlayer(first frame)
    VTT_task = mean_e diff_e
but frames are decoded with decode_image_bit (RMBench format) and the per-episode
diffs are kept, so that task separability can be analysed (cosine matrix, t-SNE).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .episodes import EpisodeReader, EpisodeRef
from .image import to_model_input


@torch.no_grad()
def episode_diff(model, ref: EpisodeRef, camera: str, image_size, device, dtype) -> np.ndarray | None:
    with EpisodeReader(ref.path) as r:
        n = r.num_frames(camera)
        if n < 2:
            return None
        frames = r.read_frames(camera, [0, n - 1])
    batch = np.stack([to_model_input(f, image_size) for f in frames])
    px = torch.from_numpy(batch).to(device, dtype)
    cls = model(pixel_values=px, return_dict=True).last_hidden_state[:, 0, :].float().cpu().numpy()
    return (cls[1] - cls[0]).astype(np.float32)


def cosine_matrix(vecs: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = sorted(vecs)
    m = np.stack([vecs[n] / (np.linalg.norm(vecs[n]) + 1e-8) for n in names])
    return names, m @ m.T


def compute_and_save(model, refs: list[EpisodeRef], camera: str, image_size, device, dtype,
                     out_dir: Path, meta_extra: dict | None = None, progress=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs: dict[str, list[np.ndarray]] = defaultdict(list)
    ep_ids: dict[str, list[int]] = defaultdict(list)
    it = progress(refs) if progress else refs
    for ref in it:
        d = episode_diff(model, ref, camera, image_size, device, dtype)
        if d is not None:
            diffs[ref.task].append(d)
            ep_ids[ref.task].append(ref.episode_id)

    vtts: dict[str, np.ndarray] = {}
    meta = {"feature": "last_layer_cls_diff (last - first), mean per task",
            "camera": camera, "image_size": list(image_size), "tasks": {}, **(meta_extra or {})}
    for task, ds in sorted(diffs.items()):
        arr = np.stack(ds)
        vtt = arr.mean(0).astype(np.float32)
        vtts[task] = vtt
        (out_dir / task).mkdir(exist_ok=True)
        np.save(out_dir / task / "task_cond.npy", vtt)
        np.savez(out_dir / task / "episode_diffs.npz", diffs=arr, episode_ids=np.array(ep_ids[task]))
        # within-task consistency: mean cosine of each episode diff to the task VTT
        cos = (arr @ vtt) / (np.linalg.norm(arr, axis=1) * np.linalg.norm(vtt) + 1e-8)
        meta["tasks"][task] = {"num_episodes": int(arr.shape[0]), "dim": int(vtt.shape[0]),
                               "norm": float(np.linalg.norm(vtt)),
                               "mean_cos_episode_to_vtt": float(cos.mean())}

    names, sim = cosine_matrix(vtts)
    with open(out_dir / "cosine_similarity.csv", "w") as f:
        f.write("task," + ",".join(names) + "\n")
        for n, row in zip(names, sim):
            f.write(n + "," + ",".join(f"{x:.4f}" for x in row) + "\n")
    off = sim[~np.eye(len(names), dtype=bool)] if len(names) > 1 else np.array([0.0])
    pairs = sorted(((sim[i, j], names[i], names[j]) for i in range(len(names))
                    for j in range(i + 1, len(names))), reverse=True)
    meta["similarity"] = {"max_offdiag": float(off.max()), "mean_offdiag": float(off.mean()),
                          "most_similar_pairs": [[a, b, round(float(s), 4)] for s, a, b in pairs[:5]]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta
