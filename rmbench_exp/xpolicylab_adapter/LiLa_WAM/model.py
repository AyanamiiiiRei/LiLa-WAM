"""XPolicyLab adapter for LiLa-WAM trained by rmbench_exp (plan step 9).

This folder lives in rmbench_exp/xpolicylab_adapter/LiLa_WAM and is SYMLINKED into
RMBench/XPolicyLab/policy/LiLa_WAM by scripts/step09_install_adapter.sh. All model logic
is in rmbench_lila.policy (shared with open-loop eval); this file only converts between
XPolicyLab observation/action dicts and plain vectors, and logs every episode.

Eval loop contract (rmbench/scripts/eval_policy_xpolicylab.py):
  prepare_case(meta) -> reset() -> update_obs(obs) -> get_action() -> [take each action,
  update_obs after each except the last] -> get_action() ... -> on_trial_end(result)
so update_obs is called exactly once per env step, which keeps the state history aligned
with the 15 Hz training frames.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info


def _subproject_root(cfg) -> Path:
    root = cfg.get("rmbench_exp_root") or os.environ.get("RMBENCH_EXP_ROOT")
    # .../rmbench_exp/xpolicylab_adapter/LiLa_WAM/model.py (resolve() follows the symlink)
    path = Path(root).expanduser().resolve() if root else Path(__file__).resolve().parents[2]
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        root = _subproject_root(model_cfg)

        import torch
        from rmbench_lila.policy import LiLaWAMPolicy
        from rmbench_lila.schema import camera_h5_name, get_action_layout, get_state_layout

        if model_cfg.get("action_type") != "joint":
            raise ValueError(self._error_msg("LiLa-WAM outputs absolute joint targets: use action_type=joint"))
        dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if list(dims["arm_dim"]) != [6, 6] or list(dims["ee_dim"]) != [1, 1]:
            raise ValueError(self._error_msg(
                f"env_cfg_type={model_cfg['env_cfg_type']} has arm_dim={dims['arm_dim']}; "
                "the RMBench data (and this model) is aloha_agilex (6+1, 6+1). Use env_cfg_type=aloha_agilex."))

        run_dir = model_cfg.get("run_dir") or (root / "runs" / str(model_cfg["ckpt_name"]))
        self.run_dir = Path(run_dir)
        device = model_cfg.get("device", "cuda:0")
        self.policy = LiLaWAMPolicy.from_run(self.run_dir, str(model_cfg.get("checkpoint", "latest")), device)
        pc = self.policy.cfg
        self.state_layout = get_state_layout(pc.rmbench.state_layout)
        self.action_layout = get_action_layout(pc.rmbench.action_layout)
        self.camera = camera_h5_name(list(pc.dataset.camera_names)[0])
        self.horizon = int(model_cfg.get("execution_horizon") or pc.common.action_execution_horizon)
        seed = model_cfg.get("noise_seed")
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
            self.gen.manual_seed(int(seed))

        self.log_dir = self.run_dir / "rmbench_eval" / f"ckpt_{self.policy.checkpoint_step}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.task = None
        self.case: dict = {}
        self._reset_counters()
        if model_cfg.get("task_name"):
            self._set_task(model_cfg["task_name"])
        print(f"[LiLa_WAM] ready: run={self.run_dir} ckpt_step={self.policy.checkpoint_step} "
              f"horizon={self.horizon} task={self.task}")

    # ------------------------------------------------------------ conversion
    def _state_vector(self, obs) -> np.ndarray:
        st = obs["state"]
        missing = [f.obs_key for f in self.state_layout.fields if f.obs_key not in st]
        if missing:
            raise KeyError(self._error_msg(f"obs['state'] lacks {missing}; has {sorted(st)}"))
        return np.concatenate([np.asarray(st[f.obs_key], dtype=np.float32).reshape(-1)
                               for f in self.state_layout.fields])

    def _rgb(self, obs) -> np.ndarray:
        img = np.asarray(obs["vision"][self.camera]["color"])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img[..., :3]          # already RGB (decoded by the server) — no colour swap

    def _action_dicts(self, chunk: np.ndarray) -> list[dict]:
        out = []
        for row in chunk:
            out.append({f.obs_key: row[sl].astype(np.float32) for f, sl in self.action_layout.slices()})
        return out

    # ------------------------------------------------------------- lifecycle
    def _set_task(self, task: str) -> None:
        if task != self.task:
            self.policy.set_task(task)
            self.task = task

    def _reset_counters(self) -> None:
        self.n_steps = 0
        self.n_chunks = 0
        self.t_start = time.time()

    def prepare_case(self, case_meta=None):
        self.case = dict(case_meta or {})
        if self.case.get("task_name"):
            self._set_task(self.case["task_name"])

    def reset(self):
        self.policy.reset()
        self._reset_counters()

    def update_obs(self, obs):
        self.policy.observe(self._state_vector(obs), self._rgb(obs))
        self.n_steps += 1

    def update_obs_batch(self, obs_list):
        raise NotImplementedError(self._error_msg("LiLa_WAM keeps per-episode history; set eval_batch: false"))

    def get_action(self):
        chunk = self.policy.predict_chunk(self.gen)
        self.n_chunks += 1
        return self._action_dicts(chunk[: self.horizon])

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError(self._error_msg("batch evaluation not supported"))

    def on_trial_end(self, result=None):
        rec = {"task": self.task, "seed": (result or {}).get("seed", self.case.get("seed")),
               "success": bool((result or {}).get("success", False)),
               "env_steps": self.n_steps, "chunks": self.n_chunks,
               "wall_s": round(time.time() - self.t_start, 1),
               "run_dir": str(self.run_dir), "ckpt_step": self.policy.checkpoint_step,
               "noise_seed": self.cfg.get("noise_seed"), "eval_seed_arg": self.cfg.get("seed"),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(self.log_dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
