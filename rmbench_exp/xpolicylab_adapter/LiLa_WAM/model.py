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
        self.policy = LiLaWAMPolicy.from_run(self.run_dir, str(model_cfg.get("checkpoint", "latest")), device,
                                             memory_override=self._opt(model_cfg, "memory_override", "LILA_MEMORY_OVERRIDE"))
        self.stages_done = None      # from obs["oracle"] (stage-memory runs, eval launcher with oracle)
        # Phase 2: what triggers memory writes (rmbench_lila/triggers.py). Default: the oracle.
        self.trigger = self._opt(model_cfg, "stage_trigger", "LILA_STAGE_TRIGGER") or "oracle"
        self._init_trigger(root)
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

        tag = self._opt(model_cfg, "eval_tag", "LILA_EVAL_TAG")   # e.g. "M-empty": separate results folder
        self.log_dir = self.run_dir / "rmbench_eval" / (f"ckpt_{self.policy.checkpoint_step}" + (f"_{tag}" if tag else ""))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Policy-view recording: the head-camera frames the model actually receives (RMBench's own
        # eval video is a third-person camera), plus states and executed actions, per episode.
        self.record = bool(model_cfg.get("record_policy_view", True))
        self.rec_dir = self.log_dir / "policy_view"
        self._writer = None
        self._rec_states: list = []
        self._rec_actions: list = []
        self.task = None
        self.case: dict = {}
        self._reset_counters()
        if model_cfg.get("task_name"):
            self._set_task(model_cfg["task_name"])
        print(f"[LiLa_WAM] ready: run={self.run_dir} ckpt_step={self.policy.checkpoint_step} "
              f"horizon={self.horizon} task={self.task}")

    def _init_trigger(self, root: Path) -> None:
        from rmbench_lila.triggers import TRIGGERS, ProprioTrigger, SwitchHead, TimeTrigger
        if self.trigger not in TRIGGERS:
            raise ValueError(self._error_msg(f"stage_trigger must be one of {TRIGGERS}, got {self.trigger!r}"))
        self.proprio = self.time_trig = self.head = None
        if self.trigger != "oracle" and (self.policy.memory is None or self.policy.memory.mode != "stage"):
            raise ValueError(self._error_msg(f"stage_trigger={self.trigger} needs a stage-memory run"))
        task = self.cfg.get("task_name") or "swap_blocks"
        art = root / "artifacts" / "stage_memory" / task
        if self.trigger == "proprio":
            self.proprio = ProprioTrigger(int(self.policy.cfg.model.stage_memory.num_slots))
        elif self.trigger == "time":
            ends = json.loads((art / "time_trigger.json").read_text())["ends_env_steps"]
            self.time_trig = TimeTrigger(ends)
        elif self.trigger == "head":
            path = self._opt(self.cfg, "switch_head", "LILA_SWITCH_HEAD") or str(art / "switch_head.pt")
            self.head = SwitchHead(path)

    @staticmethod
    def _opt(cfg, key, env):
        """deploy.yml value, else the environment variable (set by scripts/step24_eval_with_oracle.sh)."""
        v = cfg.get(key) or os.environ.get(env) or None
        return None if v in ("", "null", "none", "None") else v

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
        self.stages_done = None
        self.oracle_trace: list = []       # oracle stages_done at every chunk (all triggers)
        if self.proprio is not None:
            self.proprio.reset()
        if self.head is not None:
            self.head.reset()
        self._reset_counters()
        self._close_recording(None)          # an episode that ended without on_trial_end

    # ---------------------------------------------------------- recording
    def _rec_name(self, seed) -> str:
        return f"{self.task}_seed{seed}"

    def _record_frame(self, rgb: np.ndarray, state: np.ndarray) -> None:
        if not self.record:
            return
        import cv2
        if self._writer is None:
            self.rec_dir.mkdir(parents=True, exist_ok=True)
            h, w = rgb.shape[:2]
            self._tmp_path = self.rec_dir / f"_recording_{os.getpid()}.mp4"
            self._writer = cv2.VideoWriter(str(self._tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        frame = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)   # VideoWriter expects BGR
        cv2.putText(frame, f"t={self.n_steps}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self._writer.write(frame)
        self._rec_states.append(state)

    def _close_recording(self, seed) -> None:
        if self._writer is None:
            self._rec_states, self._rec_actions = [], []
            return
        self._writer.release()
        self._writer = None
        if seed is None:                       # incomplete episode: discard
            self._tmp_path.unlink(missing_ok=True)
        else:
            name = self._rec_name(seed)
            os.replace(self._tmp_path, self.rec_dir / f"{name}.mp4")
            np.savez_compressed(self.rec_dir / f"{name}.npz",
                                states=np.asarray(self._rec_states, np.float32),
                                actions=np.asarray(self._rec_actions, np.float32))
        self._rec_states, self._rec_actions = [], []

    def update_obs(self, obs):
        state, rgb = self._state_vector(obs), self._rgb(obs)
        oracle = obs.get("oracle") if isinstance(obs, dict) else None
        if oracle is not None:
            self.stages_done = int(oracle.get("stages_done", 0))
        if self.proprio is not None:
            self.proprio.update(state)       # raw ee16 state, every env step (like the training rule)
        self.policy.observe(state, rgb)
        self._record_frame(rgb, state)
        self.n_steps += 1

    def update_obs_batch(self, obs_list):
        raise NotImplementedError(self._error_msg("LiLa_WAM keeps per-episode history; set eval_batch: false"))

    def get_action(self):
        if self.policy.memory is not None and self.policy.memory.mode == "stage" and self.stages_done is None \
                and not self.policy.memory.force_empty:
            raise RuntimeError(self._error_msg("stage-memory run but no obs['oracle']: "
                                               "evaluate with scripts/step24_eval_with_oracle.sh"))
        self.oracle_trace.append(self.stages_done)
        if self.trigger == "oracle":
            chunk = self.policy.predict_chunk(self.gen, stages_done=self.stages_done)
        elif self.trigger == "proprio":
            chunk = self.policy.predict_chunk(self.gen, stages_done=self.proprio.done)
        elif self.trigger == "time":
            chunk = self.policy.predict_chunk(self.gen, stages_done=self.time_trig.stages_done(self.n_steps - 1))
        else:
            chunk = self.policy.predict_chunk(self.gen, head=self.head)
        self.n_chunks += 1
        if self.record:
            self._rec_actions.extend(chunk[: self.horizon])   # may run past the episode end
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
               "stages_done": self.stages_done,          # oracle (all triggers, when available)
               "stage_trigger": self.trigger,
               "oracle_trace": getattr(self, "oracle_trace", None),
               "head_log": list(self.head.log) if self.head is not None else None,
               "memory_writes": list(self.policy.memory.log) if self.policy.memory is not None else None,
               "memory_override": self._opt(self.cfg, "memory_override", "LILA_MEMORY_OVERRIDE"),
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        if self.record and self._writer is not None:
            rec["policy_view"] = str(self.rec_dir / f"{self._rec_name(rec['seed'])}.mp4")
        self._close_recording(rec["seed"])
        with open(self.log_dir / "episodes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
