"""The ONLY module in this package that imports LiLa-WAM main-project code.

Contract used (LiLa-WAM @ 339860f / 2bba854):
    models.model_runner.ModelFactory.create_vision_encoder(path, dtype, device)
        -> (dinov3, hidden_size, num_register_tokens, patch_size)
    models.model_runner.ModelFactory.create_action_model(cfg, dino_hidden_size,
        num_dino_layers, task_cond_dim, patch_size) -> VLAModel
        reads cfg.common.{action_dim,state_dim,action_chunk_size,proprio_len},
              cfg.model.{action_expert,vision_encoder,future_feat,num_registers},
              cfg.dataset.image_size
    models.model_runner.VLAWrapper(..., norm_stats_path, train_config, ...)
        reads json["robotwin2"]["action"|"state"]["min"|"max"]
        forward(batch) uses batch keys: pixel_values, action_sequence, state,
                                        task_cond, future_pixel_values
    VLAModel.forward(t, noisy_actions, qpos_history, dino_features_list, task_cond)
        -> {"final_pred": ..., "cond_tokens": ...};  VLAModel.blocks (DiT blocks)

If the main project changes any of the above, adapt it HERE; nothing else in the
sub-project imports the main project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from .paths import ensure_lila_importable
from .schema import NORM_STATS_CONTAINER_KEY  # noqa: F401  (re-exported; key VLAWrapper reads)

ensure_lila_importable()

from models.model_runner import ModelFactory, VLAWrapper  # noqa: E402  (main project)

logger = logging.getLogger(__name__)


@dataclass
class BuiltModel:
    wrapper: Any            # VLAWrapper (frozen DINOv3 + trainable action model + norm stats)
    action_model: Any       # VLAModel (the only trainable part)
    dino_hidden_size: int
    num_register_tokens: int
    patch_size: int


def train_config_dict(cfg: DictConfig) -> dict:
    """Same dict as main train.py:build_train_config_from_yaml."""
    t = cfg.training
    ff = cfg.model.get("future_feat", {}) or {}
    return {
        "time_mu": t.time_mu,
        "time_sigma": t.time_sigma,
        "use_vel_weight": t.use_vel_weight,
        "vel_weight_alpha": t.vel_weight_alpha,
        "vel_weight_sigma": t.vel_weight_sigma,
        "use_future_feat": bool(ff.get("enabled", False)),
        "lambda_future_feat": t.get("lambda_future_feat", 0.0),
    }


def build_model(cfg: DictConfig, norm_stats_path: str, device: str, dtype: torch.dtype,
                training: bool) -> BuiltModel:
    vision_encoder, hidden, n_reg, patch = ModelFactory.create_vision_encoder(
        cfg.model.vision_encoder.checkpoint_path, dtype, device)
    feat_layers = list(cfg.model.vision_encoder.feat_layers)
    use_tc = bool(cfg.model.get("use_task_cond", False))
    action_model = ModelFactory.create_action_model(
        cfg,
        dino_hidden_size=hidden,
        num_dino_layers=len(feat_layers),
        task_cond_dim=hidden if use_tc else None,
        patch_size=patch,
    )
    ff = cfg.model.get("future_feat", {}) or {}
    _apply_future_cond_tokens(action_model, ff.get("cond_tokens", None))
    wrapper = VLAWrapper(
        vision_encoder=vision_encoder,
        action_model=action_model,
        time_sampler=cfg.training.time_sampler,
        feat_layers=feat_layers,
        include_cls_register=cfg.model.vision_encoder.include_cls_register,
        num_register_tokens=n_reg,
        device=device,
        dtype=dtype,
        norm_stats_path=norm_stats_path,
        train_config=train_config_dict(cfg) if training else None,
        future_feat_target_layer=ff.get("target_layer", -1),
    )
    return BuiltModel(wrapper, action_model, hidden, n_reg, patch)


def _apply_future_cond_tokens(action_model, mode) -> None:
    """Which tokens feed the foresight decoder (config: model.future_feat.cond_tokens).

    Relies on the user's main-project option ``VLAModel.future_cond_tokens``
    ("all" = original behaviour: every token after the action tokens; "visual" = only
    the visual tokens). The user's ModelFactory already passes cfg.model.future_feat.cond_tokens;
    setting it again here is a guard that (a) fails loudly if the checkout lacks the option
    while "visual" is requested and (b) keeps working with an unmodified ModelFactory.
    """
    if mode not in ("all", "visual"):
        raise ValueError(f"model.future_feat.cond_tokens must be 'all' or 'visual', got {mode!r}")
    if not hasattr(action_model, "future_cond_tokens"):
        if mode == "all":
            return      # unmodified main project: "all" is its only (original) behaviour
        raise RuntimeError(
            "model.future_feat.cond_tokens='visual' needs VLAModel.future_cond_tokens, which this "
            "LiLa-WAM checkout does not have (models/vla_model_fm.py).")
    action_model.future_cond_tokens = mode
    logger.info(f"foresight decoder conditioned on {mode!r} tokens")


def load_vision_encoder(cfg: DictConfig, device: str, dtype: torch.dtype):
    """Frozen DINOv3 only (used by the VTT script)."""
    model, hidden, _, _ = ModelFactory.create_vision_encoder(
        cfg.model.vision_encoder.checkpoint_path, dtype, device)
    return model, hidden


def dit_blocks(action_model):
    """The DiT block list (used by the shape check to hook the token sequence)."""
    return action_model.blocks


def predict_velocity(action_model, t, noisy_actions, qpos_history, dino_features_list, task_cond):
    return action_model(
        t=t,
        noisy_actions=noisy_actions,
        qpos_history=qpos_history,
        dino_features_list=dino_features_list,
        task_cond=task_cond,
    )["final_pred"]


def checkpoint_payload(action_model, optimizer=None, scheduler=None, **meta) -> dict:
    """Checkpoint dict compatible with the main project's loaders ('model_state_dict')."""
    payload = {"model_state_dict": action_model.state_dict(), **meta}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload
