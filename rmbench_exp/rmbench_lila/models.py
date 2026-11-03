"""Model variants selected by ``model.variant`` in the config (default: "lila_wam").

"lila_wam" is the unmodified main-project model. A variant for control experiment B2/B3
(extra history images) would be registered here as a new builder that subclasses /
wraps the main-project classes inside this sub-project — see docs/EXTENDING.md.
Trainer and policy only ever call ``build()``.
"""

from __future__ import annotations

from typing import Callable

from omegaconf import DictConfig

from .data.registry import Registry

MODELS = Registry("model variant")


def build(cfg: DictConfig, norm_stats_path: str, device: str, dtype, training: bool):
    name = cfg.model.get("variant", "lila_wam")
    builder: Callable = MODELS.get(name)
    return builder(cfg, norm_stats_path, device, dtype, training)


@MODELS.register("lila_wam")
def _build_lila_wam(cfg, norm_stats_path, device, dtype, training):
    from . import lila_api
    return lila_api.build_model(cfg, norm_stats_path, device, dtype, training)


@MODELS.register("lila_wam_stage_memory")
def _build_lila_wam_stage_memory(cfg, norm_stats_path, device, dtype, training):
    """lila_wam + N stage-memory tokens (docs/STAGE_MEMORY.md). Config: model.stage_memory."""
    from . import lila_api
    from .stage_memory import attach_stage_memory, attach_wrapper
    sm = cfg.model.get("stage_memory", None)
    if not sm or not sm.get("enabled", False):
        raise ValueError("variant lila_wam_stage_memory needs model.stage_memory.enabled: true "
                         "(use variant lila_wam for the original model)")
    built = lila_api.build_model(cfg, norm_stats_path, device, dtype, training)
    attach_stage_memory(built.action_model, int(sm.num_slots), built.dino_hidden_size)
    attach_wrapper(built.wrapper, built.action_model)
    return built


def uses_stage_memory(cfg: DictConfig) -> bool:
    sm = cfg.model.get("stage_memory", None)
    return bool(sm and sm.get("enabled", False))


def load_init_weights(action_model, state_dict: dict, cfg: DictConfig) -> None:
    """Warm start. The original model loads strictly; a stage-memory model may only be
    missing its new `stage_memory.*` parameters (trained from scratch, as the plan says)."""
    if not uses_stage_memory(cfg):
        action_model.load_state_dict(state_dict, strict=True)
        return
    res = action_model.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith("stage_memory.")]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(f"warm start mismatch: missing {bad_missing[:5]}..., unexpected {res.unexpected_keys[:5]}...")
