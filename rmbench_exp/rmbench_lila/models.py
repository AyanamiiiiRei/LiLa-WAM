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
