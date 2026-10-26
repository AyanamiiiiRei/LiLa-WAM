"""Config loading with inheritance, CLI overrides and consistency checks.

A config file may name a parent with ``_base_`` (string or list, relative to the
file). Parents are merged first, children override. Experiment configs in
configs/exp/ therefore only contain what differs from configs/base.yaml.

Resolvers available inside YAML:
    ${lila_root:}      LiLa-WAM main project root
    ${subproject:}     this sub-project root (rmbench_exp/)
    ${oc.env:VAR,default}   OmegaConf built-in environment lookup
Relative paths in the keys listed in PATH_KEYS are resolved against the sub-project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

from .paths import SUBPROJECT_ROOT, lila_root, resolve

PATH_KEYS = (
    "rmbench.data_root",
    "dataset.task_cond_dir",
    "model.vision_encoder.checkpoint_path",
    "paths.norm_stats",
    "experiment.output_root",
)

_RESOLVERS_REGISTERED = False


def _register_resolvers() -> None:
    global _RESOLVERS_REGISTERED
    if _RESOLVERS_REGISTERED:
        return
    # Some OmegaConf releases deprecate register_new_resolver in favour of register_resolver,
    # others the reverse; both behave the same for these zero-argument resolvers.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        OmegaConf.register_new_resolver("lila_root", lambda: str(lila_root()), replace=True)
        OmegaConf.register_new_resolver("subproject", lambda: str(SUBPROJECT_ROOT), replace=True)
    _RESOLVERS_REGISTERED = True


def _load_with_bases(path: Path, _seen: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular _base_ chain: {' -> '.join(map(str, _seen + (path,)))}")
    cfg = OmegaConf.load(path)
    bases = cfg.pop("_base_", None)
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    merged = OmegaConf.create()
    for b in bases:
        merged = OmegaConf.merge(merged, _load_with_bases(path.parent / b, _seen + (path,)))
    return OmegaConf.merge(merged, cfg)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> DictConfig:
    """Load a config, apply ``key=value`` overrides, resolve paths, validate."""
    _register_resolvers()
    cfg = _load_with_bases(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    for key in PATH_KEYS:
        val = OmegaConf.select(cfg, key)
        if val:
            OmegaConf.update(cfg, key, str(resolve(val)), merge=False)
    validate(cfg)
    return cfg


def validate(cfg: DictConfig) -> None:
    """Fail early on inconsistent settings instead of deep inside training."""
    from .schema import get_action_layout, get_state_layout, ACTION_ALIGNMENTS

    errs: list[str] = []
    s_layout = get_state_layout(cfg.rmbench.state_layout)
    a_layout = get_action_layout(cfg.rmbench.action_layout)
    if cfg.common.state_dim != s_layout.dim:
        errs.append(f"common.state_dim={cfg.common.state_dim} but state layout "
                    f"{s_layout.name!r} has dim {s_layout.dim}")
    if cfg.common.action_dim != a_layout.dim:
        errs.append(f"common.action_dim={cfg.common.action_dim} but action layout "
                    f"{a_layout.name!r} has dim {a_layout.dim}")
    if cfg.rmbench.action_alignment not in ACTION_ALIGNMENTS:
        errs.append(f"rmbench.action_alignment must be one of {sorted(ACTION_ALIGNMENTS)}")

    idx = cfg.dataset.indices_config
    if len(idx.action_indices) != cfg.common.action_chunk_size:
        errs.append("len(dataset.indices_config.action_indices) != common.action_chunk_size")
    if len(idx.state_indices) != cfg.common.proprio_len:
        errs.append("len(dataset.indices_config.state_indices) != common.proprio_len "
                    "(the model's proprio position embedding has proprio_len slots)")
    if cfg.common.action_execution_horizon > cfg.common.action_chunk_size:
        errs.append("common.action_execution_horizon > action_chunk_size")

    w, h = cfg.dataset.image_size
    if w % 16 or h % 16:
        errs.append(f"dataset.image_size {w}x{h}: both must be multiples of 16 (DINOv3 patch)")

    feats = list(cfg.dataset.features)
    ff_on = bool(OmegaConf.select(cfg, "model.future_feat.enabled", default=False))
    if ff_on and "future_image" not in feats:
        errs.append("model.future_feat.enabled=true requires 'future_image' in dataset.features")
    if not ff_on and "future_image" in feats:
        errs.append("'future_image' in dataset.features but model.future_feat.enabled=false "
                    "(wasted I/O; remove it)")
    ct = OmegaConf.select(cfg, "model.future_feat.cond_tokens", default="all")
    if ct not in ("all", "visual"):
        errs.append(f"model.future_feat.cond_tokens must be 'all' or 'visual' (got {ct!r}); "
                    "null breaks ModelFactory, which reads this key")
    tc_on = bool(cfg.model.get("use_task_cond", False))
    if tc_on != ("task_cond" in feats):
        errs.append("model.use_task_cond and the 'task_cond' dataset feature must agree")
    for need in ("state", "action_chunk", "image"):
        if need not in feats:
            errs.append(f"dataset.features must contain {need!r}")

    if errs:
        raise ValueError("Invalid config:\n  - " + "\n  - ".join(errs))


def save_config(cfg: DictConfig, path: str | Path) -> None:
    OmegaConf.save(cfg, str(path), resolve=True)
