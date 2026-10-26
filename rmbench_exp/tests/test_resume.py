"""Resumable training pieces that need torch but no GPU/DINOv3."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rmbench_lila.trainer import ResumableRandomSampler, atomic_save  # noqa: E402


def test_sampler_resume_skips_exactly():
    s = ResumableRandomSampler(103, seed=7)
    s.set_epoch(2, 0)
    full = list(s)
    s.set_epoch(2, 40)
    assert list(s) == full[40:] and len(s) == 63
    s.set_epoch(3, 0)
    assert list(s) != full and sorted(list(s)) == list(range(103))


def test_atomic_save_leaves_no_tmp(tmp_path):
    p = tmp_path / "latest.pt"
    atomic_save({"a": 1}, p)
    assert torch.load(p)["a"] == 1 and not (tmp_path / "latest.pt.tmp").exists()
