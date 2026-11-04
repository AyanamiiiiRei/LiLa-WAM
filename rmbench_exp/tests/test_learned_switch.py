"""Phase 3 (learned switch): soft write weights, the extra switch token, straight-through gradient."""

import copy

import pytest
import torch

from rmbench_lila.learned_switch import _make_plan, attach_learned_switch
from rmbench_lila.stage_memory import StageMemory


@pytest.fixture(scope="module")
def small_model():
    from rmbench_lila.paths import ensure_lila_importable
    ensure_lila_importable()
    from models.vla_model_fm import VLAModel
    torch.manual_seed(0)
    return VLAModel(action_dim=14, proprio_dim=16, hidden_dim=64, num_heads=4, depth=2, action_len=32,
                    num_registers=2, dino_feat_dims=(32, 32, 32), vlm_num_queries=8, adapter_depth=1,
                    task_cond_dim=32, fusion_mode="concat", use_future_feat=True, future_feat_num_queries=6,
                    future_feat_out_dim=32, future_cond_tokens="visual").eval()


def _inputs(B=2):
    g = torch.Generator().manual_seed(1)
    return dict(t=torch.rand(B, generator=g), noisy_actions=torch.randn(B, 32, 14, generator=g),
                qpos_history=torch.randn(B, 1, 16, generator=g),
                dino_features_list=[torch.randn(B, 20, 32, generator=g) for _ in range(3)],
                task_cond=torch.randn(B, 32, generator=g))


def test_float_written_equals_bool_at_0_and_1():
    sm = StageMemory(3, 32, 64)
    torch.nn.init.normal_(sm.mlp[-1].weight)             # non-zero content
    cls = torch.randn(2, 3, 2, 32)
    wb = torch.tensor([[True, False, True], [False, False, False]])
    assert torch.allclose(sm(cls, wb), sm(cls, wb.float()), atol=1e-6)


def test_switch_token_layout_and_prior(small_model):
    m = copy.deepcopy(small_model)
    attach_learned_switch(m, num_slots=3, cls_dim=32, prior_rate=0.08)
    x = _inputs()
    m.set_memory_input(torch.zeros(2, 3, 2, 32), torch.zeros(2, 3))
    out = m(**x)
    assert out["final_pred"].shape == (2, 32, 14)
    assert m._last_layout["seq_len"] == 44 + 3 + 1           # original + 3 memory + 1 switch token
    assert m._switch_feat.shape == (2, 64) and not m._switch_feat.requires_grad
    p = torch.sigmoid(m.switch_logit(torch.randn(2, 32), torch.tensor([0, 2])))
    assert torch.allclose(p, torch.full((2,), 0.08), atol=1e-4)  # zero-init head -> prior rate


def test_straight_through_reaches_the_head(small_model):
    m = copy.deepcopy(small_model)
    attach_learned_switch(m, 3, 32, 0.08)
    torch.nn.init.normal_(m.stage_memory.mlp[-1].weight)
    x = _inputs(1)
    m.set_memory_input(torch.zeros(1, 3, 2, 32), torch.zeros(1, 3))
    m(**x)                                                   # chunk c: decide
    p = torch.sigmoid(m.switch_logit(torch.randn(1, 32), torch.tensor([0])))
    s = torch.ones(1)                                        # sampled switch
    n_st = s + p - p.detach()
    w = (n_st[:, None] - torch.arange(3)[None]).clamp(0, 1)  # forward [1, 0, 0]
    assert torch.equal(w.detach(), torch.tensor([[1.0, 0.0, 0.0]]))
    m.set_memory_input(torch.randn(1, 3, 2, 32), w)          # chunk c+1 uses the write
    m(**x)["final_pred"].sum().backward()
    assert m.switch.head[-1].bias.grad is not None and m.switch.head[-1].bias.grad.abs().sum() > 0


def test_plan_uses_whole_demos():
    plan = _make_plan([600, 590, 621, 585], steps=5, per_step=2, seed=0)
    for eps, n in plan:
        assert len(set(eps)) == 2 and n == min((([600, 590, 621, 585][e] - 1) // 16 + 1) for e in eps)
