"""Sub-task design: pre/post decision states, executor tokens, chooser masking."""

import copy

import numpy as np
import pytest
import torch

from rmbench_lila.subtask import Chooser, SubtaskModule, attach_subtask, state_at


def test_pre_post_states_and_entries():
    cls = np.arange(400, dtype=np.float32).reshape(200, 2)
    types, ends = [5, 6, 7], [48, 112]
    d, cur, mem, mask = state_at(48, types, ends, cls, 4, pre=True)
    assert (d, cur, mask.tolist()) == (0, 5, [True, False, False, False])     # before the decision at 48
    d, cur, mem, mask = state_at(48, types, ends, cls, 4, pre=False)
    assert (d, cur, mask.tolist()) == (1, 6, [True, True, False, False])      # after it
    assert np.array_equal(mem[1, 0], cls[0]) and np.array_equal(mem[1, 1], cls[48])
    d, cur, mem, mask = state_at(150, types, ends, cls, 4, pre=False)
    assert cur == 7 and np.array_equal(mem[2, 0], cls[48]) and np.array_equal(mem[2, 1], cls[112])


def test_memory_keeps_frame0_and_latest():
    cls = np.random.default_rng(0).normal(size=(400, 3)).astype(np.float32)
    types, ends = list(range(8)), [16 * (i + 1) for i in range(7)]
    _, _, mem, mask = state_at(399, types, ends, cls, 4, pre=False)
    assert mask.all() and np.array_equal(mem[0, 0], cls[0])
    assert np.array_equal(mem[3, 1], cls[ends[-1]])                           # newest entry last


def test_subtask_tokens_and_chooser_mask():
    m = SubtaskModule(n_types=5, num_mem=4, cls_dim=8, hidden=16)
    toks, w = m(torch.randn(2, 8), torch.tensor([0, 3]), torch.randn(2, 4, 2, 8),
                torch.tensor([[True, True, False, False], [True, False, False, False]]))
    assert toks.shape == (2, 2, 16) and torch.allclose(w[1], torch.tensor([1.0, 0, 0, 0]))
    ch = Chooser(hidden=16, cls_dim=8, n_types=5, d=32, layers=1, heads=4)
    tm = torch.tensor([[True, True, False, False, False]] * 2)
    end, nxt = ch(torch.randn(2, 10, 16), torch.randn(2, 8), torch.tensor([0, 1]), tm)
    assert end.shape == (2,) and (nxt.argmax(-1) < 2).all()


def test_attach_layout():
    from rmbench_lila.paths import ensure_lila_importable
    ensure_lila_importable()
    from models.vla_model_fm import VLAModel
    torch.manual_seed(0)
    m = VLAModel(action_dim=14, proprio_dim=16, hidden_dim=64, num_heads=4, depth=2, action_len=32,
                 num_registers=2, dino_feat_dims=(32, 32, 32), vlm_num_queries=8, adapter_depth=1,
                 task_cond_dim=32, fusion_mode="concat", use_future_feat=True, future_feat_num_queries=6,
                 future_feat_out_dim=32, future_cond_tokens="visual").eval()
    attach_subtask(m, n_types=5, num_mem=4, cls_dim=32, chooser_cfg={"d": 32, "layers": 1, "heads": 4})
    g = torch.Generator().manual_seed(1)
    x = dict(t=torch.rand(2, generator=g), noisy_actions=torch.randn(2, 32, 14, generator=g),
             qpos_history=torch.randn(2, 1, 16, generator=g),
             dino_features_list=[torch.randn(2, 20, 32, generator=g) for _ in range(3)],
             task_cond=torch.randn(2, 32, generator=g))
    with pytest.raises(RuntimeError):
        m(**x)
    m.set_subtask_input(torch.randn(2, 32), torch.tensor([1, 2]), torch.randn(2, 4, 2, 32),
                        torch.tensor([[True, False, False, False]] * 2))
    out = m(**x)
    assert m._last_layout == {"dino_start": 36, "dino_end": 44, "seq_len": 46}
    assert out["seq"].shape == (2, 46, 64) and out["cond_tokens"].shape == (2, 8, 64)
    end, nxt = m.chooser(out["seq"], torch.randn(2, 32), torch.tensor([1, 2]), torch.ones(2, 5, dtype=torch.bool))
    assert end.shape == (2,) and nxt.shape == (2, 5)
