"""Stage-memory experiment: checks from the plan's checklist that need no data or GPU.

  python -m pytest tests/test_stage_memory.py -q
"""

import random

import numpy as np
import pytest
import torch

from rmbench_lila.stage_memory import OnlineStageMemory, StageMemory, _forward_with_memory, attach_stage_memory
from rmbench_lila.stages import grid_stages, jitter_ends, slots_at, stage_index, swap_blocks_event_ends


# ------------------------------------------------------------------ boundaries
def test_grid_stages_round_up_and_chain():
    st = grid_stages([178, 375, 540, 620], 621)
    assert st == [[0, 192], [192, 384], [384, 544], [544, 608]]
    assert all(e % 16 == 0 for _, e in st)


def test_slots_never_use_future_stages():
    ends = [192, 384, 544]
    for t in range(0, 620):
        pairs, written = slots_at(t, ends, 3)
        for k in range(3):
            assert written[k] == (t >= ends[k])
            if written[k]:
                assert pairs[k][1] <= t                      # last frame of a written slot is not in the future
    pairs, _ = slots_at(600, ends, 3)
    assert pairs == [(0, 192), (192, 384), (384, 544)]       # first frame of slot n = end of slot n-1


def test_jitter_stays_ordered_and_inside():
    st = [[0, 192], [192, 384], [384, 544], [544, 608]]
    rng = random.Random(0)
    seen = set()
    for _ in range(2000):
        e = jitter_ends(st, 3, 621, p=0.5, rng=rng)
        assert e[0] >= 16 and all(a < b for a, b in zip(e, e[1:])) and e[-1] <= 608
        assert all(abs(a - b) in (0, 16) for a, b in zip(e, [192, 384, 544]))
        seen.add(tuple(e))
    assert len(seen) > 10
    assert jitter_ends(st, 3, 621, p=0.0) == [192, 384, 544]


def test_stage_index():
    st = [[0, 192], [192, 384], [384, 544], [544, 608]]
    assert [stage_index(t, st) for t in (0, 191, 192, 543, 544, 620)] == [0, 0, 1, 2, 3, 3]


def test_swap_blocks_event_ends_synthetic():
    T = 300
    L = np.zeros((T, 3)); R = np.zeros((T, 3)); gl = np.ones(T); gr = np.ones(T)
    for a, b, arm, g in ((10, 40, R, gr), (80, 110, R, gr), (150, 180, L, gl)):
        arm[a:b + 20] = 0.1                      # away from home until 20 frames after release
        g[a + 5:b] = 0.0                         # gripper closed while carrying
    ends = swap_blocks_event_ends(L, R, gl, gr)
    assert ends == [60, 130, 200]


# ------------------------------------------------------------------- model
@pytest.fixture(scope="module")
def small_model():
    from rmbench_lila.paths import ensure_lila_importable
    ensure_lila_importable()
    from models.vla_model_fm import VLAModel
    torch.manual_seed(0)
    m = VLAModel(action_dim=14, proprio_dim=16, hidden_dim=64, num_heads=4, depth=2, action_len=32,
                 num_registers=2, dino_feat_dims=(32, 32, 32), vlm_num_queries=8, adapter_depth=1,
                 task_cond_dim=32, fusion_mode="concat", use_future_feat=True, future_feat_num_queries=6,
                 future_feat_out_dim=32, future_cond_tokens="visual").eval()
    return m


def _inputs(B=2):
    g = torch.Generator().manual_seed(1)
    return dict(t=torch.rand(B, generator=g), noisy_actions=torch.randn(B, 32, 14, generator=g),
                qpos_history=torch.randn(B, 1, 16, generator=g),
                dino_features_list=[torch.randn(B, 20, 32, generator=g) for _ in range(3)],
                task_cond=torch.randn(B, 32, generator=g))


def test_copied_forward_equals_original(small_model):
    """Checklist 'equivalence': the sub-project's forward without memory == main-project forward."""
    x = _inputs()
    with torch.no_grad():
        a = small_model(**x)
        b = _forward_with_memory(small_model, **x, memory_tokens=None)
    assert torch.equal(a["final_pred"], b["final_pred"]) and torch.equal(a["cond_tokens"], b["cond_tokens"])


def test_memory_tokens_appended_and_visual_slice_unchanged(small_model):
    """Checklist 'index': sequence = original + N; action / VTT / visual positions unchanged;
    the foresight decoder ('visual') still gets exactly the visual tokens."""
    import copy
    m = copy.deepcopy(small_model)
    attach_stage_memory(m, num_slots=3, cls_dim=32)
    x = _inputs()
    with torch.no_grad():
        base = _forward_with_memory(small_model, **x, memory_tokens=None)
        lay0 = dict(small_model._last_layout)
        m.set_memory_input(torch.randn(2, 3, 2, 32), torch.tensor([[True, False, False]] * 2))
        out = m(**x)
        m.clear_memory_input()
    lay = m._last_layout
    assert lay0 == {"dino_start": 36, "dino_end": 44, "seq_len": 44}       # 32 act + 1 state + 2 reg + 1 VTT + 8 visual
    assert lay == {"dino_start": 36, "dino_end": 44, "seq_len": 47}
    assert out["cond_tokens"].shape == base["cond_tokens"].shape == (2, 8, 64)
    assert out["final_pred"].shape == (2, 32, 14)


def test_model_refuses_missing_memory(small_model):
    import copy
    m = copy.deepcopy(small_model)
    attach_stage_memory(m, 3, 32)
    with pytest.raises(RuntimeError):
        m(**_inputs())


def test_stage_memory_module_zero_init_and_written_mask():
    sm = StageMemory(3, 32, 64)
    cls = torch.randn(2, 3, 2, 32)
    w = torch.tensor([[True, True, False], [False, False, False]])
    tok = sm(cls, w)
    # at init the content MLP outputs 0: a written slot = slot_emb, an empty slot = empty + slot_emb
    assert torch.allclose(tok[0, 0], sm.slot_emb[0]) and torch.allclose(tok[0, 2], sm.empty + sm.slot_emb[2])
    assert torch.allclose(tok[1, 1], sm.empty + sm.slot_emb[1])


# ------------------------------------------------- train / inference memory logic
def test_online_memory_matches_training_slots():
    """Checklist 'inference consistency' (logic part): with stages_done = number of stage ends
    <= t, the online memory at every chunk start equals the training feature's slots."""
    T, ends = 621, [192, 384, 544]
    cls = np.random.default_rng(0).normal(size=(T, 8)).astype(np.float32)
    mem = OnlineStageMemory(3, 8, "stage")
    for t in range(0, T, 16):
        mem.update(cls[t], stages_done=sum(t >= e for e in ends))
        pairs, written = slots_at(t, ends, 3)
        assert (mem.written == written).all()
        for k in range(3):
            if written[k]:
                assert np.array_equal(mem.cls[k, 0], cls[pairs[k][0]]) and np.array_equal(mem.cls[k, 1], cls[pairs[k][1]])


def test_online_memory_first_frame_and_empty():
    cls = np.arange(40, dtype=np.float32).reshape(5, 8)
    f = OnlineStageMemory(1, 8, "first_frame")
    for i in range(5):
        f.update(cls[i], None)
    assert f.written.all() and np.array_equal(f.cls[0, 0], cls[0]) and np.array_equal(f.cls[0, 1], cls[0])
    e = OnlineStageMemory(3, 8, "stage", force_empty=True)
    for i in range(5):
        e.update(cls[i], stages_done=3)
    assert not e.written.any()
