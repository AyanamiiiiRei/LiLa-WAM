#!/usr/bin/env python
"""Where along the memory path is the information lost? (sub-task design, plan 2; no training,
no closed loop; the model is NOT changed)

  python scripts/step41_memory_probe.py --run runs/sub_P2-seed0
      [--tasks cover_blocks put_back_block place_block_mat] [--inspect_h5 observe_and_pickup]

Question. At a decision that needs information seen only earlier, does the information reach
each stage of the path

  (1) memory entries (CLS pairs)  ->  (2) selected-memory token (SubtaskModule, chosen by the
  CURRENT sub-task)  ->  (3) DiT output at position 101  ->  (4) chooser hidden state (query
  position)  ->  (5) chooser decision

and is it USED (swap test)? Run on the demos (train + held-out) at the true decision states,
pre-decision inputs at flow time 0 exactly as in training (step 38).

Information label ("info") and decision label per task:
  cover_blocks     info = colour-to-cover permutation (6 classes; the demo's open order);
                   decision = the next cover to open, before each of the 3 opens   [chooser]
  put_back_block   info = decision = the mat the block came from (4; the demo's "move back" type)
                   before the switch to "move back"                                [chooser]
  place_block_mat  info = the mat the block now on the green mat came from (gripper-close x in
                   the demo's "move ... onto the green mat" segment, clustered to mats); at the
                   first chunk of each "put it back" sub-task, POST-decision input  [executor]

Stage (1) candidates (the content a memory entry could hold), from the frame(s) that carry the
information: frame 0 (cover, put_back) or (start, end) of the finished "move onto green"
segment (place): DINOv3 last-layer CLS (= what memory stores now), patch tokens of blocks
13/17/21/24 average-pooled to 2x2 and 4x4 grids, the executor's 64 visual tokens (adapter
output). Also "all entries as stored" (the CLS pairs the model actually holds).
Controls: current-frame CLS and current-frame 4x4 patch grid (block 21) -- information hidden
now must NOT be decodable from these, else the label is visible and the test is void;
shuffled labels (permuted, 10x, same pipeline) = the real chance level.

Probe: standardise -> PCA keeping all components of the training fold (lossless for an L2 linear
model; only for speed) -> L2 logistic regression, C chosen by inner CV; a negative result means "not linearly decodable from ~50 demos",
not "absent"; GroupKFold(5) by episode; accuracy with a 95% bootstrap CI over episodes.

Also: attention of the selected-memory token over the entries (which entry is read), and the
swap test (chooser tasks): replace the memory entries by those of another demo with the same
number of entries and a different label; does the decision follow the donor? Plus an ablation
where every stored CLS is replaced by the current-frame CLS (no past information).

Output: <run>/memory_probe/{report.md, results.json}
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import _common
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rmbench_lila import models
from rmbench_lila.data.dataset import RMBenchDataset, collate
from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.image import to_model_input
from rmbench_lila.policy import resolve_checkpoint
from rmbench_lila.schema import STATE_GROUP, get_state_layout
from rmbench_lila.subtask import Library

POS_SUB, POS_MEM = 100, 101          # sub-task token, selected-memory token in the DiT sequence
DINO_LAYERS = {"b13": -12, "b17": -8, "b21": -4, "b24": -1}


# ------------------------------------------------------------------ labels / decisions
def decisions_for(task, lib, refs, lengths, states):
    """-> list of dicts {e, t, k, info, decision, mem_frames, pre}"""
    names = lib.names
    out, info_names = [], None
    mats = None
    if task == "place_block_mat":
        xs = []
        for e, ref in enumerate(refs):
            for row in lib.segments[task][Path(ref.path).stem]:
                if "onto the green mat" in names[row[0]]:
                    xs.append(_grasp_x(states[e], row[1], row[2]))
        xs = np.array([x for x in xs if x is not None])
        mats = _cluster_1d(xs, 6)
        info_names = [f"mat x={c:+.3f}" for c in mats]
    for e, ref in enumerate(refs):
        stem = Path(ref.path).stem
        types, ends = lib.episode(task, stem, int(lengths[e]))
        rows = lib.segments[task][stem]
        if task == "cover_blocks":
            opens = [names[t].split()[2][0].upper() for t in types if names[t].startswith("Open the ")]
            if len(opens) < 3:
                continue
            perm = "".join(opens)                           # covers of red, green, blue
            k = 0
            for j, g in enumerate(ends):
                if names[types[j + 1]].startswith("Open the "):
                    out.append(dict(e=e, t=g, k=k, info=perm, decision=types[j + 1], mem_frames=(0,), pre=True))
                    k += 1
        elif task == "put_back_block":
            for j, g in enumerate(ends):
                n = names[types[j + 1]]
                if "back to the" in n:
                    out.append(dict(e=e, t=g, k=0, info=n.split("back to the ")[1].split(" mat")[0],
                                    decision=types[j + 1], mem_frames=(0,), pre=True))
        elif task == "place_block_mat":
            k = 0
            for j, g in enumerate(ends):
                if "put it back" in names[types[j + 1]] and "onto the green mat" in names[types[j]]:
                    a, b = rows[j][1], rows[j][2]
                    x = _grasp_x(states[e], a, b)
                    if x is None:
                        continue
                    m = int(np.argmin(np.abs(mats - x)))
                    out.append(dict(e=e, t=g, k=k, info=info_names[m], decision=types[j + 1],
                                    mem_frames=(a, b), pre=False))
                    k += 1
        else:
            raise ValueError(f"no label rule for {task}")
    return out


def _grasp_x(s, a, b):
    """x of the first gripper close (either arm, block on the row y > -0.15) inside frames [a, b]."""
    for t in range(max(a, 1), min(b + 1, len(s))):
        for o, g in ((0, 7), (8, 15)):
            if s[t, g] < 0.5 <= s[t - 1, g] and s[t, o + 1] > -0.15:
                return float(s[t, o])
    return None


def _cluster_1d(x, k):
    x = np.sort(x)
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(100):
        lab = np.argmin(np.abs(x[:, None] - c[None]), 1)
        c = np.array([x[lab == i].mean() if (lab == i).any() else c[i] for i in range(k)])
    return np.sort(c)


# ------------------------------------------------------------------ model passes
@torch.no_grad()
def chain(wrapper, am, batch, pre: bool, mem=None, mask=None, seed=0):
    """Stages (2)-(5) for a batch. mem/mask override the memory entries (swap / ablation)."""
    dev, dt = wrapper.device, wrapper.dtype
    sfx = "_pre" if pre else ""
    vtt = batch["sub_vtt" + sfx].to(dev, dt)
    typ = batch["sub_type" + sfx].to(dev)
    mem = (batch["sub_mem" + sfx] if mem is None else mem).to(dev, dt)
    mask = (batch["sub_mask" + sfx] if mask is None else mask).to(dev)
    feats = wrapper.get_vision_features(batch["pixel_values"])
    q = batch["state"].to(dev, dt)
    q = q.unsqueeze(1) if q.dim() == 2 else q
    tc = batch["task_cond"].to(dev, dt) if batch.get("task_cond") is not None else None
    g = torch.Generator(device=dev).manual_seed(seed)
    x0 = torch.randn(tuple(batch["action_sequence"].shape), device=dev, dtype=dt, generator=g)
    t0 = torch.zeros(len(x0), device=dev, dtype=dt)
    with torch.autocast("cuda", dtype=dt, enabled=str(dev).startswith("cuda")):
        toks, w = am.subtask(vtt, typ, mem, mask)
        am.set_subtask_input(vtt, typ, mem, mask)
        try:
            seq = am(t=t0, noisy_actions=x0, qpos_history=wrapper.normalize_state(q), dino_features_list=feats,
                     task_cond=tc)["seq"]
        finally:
            am.clear_subtask_input()
        ch = am.chooser
        cdt = ch.in_proj.weight.dtype
        x = ch.in_proj(seq.to(cdt)) + ch.seq_pos[:, : seq.shape[1]].to(cdt)
        x = torch.cat([ch.query.to(cdt).expand(len(seq), -1, -1), x,
                       ch.cls_proj(batch["sub_cls"].to(dev, cdt))[:, None], ch.type_emb(typ)[:, None]], 1)
        h = ch.enc(x)[:, 0]
        nxt = ch.next_head(h).float().masked_fill(~batch["sub_tmask"].to(dev), -1e4)
    f = lambda a: a.float().cpu().numpy()  # noqa: E731
    return {"s2_selected_mem": f(toks[:, 1]), "attn": f(w), "s3_dit_pos101": f(seq[:, POS_MEM]),
            "s3_dit_mean": f(seq.mean(1)), "s3_dit_action_mean": f(seq[:, :32].mean(1)),
            "s4_chooser_hidden": f(h), "next": f(nxt).argmax(-1),
            "stored_cls": f((mem * mask[..., None, None]).flatten(1))}


@torch.no_grad()
def image_candidates(wrapper, am, px):
    """Stage-(1) candidate features of frames px (B, 3, H, W)."""
    dev, dt = wrapper.device, wrapper.dtype
    enc = wrapper.vision_encoder
    out = enc(pixel_values=px.to(dev, dt), output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    nreg = int(getattr(enc.config, "num_register_tokens", 0))
    H, W = px.shape[-2] // 16, px.shape[-1] // 16
    res = {"cls": out.last_hidden_state[:, 0].float().cpu().numpy()}
    for name, li in DINO_LAYERS.items():
        p = hs[li][:, 1 + nreg:].float()
        p = p.reshape(len(p), H, W, -1).permute(0, 3, 1, 2)
        for gsz in (2, 4):
            res[f"patch_{name}_{gsz}x{gsz}"] = F.adaptive_avg_pool2d(p, gsz).flatten(1).cpu().numpy()
    feats = wrapper.get_vision_features(px)
    with torch.autocast("cuda", dtype=dt, enabled=str(dev).startswith("cuda")):
        tok = am.dino_adapters[0](am.concat_fusion(feats))
    res["adapter_64tok"] = tok.float().flatten(1).cpu().numpy()
    return res


# ------------------------------------------------------------------ probe
def probe(X, y, groups, seed=0, n_shuffle=10):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    classes, yi = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        return None

    X = np.asarray(X, np.float32)
    ng = len(np.unique(groups))
    chosen = []

    def cv_pred(labels, C=None):
        pred = np.full(len(labels), -1)
        for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, labels, groups):
            if len(np.unique(labels[tr])) < 2:
                pred[te] = labels[tr][0]
                continue
            # PCA keeping ALL components of the training fold (<= #samples) is lossless for an
            # L2 linear model (its weights lie in the span of the training samples) and makes it fast
            pipe = make_pipeline(StandardScaler(), PCA(n_components=min(len(tr), X.shape[1])),
                                 LogisticRegression(max_iter=3000, C=C or 1.0))
            if C is None:                    # inner CV for the regularisation strength
                inner = GroupKFold(n_splits=min(3, len(np.unique(groups[tr]))))
                try:
                    pipe = GridSearchCV(pipe, {"logisticregression__C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
                                        cv=list(inner.split(X[tr], labels[tr], groups[tr]))).fit(X[tr], labels[tr])
                    chosen.append(pipe.best_params_["logisticregression__C"])
                except ValueError:           # an inner fold with one class
                    pipe = make_pipeline(StandardScaler(), PCA(n_components=min(len(tr), X.shape[1])),
                                         LogisticRegression(max_iter=3000, C=1e-2)).fit(X[tr], labels[tr])
            else:
                pipe.fit(X[tr], labels[tr])
            pred[te] = pipe.predict(X[te])
        return pred

    pred = cv_pred(yi)
    ok = (pred == yi).astype(float)
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by_g = {g: ok[groups == g] for g in ug}
    boot = [np.concatenate([by_g[g] for g in rng.choice(ug, len(ug))]).mean() for _ in range(1000)]
    # shuffled labels: same pipeline (C fixed to the most chosen value), labels permuted -> real chance
    C = Counter(chosen).most_common(1)[0][0] if chosen else 1e-2
    sh = []
    for _ in range(n_shuffle):
        lab = rng.permutation(yi)
        sh.append((cv_pred(lab, C) == lab).mean())
    return {"acc": float(ok.mean()), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "majority": float(Counter(yi).most_common(1)[0][1] / len(yi)),
            "shuffled_mean": float(np.mean(sh)), "shuffled_p95": float(np.quantile(sh, 0.95)),
            "n": int(len(yi)), "episodes": int(len(ug)), "classes": int(len(classes)), "pred": pred, "y": yi}


# ------------------------------------------------------------------ main
def build(run, device, dtype):
    cfg = OmegaConf.load(run / "config.yaml")
    built = models.build(cfg, str(run / "norm_stats.json"), device, dtype, training=True)
    ck = torch.load(resolve_checkpoint(run, "latest"), map_location="cpu", weights_only=False)
    built.action_model.load_state_dict(ck["model_state_dict"], strict=True)
    built.wrapper.to(device, dtype)
    built.action_model.eval()
    if "chooser_steps" not in ck:
        print("WARNING: this checkpoint has no trained chooser (chooser_steps missing)")
    return cfg, built, ck


def fmt(r):
    if r is None:
        return "– | – | – | –"
    return (f"{r['acc']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | {r['shuffled_mean']:.2f} (p95 {r['shuffled_p95']:.2f}) | "
            f"{r['majority']:.2f} | {r['n']} / {r['episodes']}")


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--run", required=True)
    p.add_argument("--tasks", nargs="+", default=["cover_blocks", "put_back_block", "place_block_mat"])
    p.add_argument("--inspect_h5", default=None, help="print the HDF5 keys of one demo of this task and exit")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run = Path(args.run)
    cfg0 = OmegaConf.load(run / "config.yaml")

    if args.inspect_h5:
        import h5py
        OmegaConf.update(cfg0, "rmbench.tasks", [args.inspect_h5], merge=False)
        ref = select_episodes(cfg0, "all")[0]
        with h5py.File(ref.path, "r") as f:
            f.visititems(lambda n, o: print(n, getattr(o, "shape", ""), getattr(o, "dtype", "")))
            for k, v in f.attrs.items():
                print("attr", k, v)
        return

    device, dtype = _common.pick_device_dtype(args.device)
    cfg, built, ck = build(run, device, dtype)
    wrapper, am = built.wrapper, built.action_model
    lib = Library(cfg.dataset.subtask.library_dir)
    layout = get_state_layout(cfg.rmbench.state_layout)
    cam = list(cfg.dataset.camera_names)[0]
    size = tuple(cfg.dataset.image_size)
    outdir = run / "memory_probe"
    outdir.mkdir(exist_ok=True)
    results, lines = {}, [f"# Memory probe ({run.name}, step {ck.get('global_step')}, chooser steps {ck.get('chooser_steps')})", ""]

    for task in args.tasks:
        tcfg = copy.deepcopy(cfg)
        OmegaConf.update(tcfg, "rmbench.tasks", [task], merge=False)
        OmegaConf.update(tcfg, "dataset.anchor_stride", 16, merge=True)
        refs = select_episodes(tcfg, "all")
        held = {r.episode_id for r in select_episodes(tcfg, "holdout")}
        ds = RMBenchDataset(tcfg, "all", episodes=refs)
        states = []
        for ref in refs:
            with EpisodeReader(ref.path) as r:
                states.append(r.read_vector(layout, STATE_GROUP))
        dec = decisions_for(task, lib, refs, ds.lengths, states)
        ds.index = [(d["e"], d["t"]) for d in dec]
        print(f"[{task}] {len(refs)} demos, {len(dec)} decisions", flush=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate,
                                             num_workers=int(cfg.system.num_workers))
        pre = dec[0]["pre"]
        acc = defaultdict(list)
        batches = []
        for b in loader:
            r = chain(wrapper, am, b, pre)
            for k, v in r.items():
                acc[k].append(v)
            acc["cur_cls"].append(b["sub_cls"].float().numpy())
            batches.append({k: b[k] for k in ("sub_mem_pre", "sub_mask_pre", "sub_mem", "sub_mask", "sub_cls")})
        S = {k: np.concatenate(v) for k, v in acc.items()}
        print(f"[{task}] model passes done, extracting stage-(1) candidates", flush=True)

        # stage (1) candidates on the information frames, and current-frame controls
        cand = defaultdict(list)
        cache = {}
        for d in dec:
            ref = refs[d["e"]]
            parts = []
            for fr in d["mem_frames"]:
                key = (d["e"], fr)
                if key not in cache:
                    with EpisodeReader(ref.path) as r:
                        img = r.read_frames(cam, [fr])[0]
                    cache[key] = image_candidates(wrapper, am, torch.from_numpy(to_model_input(img, size))[None])
                parts.append(cache[key])
            for k in parts[0]:
                cand[k].append(np.concatenate([q[k][0] for q in parts]))
            with EpisodeReader(ref.path) as r:
                cur = image_candidates(wrapper, am, torch.from_numpy(to_model_input(r.read_frames(cam, [d["t"]])[0], size))[None])
            cand["CONTROL_current_patch_b21_4x4"].append(cur["patch_b21_4x4"][0])
        C = {k: np.stack(v) for k, v in cand.items()}
        print(f"[{task}] candidates done, probing", flush=True)

        groups = np.array([d["e"] for d in dec])
        info = np.array([d["info"] for d in dec])
        decision = np.array([d["decision"] for d in dec])
        k_idx = np.array([d["k"] for d in dec])
        T = {"stages": {}, "candidates": {}}
        stage_feats = {"(1) stored entries (CLS pairs, as held)": S["stored_cls"],
                       "(2) selected-memory token": S["s2_selected_mem"],
                       "(3) DiT output, position 101": S["s3_dit_pos101"],
                       "(3) DiT output, mean of 102": S["s3_dit_mean"],
                       "(3) DiT output, mean of action tokens": S["s3_dit_action_mean"],
                       "CONTROL current-frame CLS": S["cur_cls"],
                       "CONTROL current-frame patch b21 4x4": C["CONTROL_current_patch_b21_4x4"]}
        if pre:
            stage_feats["(4) chooser hidden"] = S["s4_chooser_hidden"]
        lines += [f"## {task}", "", f"demos {len(refs)} (held-out ids {sorted(held)}), decisions {len(dec)}; "
                  f"info classes {dict(Counter(info))}", ""]
        lines += ["### Stages: is the information decodable? (label = info)", "",
                  "| features | accuracy [95% CI] | shuffled mean (p95) | majority | samples / episodes |", "|---|---|---|---|---|"]
        for name, X in stage_feats.items():
            t0 = time.time()
            r = probe(X, info, groups)
            print(f"  [{task}] {name}: {fmt(r)}  ({time.time() - t0:.0f} s)", flush=True)
            T["stages"][name] = r
            lines.append(f"| {name} | {fmt(r)} |")
        if pre and task == "cover_blocks":
            lines += ["", "decision label (next cover) from the chooser hidden state, by open #:", ""]
            r = probe(S["s4_chooser_hidden"], decision, groups)
            T["stages"]["(4) chooser hidden -> next cover"] = r
            lines.append(f"| (4) chooser hidden -> next cover | {fmt(r)} |")
        lines += ["", "### Stage (1) candidates: what a memory entry could hold (label = info)", "",
                  "| candidate | accuracy [95% CI] | shuffled mean (p95) | majority | samples / episodes |", "|---|---|---|---|---|"]
        uniq = np.unique(groups, return_index=True)[1] if task != "place_block_mat" else np.arange(len(dec))
        for name, X in C.items():
            if name.startswith("CONTROL"):
                continue
            t0 = time.time()
            r = probe(X[uniq], info[uniq], groups[uniq])
            print(f"  [{task}] candidate {name}: {fmt(r)}  ({time.time() - t0:.0f} s)", flush=True)
            T["candidates"][name] = r
            lines.append(f"| {name} | {fmt(r)} |")

        # stage (5) and swap test (chooser tasks)
        if pre:
            ok = S["next"] == decision
            lines += ["", "### (5) Chooser decision on the demos", "",
                      "| open # / decision | correct | held-out correct |", "|---|---|---|"]
            for k in sorted(set(k_idx)):
                m = k_idx == k
                mh = m & np.isin([refs[e].episode_id for e in groups], list(held))
                lines.append(f"| {k + 1} | {ok[m].sum()}/{m.sum()} | {ok[mh].sum()}/{mh.sum()} |")
            T["decision_correct"] = {int(k): [int(ok[k_idx == k].sum()), int((k_idx == k).sum())] for k in set(k_idx)}

            mem_pre = torch.cat([b["sub_mem_pre"] for b in batches])
            mask_pre = torch.cat([b["sub_mask_pre"] for b in batches])
            cur_cls = torch.cat([b["sub_cls"] for b in batches])
            rng = np.random.default_rng(0)
            donor = np.full(len(dec), -1)
            for i, d in enumerate(dec):
                cands = [j for j, dj in enumerate(dec) if dj["k"] == d["k"] and dj["e"] != d["e"]
                         and decision[j] != decision[i] and mask_pre[j].sum() == mask_pre[i].sum()]
                if cands:
                    donor[i] = rng.choice(cands)
            sw = {"follows donor": 0, "keeps own": 0, "other": 0}
            abl_same = 0
            for s0 in range(0, len(dec), args.batch):
                idx = np.arange(s0, min(len(dec), s0 + args.batch))
                b = collate([ds[i] for i in idx])
                have = donor[idx] >= 0
                if have.any():
                    mem_sw = mem_pre[idx].clone()
                    mem_sw[have] = mem_pre[donor[idx][have]]
                    r = chain(wrapper, am, b, True, mem=mem_sw, mask=mask_pre[idx])
                    for i, nx in zip(idx[have], r["next"][have]):
                        sw["follows donor" if nx == decision[donor[i]] else "keeps own" if nx == decision[i] else "other"] += 1
                abl = cur_cls[idx][:, None, None, :].expand_as(mem_pre[idx]).clone()
                r = chain(wrapper, am, b, True, mem=abl, mask=mask_pre[idx])
                abl_same += int((r["next"] == S["next"][idx]).sum())
            lines += ["", "### Swap test: memory entries replaced by another demo's (same #entries, different decision)", "",
                      f"follows donor {sw['follows donor']}, keeps own {sw['keeps own']}, other {sw['other']} "
                      f"(of {sum(sw.values())}). Ablation (every stored CLS = current-frame CLS): decision unchanged "
                      f"in {abl_same}/{len(dec)}.", ""]
            T["swap"] = sw
            T["ablation_unchanged"] = [abl_same, len(dec)]

        # attention of the selected-memory token
        W = S["attn"]
        lines += ["### Attention of the selected-memory token over the entries (mean weight)", "",
                  "| decision # | " + " | ".join(f"entry {i}" for i in range(W.shape[1])) + " |",
                  "|---" * (W.shape[1] + 1) + "|"]
        for k in sorted(set(k_idx)):
            m = W[k_idx == k].mean(0)
            lines.append(f"| {k + 1} | " + " | ".join(f"{v:.2f}" for v in m) + " |")
        lines.append("")
        T["attention"] = {int(k): W[k_idx == k].mean(0).tolist() for k in set(k_idx)}
        for part in ("stages", "candidates"):
            for r in T[part].values():
                if r:
                    r.pop("pred", None), r.pop("y", None)
        results[task] = T
        (outdir / "report.md").write_text("\n".join(lines) + "\n")
        (outdir / "results.json").write_text(json.dumps(results, indent=1, default=str))
        print("\n".join(lines[-40:]), flush=True)
    print(f"written {outdir}/report.md, results.json")


if __name__ == "__main__":
    main()
