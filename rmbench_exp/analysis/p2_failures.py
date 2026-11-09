"""Classify the failed closed-loop episodes of a sub-task run (plan 2) like round 1, with the
sub-task chooser's log added. CPU only; no training, no new evaluation.

  python analysis/p2_failures.py runs/sub_P2-seed0/rmbench_eval/ckpt_85928 [--per_task 8]

Categories (fixed):
  memory      the chosen sub-task / target contradicts information seen only earlier
              (e.g. put_back_block: chose to return to a mat other than the block's own)
  chooser     sub-task judgement: never left a sub-task, left it before it was done,
              or switched in a way unrelated to memory
  execution   the sub-task was chosen correctly but the executor did not do it
              (went elsewhere, did not move, did a different sub-task)
  precision   right sub-task, right motion, but not precise enough (shallow press, dropped,
              placed off target)
  video       the recorded states cannot decide -> contact sheet written for manual review
Rules use only the recorded ee16 states (policy_view/*.npz), the chooser log and the sub-task
library names. Output in <ckpt_dir>/failure_review/: failures.csv, summary.md,
sheets/<task>_seed<seed>.png (12 frames from the policy-view video with step and sub-task).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PBB_MATS = {"left": (0.0, -0.1), "right": (0.2, -0.1), "front": (0.1, -0.2), "back": (0.1, 0.0)}
PBB_CENTRE = (0.1, -0.1)
PBB_DEMO_PRESS_Z = 0.926


def edges(mask):
    return [t for t in range(1, len(mask)) if mask[t] and not mask[t - 1]]


def falling(mask):
    return [t for t in range(1, len(mask)) if not mask[t] and mask[t - 1]]


def nearest(p, places, tol=0.04):
    k = min(places, key=lambda k: np.hypot(p[0] - places[k][0], p[1] - places[k][1]))
    d = np.hypot(p[0] - places[k][0], p[1] - places[k][1])
    return (k if d < tol else "other"), d


def type_at(log, first, chunk):
    cur = first
    for c, _, new, _ in log:
        if c < chunk:
            cur = new
    return cur


def classify(task, e, s, names, first_type):
    """-> (label, reason)"""
    log = e.get("subtask_log") or []
    seq = [names[first_type]] + [names[x[2]] for x in log]
    sw = [x[0] for x in log]
    timeout = e["env_steps"] >= e.get("_max_steps", 10 ** 9)
    if task == "put_back_block":
        closed = s[:, 15] < 0.5
        g, r = edges(closed), falling(closed)
        own = nearest(s[g[0], 8:10], dict(PBB_MATS, centre=PBB_CENTRE))[0] if g else None
        chosen = [n.split("back to the ")[1].split(" mat")[0] for n in seq if "back to the" in n]
        if not chosen:
            return "chooser", "never switched to a 'move back' sub-task"
        if own in PBB_MATS and chosen[-1] != own:
            return "memory", f"chose '{chosen[-1]}' mat, block came from '{own}'"
        if not r:
            return "execution", "never released the block"
        last, d = nearest(s[r[-1], 8:10], dict(PBB_MATS, centre=PBB_CENTRE), tol=0.04)
        if last == own:
            if float(s[:, 2].min()) > PBB_DEMO_PRESS_Z + 0.003:
                return "precision", f"returned to own mat, press too shallow (min z {s[:, 2].min():.3f})"
            return "video", "returned to own mat and pressed, still failed"
        tgt = PBB_MATS.get(chosen[-1])
        if tgt is not None and np.hypot(s[r[-1], 8] - tgt[0], s[r[-1], 9] - tgt[1]) < 0.07:
            return "precision", f"released {np.hypot(s[r[-1], 8] - tgt[0], s[r[-1], 9] - tgt[1]) * 100:.1f} cm from the chosen mat"
        return "execution", f"chose '{chosen[-1]}' but released at '{last}'"
    if task == "observe_and_pickup":
        dL = np.linalg.norm(s[:, 0:3] - s[0, 0:3], axis=1)
        dR = np.linalg.norm(s[:, 8:11] - s[0, 8:11], axis=1)
        moved = np.where((dL > 0.03) | (dR > 0.03))[0]
        if not len(moved):
            return "execution", "never moved"
        if moved[0] < 20:
            if sw and sw[0] * 16 <= moved[0]:
                return "chooser", f"left 'wait' at chunk {sw[0]}, moved at step {moved[0]} (< 20: auto-fail)"
            return "execution", f"moved at step {moved[0]} while still in 'wait' (< 20: auto-fail)"
        return "video", "picked something: wrong object (memory) or grasp failed (precision)?"
    if task == "blocks_ranking_try":
        if not log:
            return "chooser", "never left the first 'press the button'"
        press_lengths = [b[0] - a[0] for a, b in zip(log, log[1:]) if names[a[2]].startswith("Press")]
        if press_lengths and max(press_lengths) <= 1:
            return "chooser", f"every later 'press' left after {max(press_lengths)} chunk (demos ~3)"
        return "video", f"{len(log)} switches"
    if task == "swap_blocks":
        ev = []
        for o, gi in ((0, 7), (8, 15)):
            for t in edges(s[:, gi] < 0.5):
                ev.append((t, "press" if s[t, o + 1] < -0.24 else "move"))
        ev.sort()
        moves_before_press = next((i for i, (_, k) in enumerate(ev) if k == "press"), len(ev))
        ok_seq = [names[x] for x in [first_type] + [x[2] for x in log]][:4] == ["move 1", "move 2", "move 3", "press"]
        if not ok_seq:
            return "chooser", "sub-task sequence " + " > ".join(seq[:6])
        if moves_before_press == 3:
            return "video", "3 moves + press with the right sub-task sequence, still failed"
        # where did the executor deviate from its sub-task token?
        return "execution", f"chooser sequence correct, robot pressed after {moves_before_press} moves"
    if task == "rearrange_blocks":
        closed = s[:, 15] < 0.5
        g = [(t, float(s[t, 8]), float(s[t, 9])) for t in edges(closed)]
        r = [(t, float(s[t, 8]), float(s[t, 9])) for t in falling(closed)]
        if len(g) < 2 or not r:
            return "video", "no second pick"
        if np.hypot(g[1][1] - r[0][1], g[1][2] - r[0][2]) < 0.05:
            return "memory", f"picked the block it had just placed (chose '{seq[2] if len(seq) > 2 else '?'}')"
        z = s[: g[1][0], 2]
        dips = sum(1 for t in range(1, len(z)) if z[t] < 0.935 <= z[t - 1])
        if dips > 1:
            return "chooser", f"pressed the button {dips} times (press_cnt > 1)"
        if dips == 0:
            return "precision", "press too shallow"
        return "video", "right block, one press, still failed"
    return "video", " > ".join(seq[:8])


def contact_sheet(mp4: Path, out: Path, steps: int, log, first, names):
    import cv2
    idx = np.linspace(0, max(steps - 1, 0), 12).astype(int)
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            f = np.zeros((240, 320, 3), np.uint8)
        f = cv2.resize(f, (320, 240))
        cur = names[type_at(log, first, i // 16)]
        cv2.rectangle(f, (0, 0), (320, 34), (0, 0, 0), -1)
        cv2.putText(f, f"step {i}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(f, cur[:52], (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 255, 120), 1)
        frames.append(f)
    cap.release()
    rows = [np.hstack(frames[k:k + 4]) for k in range(0, 12, 4)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--library", default="artifacts/subtasks")
    ap.add_argument("--per_task", type=int, default=8)
    args = ap.parse_args()
    d = Path(args.ckpt_dir)
    z = np.load(Path(args.library) / "library.npz")
    names = [str(x) for x in z["names"]]
    lib_task = [str(x) for x in z["task"]]
    seg = json.loads((Path(args.library) / "segments.json").read_text())
    first_type = {}
    for t, eps in seg.items():
        f = [row[0][0] for row in eps.values() if row]
        first_type[t] = max(set(f), key=f.count)
    eps = {}
    for line in open(d / "episodes.jsonl"):
        if line.strip():
            e = json.loads(line)
            eps[(e["task"], e["seed"])] = e
    max_steps = defaultdict(int)
    for (t, _), e in eps.items():
        max_steps[t] = max(max_steps[t], e["env_steps"])
    out = d / "failure_review"
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    rows, per = [], defaultdict(Counter)
    video_left = defaultdict(int)
    rng = np.random.default_rng(0)
    order = list(eps.items())
    rng.shuffle(order)
    for (task, seed), e in sorted(order, key=lambda kv: kv[0]):
        if e["success"]:
            per[task]["success"] += 1
            continue
        f = d / "policy_view" / f"{task}_seed{seed}.npz"
        if not f.exists():
            per[task]["no record"] += 1
            continue
        s = np.load(f)["states"][: e["env_steps"]]
        e["_max_steps"] = max_steps[task]
        label, reason = classify(task, e, s, names, first_type[task])
        per[task][label] += 1
        seq = [names[first_type[task]]] + [names[x[2]] for x in (e.get("subtask_log") or [])]
        sheet = ""
        mp4 = f.with_suffix(".mp4")
        if label == "video" and video_left[task] < args.per_task and mp4.exists():
            sheet = f"sheets/{task}_seed{seed}.png"
            try:
                contact_sheet(mp4, out / sheet, e["env_steps"], e.get("subtask_log") or [], first_type[task], names)
                video_left[task] += 1
            except Exception as ex:      # keep going without the sheet
                sheet = f"(sheet failed: {ex})"
        rows.append({"task": task, "seed": seed, "env_steps": e["env_steps"], "timeout": e["env_steps"] >= max_steps[task],
                     "switches": len(e.get("subtask_log") or []), "label": label, "reason": reason,
                     "subtasks": " > ".join(seq[:12]), "sheet": sheet})
    with open(out / "failures.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cats = ["success", "memory", "chooser", "execution", "precision", "video", "no record"]
    lines = ["| task | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for t in sorted(per):
        lines.append(f"| {t} | " + " | ".join(str(per[t][c]) for c in cats) + " |")
    tot = Counter()
    for c in per.values():
        tot.update(c)
    lines.append("| ALL | " + " | ".join(str(tot[c]) for c in cats) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out}/failures.csv, summary.md, {sum(video_left.values())} contact sheets in {out}/sheets")


if __name__ == "__main__":
    main()
