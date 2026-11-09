"""Second pass over the failures that analysis/p2_failures.py left as "video" or labelled with a
rule that was too coarse. Uses only the existing closed-loop records (episodes.jsonl,
policy_view/*.npz, frame 0 of policy_view/*.mp4) and the sub-task library. CPU only.

  python analysis/p2_failures_detail.py runs/sub_P2-seed0/rmbench_eval/ckpt_85928 [--library artifacts/subtasks]

Per task (prints a short report; writes <ckpt_dir>/failure_review/detail.md):
  swap_blocks     completed moves (gripper releases off the button) before the chooser switched to
                  "press"; which token was active when the robot first pressed
  press_button    presses per button (ee z < 0.93 m; x < -0.075 left, < 0.075 middle, else confirm)
                  and the closed-loop duration of each sub-task vs the demo durations
  cover_blocks    required open order from the block colours in frame 0 (red, green, blue) vs the
                  chooser's "open" sequence
  place_block_mat put-back error: |x(release on the row) - x(grasp on the row)| of the same block
                  (mats ~11 cm apart: >= 5 cm = wrong mat)
  swap_T          rotation the held T needs (other arm's grasp yaw - own grasp yaw) vs rotation done
                  (release yaw - grasp yaw), and release position vs the other T's grasp position
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def rising(c):
    return [t for t in range(1, len(c)) if c[t] and not c[t - 1]]


def falling(c):
    return [t for t in range(1, len(c)) if c[t - 1] and not c[t]]


def token_fn(log, names, first):
    sw = [(c * 16, names[n]) for c, _, n, _ in log]
    return sw, (lambda t: ([names[first]] + [n for c, n in sw if c <= t])[-1])


def swap_blocks(eps, load, names, first, out):
    c = Counter()
    for e in eps:
        s = load(e)
        sw, tok = token_fn(e["subtask_log"] or [], names, first)
        rel, press = [], []
        for o, g in ((0, 7), (8, 15)):
            cl = s[:, g] < 0.5
            press += [t for t in rising(cl) if s[t, o + 1] < -0.24]
            rel += [t for t in falling(cl) if s[t, o + 1] >= -0.24]
        pt = next((t for t, n in sw if n == "press"), 10 ** 9)
        done = sum(t < pt for t in rel)
        c[(e["success"], min(done, 3), tok(min(press)) if press else "no press")] += 1
    out.append("swap_blocks: (success, moves completed before the chooser chose 'press', token at the robot's first press): count")
    out += [f"  {k}: {v}" for k, v in sorted(c.items(), key=str)]


def press_button(eps, load, names, first, seg, out):
    dur = defaultdict(list)
    for r in seg.values():
        for ty, a, b in r:
            dur[ty].append(b - a + 1)
    cnt = Counter()
    early = defaultdict(lambda: [0, 0])
    for e in eps:
        s = load(e)
        n = Counter()
        for o in (0, 8):
            for t in rising(s[:, o + 2] < 0.93):
                x = s[t, o]
                n["left" if x < -0.075 else "middle" if x < 0.075 else "confirm"] += 1
        cnt[(min(n["left"], 3), min(n["middle"], 3), min(n["confirm"], 3))] += 1
        log = e["subtask_log"] or []
        starts = [0] + [c * 16 for c, _, _, _ in log]
        types = [first] + [x[2] for x in log]
        for i in range(len(log)):
            early[types[i]][0] += 1
            early[types[i]][1] += (starts[i + 1] - starts[i]) < min(dur[types[i]])
    out.append("press_button: (#left, #middle, #confirm presses, capped at 3): episodes")
    out += [f"  {k}: {v}" for k, v in cnt.most_common()]
    for ty, (n, lo) in early.items():
        out.append(f"  '{names[ty]}' demo duration {min(dur[ty])}-{max(dur[ty])} steps; closed loop shorter than the demo minimum in {lo}/{n}")


def cover_blocks(eps, mp4, names, out):
    import cv2
    opens = {i: n.split()[2][0].upper() for i, n in enumerate(names) if n.startswith("Open the ")}
    res, first_ok = Counter(), Counter()
    for e in eps:
        cap = cv2.VideoCapture(str(mp4(e)))
        ok, f = cap.read()
        cap.release()
        if not ok:
            res["no video"] += 1
            continue
        h = f.shape[0]
        hsv = cv2.cvtColor(f[int(h * 0.55):int(h * 0.9)], cv2.COLOR_BGR2HSV).astype(int)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sat = (S > 120) & (V > 80)
        xs = [np.nonzero(m & sat)[1].mean() for m in ((H < 8) | (H > 170), (H > 40) & (H < 80), (H > 100) & (H < 130))]
        need = ["LMR"[i] for i in np.argsort(np.argsort(xs))]          # cover to open for red, green, blue
        got = [opens[x[2]] for x in (e["subtask_log"] or []) if x[2] in opens]
        k = 0
        while k < min(3, len(got)) and got[k] == need[k]:
            k += 1
        res["success" if e["success"] else ("order right" if k == 3 else f"wrong at open #{k + 1}" if len(got) > k else f"stopped after {len(got)}")] += 1
        first_ok[bool(got) and got[0] == need[0]] += 1
    out.append(f"cover_blocks: {dict(res)}; first open right {first_ok[True]}/{sum(first_ok.values())} (chance 1/3)")


def place_block_mat(eps, load, out):
    errs, per = [], Counter()
    for e in eps:
        s = load(e)
        ev = []
        for o, g in ((0, 7), (8, 15)):
            cl = s[:, g] < 0.5
            ev += [(t, "g", s[t, o], s[t, o + 1]) for t in rising(cl)] + [(t, "r", s[t, o], s[t, o + 1]) for t in falling(cl)]
        origin, ep = None, []
        for t, k, x, y in sorted(ev):
            if k == "g" and y > -0.15:
                origin = x
            if k == "r" and y > -0.15 and origin is not None:
                ep.append(abs(x - origin) * 100)
                origin = None
        errs += ep
        per["chooser never switched" if not e["subtask_log"] else "no put-back" if not ep else
            "a wrong mat (>= 5 cm)" if max(ep) >= 5 else "all put-backs < 5 cm"] += 1
    a = np.array(errs)
    out.append(f"place_block_mat: put-backs {len(a)}: <2 cm {np.mean(a < 2):.0%}, 2-5 cm {np.mean((a >= 2) & (a < 5)):.0%}, "
               f">=5 cm {np.mean(a >= 5):.0%}; episodes: {dict(per)}")


def swap_T(eps, load, out):
    from scipy.spatial.transform import Rotation as R
    ya = lambda q: np.degrees(R.from_quat(q).as_euler("zyx"))[1]   # noqa: E731  rotation about the tool axis
    rows = {True: [], False: []}
    miss = 0
    for e in eps:
        s = load(e)
        g = {a: (rising(s[:, k] < 0.5), falling(s[:, k] < 0.5)) for a, k in (("L", 7), ("R", 15))}
        (Lg, Lr), (Rg, Rr) = g["L"], g["R"]
        if not (Lg and Lr and Rg and Rr):
            miss += 1
            continue
        need = ya(s[Rg[0], 11:15]) - ya(s[Lg[0], 3:7])
        dL = ya(s[Lr[-1], 3:7]) - ya(s[Lg[-1], 3:7])
        dR = ya(s[Rr[-1], 11:15]) - ya(s[Rg[-1], 11:15])
        rows[e["success"]].append((abs(dL - need), abs(dR + need),
                                   np.hypot(*(s[Lr[-1], 0:2] - s[Rg[0], 8:10])) * 100,
                                   np.hypot(*(s[Rr[-1], 8:10] - s[Lg[0], 0:2])) * 100))
    for k, v in rows.items():
        if v:
            a = np.array(v)
            out.append(f"swap_T {'success' if k else 'failure'} n={len(a)}: median rotation error L {np.median(a[:, 0]):.0f} R {np.median(a[:, 1]):.0f} deg, "
                       f"median position error L {np.median(a[:, 2]):.1f} R {np.median(a[:, 3]):.1f} cm, rotation error > 15 deg in {np.mean(a[:, :2].max(1) > 15):.0%}")
    out.append(f"swap_T: {miss} episodes with an arm that never released")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--library", default="artifacts/subtasks")
    args = ap.parse_args()
    d = Path(args.ckpt_dir)
    z = np.load(Path(args.library) / "library.npz")
    names = [str(x) for x in z["names"]]
    seg = json.loads((Path(args.library) / "segments.json").read_text())
    first = {t: Counter(r[0][0] for r in eps.values() if r).most_common(1)[0][0] for t, eps in seg.items()}
    E = defaultdict(list)
    for line in open(d / "episodes.jsonl"):
        if line.strip():
            e = json.loads(line)
            if (d / "policy_view" / f"{e['task']}_seed{e['seed']}.npz").exists():
                E[e["task"]].append(e)
    load = lambda e: np.load(d / "policy_view" / f"{e['task']}_seed{e['seed']}.npz")["states"][: e["env_steps"]]  # noqa: E731
    mp4 = lambda e: d / "policy_view" / f"{e['task']}_seed{e['seed']}.mp4"  # noqa: E731
    fails = lambda t: [e for e in E[t] if not e["success"]]  # noqa: E731
    out = []
    swap_blocks(E["swap_blocks"], load, names, first["swap_blocks"], out)
    press_button(fails("press_button"), load, names, first["press_button"], seg["press_button"], out)
    cover_blocks(E["cover_blocks"], mp4, names, out)
    place_block_mat(fails("place_block_mat"), load, out)
    swap_T(E["swap_T"], load, out)
    text = "\n".join(out)
    print(text)
    (d / "failure_review").mkdir(exist_ok=True)
    (d / "failure_review" / "detail.md").write_text("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
