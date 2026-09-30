"""Compare two eval runs item by item from their eval_preds.jsonl (no model, no GPU), e.g. v1 against v0.

  uv run zhjudge compare runs/mmbert-base-zh-v1/eval_preds.jsonl v0/eval_preds.jsonl --label-a v1 --label-b v0 \
      [--out compare.md] [--allow-diff] [--t1]

Items are matched by id. For every slice both files have: the accuracy of each, the paired difference A - B with a 95%
interval (typed-decisions: the questions of one case resampled together), the Brier and NLL differences, the exact
McNemar p and both slice fingerprints. A slice whose items differ between the files (ids only one side scored, or the
same id with another gold or rendering) is not compared, and the command exits 1, unless --allow-diff (then the
differences cover the common items only). Rows written without their slices (v0's eval_preds.jsonl) get them from the
id; a typed-decisions workflow is not in the id, so its family slices take the items from the file that stored them.
--t1 compares the probabilities at temperature 1: each file's calibration is undone with the temperatures its eval
used (meta / baseline_meta of the eval.json next to it, or --temps-a / --temps-b: an eval.json or rl_agent_config.json).
The intervals cover item sampling only, not training-seed variance.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .baselines import HEADLINE
from .metrics import summarize
from .records import QTYPES, read_jsonl
from .stats import fingerprint, fmt_d, paired, score_table, slices_from_id


def slice_members(rows_a, rows_b, eval_only=frozenset()) -> dict:
    """{slice: (ids of A, ids of B)}. Stored slices win (A's over B's, per id), else they come from the id."""
    stored = {r["id"]: r["slices"] for rows in (rows_b, rows_a) for r in rows if r.get("slices")}
    out = defaultdict(lambda: (set(), set()))
    for j, rows in enumerate((rows_a, rows_b)):
        for r in rows:
            for s in stored.get(r["id"]) or slices_from_id(r["id"], r["qtype"], eval_only):
                out[s][j].add(r["id"])
    return out


def compare_rows(rows_a, rows_b, eval_only=frozenset(), allow_diff=False) -> dict:
    """{slice: paired A - B + fingerprints + status}; a slice with different items is only counted unless allow_diff."""
    ia, ib = {r["id"]: r for r in rows_a}, {r["id"]: r for r in rows_b}
    sa, sb = score_table(rows_a), score_table(rows_b)
    out = {}
    for s, (A, B) in slice_members(rows_a, rows_b, eval_only).items():
        if not A or not B:
            out[s] = {"n": 0, "only_a": len(A), "only_b": len(B), "status": "only in A" if A else "only in B"}
            continue
        ra, rb = [ia[i] for i in sorted(A)], [ib[i] for i in sorted(B)]
        p = paired(ra, rb, sa, sb)
        p.update(fingerprint_a=fingerprint(ra), fingerprint_b=fingerprint(rb),
                 ece15_a=summarize(ra, bootstrap=False)["ece15"], ece15_b=summarize(rb, bootstrap=False)["ece15"])
        p["status"] = "same items" if p["fingerprint_a"] == p["fingerprint_b"] else "items differ"
        if p["status"] != "same items" and not allow_diff:
            p = {k: p[k] for k in ("n", "only_a", "only_b", "gold_mismatch", "input_mismatch", "fingerprint_a",
                                   "fingerprint_b", "status")}
        out[s] = p
    return out


def slice_order(keys) -> list[str]:
    """Headline sets first, then the held-out aggregates, test:laya, probe families / types, and single sources."""
    head, test = [k for k, _ in HEADLINE], ("test:ALL", "test:group:", "test:lang:", "test:qtype:")

    def rank(k):
        if k in head:
            return 0, head.index(k), k
        if k.startswith("test:") and not k.startswith(("test:source:", "test:laya:")):
            return 1, next((i for i, p in enumerate(test) if k.startswith(p)), 9), k
        if k.startswith("test:laya:"):
            return 2, 0 if k == "test:laya:ALL" else 1, k
        if k.startswith("probe:"):
            return 3, 0, k
        return 4, 0, k
    return sorted(keys, key=rank)


def read_temps(path: Path, baseline=False):
    """(temperature, temperature_by_options) from an eval.json (meta, or baseline_meta) or an rl_agent_config.json."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    if "meta" in j:
        j = j["baseline_meta" if baseline else "meta"] or {}
    return j["temperature"], j.get("temperature_by_options") or {}


def untemper(rows, temps, by_opt):
    """Rows with the probabilities at T = 1: softmax(z / T) ** T renormalised = softmax(z)."""
    from .infer import temperature_for

    out = []
    for r in rows:
        p = np.asarray(r["probs"], float)
        q = p ** temperature_for(QTYPES[r["qtype"]], len(p), temps, by_opt)
        out.append({**r, "probs": (q / q.sum()).tolist() if q.sum() > 0 else r["probs"]})
    return out


def to_md(res, label_a, label_b, t1=False) -> str:
    same = sum(p["status"] == "same items" for p in res.values())
    diff = [k for k, p in res.items() if p["status"].startswith("items differ")]
    one = [k for k, p in res.items() if p["status"].startswith("only in")]
    L = [f"# {label_a} (A) vs {label_b} (B)" + (" at temperature 1" if t1 else ""), "",
         f"{same} of {len(res)} slices have exactly the same items (ids, golds, renderings) in both; {len(diff)} differ"
         + ("" if all("acc_a" in res[k] for k in diff) else " (not compared: rerun with --allow-diff to compare their "
            "common items)") + f"; {len(one)} are in one file only. Δ = A - B in points (Brier and NLL x100) on the "
         "common items, paired 95% interval (typed-decisions: resampled by case), * = the interval excludes 0. The "
         "intervals cover item sampling only, not training-seed variance.", "",
         "| slice | n | only A | only B | B acc | A acc | Δ acc | Δ Brier | Δ NLL | McNemar p | ECE-15 B / A | items |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in slice_order(res):
        p = res[k]
        if not p["n"]:
            continue
        cells = ["-"] * 7
        if "acc_a" in p:
            cells = [f"{100 * p['acc_b']:.1f}", f"{100 * p['acc_a']:.1f}", fmt_d(p), fmt_d(p, "d_brier"),
                     fmt_d(p, "d_nll"), "-" if p.get("mcnemar_p") is None else f"{p['mcnemar_p']:.3g}",
                     f"{p['ece15_b']:.3f} / {p['ece15_a']:.3f}"]
        L.append(f"| {k} | {p['n']} | {p['only_a']} | {p['only_b']} | " + " | ".join(cells) + f" | {p['status']} |")
    if one:
        L += ["", "In one file only: " + ", ".join(f"{k} ({res[k]['status'][8:]})" for k in slice_order(one))]
    return "\n".join(L) + "\n"


def main(argv=None):
    from .data.build import EVAL_ONLY

    ap = argparse.ArgumentParser(prog="zhjudge compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="eval_preds.jsonl of run A (e.g. v1)")
    ap.add_argument("b", help="eval_preds.jsonl of run B (e.g. v0)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--allow-diff", action="store_true", help="also compare slices whose items differ (common items)")
    ap.add_argument("--t1", action="store_true", help="undo each run's calibration temperatures first")
    ap.add_argument("--temps-a", default=None, help="eval.json / rl_agent_config.json with A's temperatures")
    ap.add_argument("--temps-b", default=None, help="eval.json / rl_agent_config.json with B's temperatures")
    ap.add_argument("--eval-only", default=",".join(sorted(EVAL_ONLY)),
                    help="sources counted as test:group:eval-only for rows without stored slices")
    ap.add_argument("--out", default=None, help="also write the table here (markdown) and <out>.json")
    a = ap.parse_args(argv)
    rows = []
    for path, temps in ((Path(a.a), a.temps_a), (Path(a.b), a.temps_b)):
        r = read_jsonl(path)
        if a.t1:
            src = Path(temps) if temps else path.parent / "eval.json"
            if not src.exists():
                raise SystemExit(f"--t1: no temperatures for {path} ({src} not found; pass --temps-a / --temps-b)")
            r = untemper(r, *read_temps(src, baseline="baseline" in path.name and not temps))
        rows.append(r)
    res = compare_rows(rows[0], rows[1], frozenset(x for x in a.eval_only.split(",") if x), a.allow_diff)
    md = to_md(res, a.label_a or Path(a.a).parent.name or a.a, a.label_b or Path(a.b).parent.name or a.b, a.t1)
    print(md)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if any(p["status"].startswith("items differ") and "acc_a" not in p for p in res.values()):
        print("[compare] some slices have different items in the two files: not compared (see --allow-diff)",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
