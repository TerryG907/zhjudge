"""Independent checks on data/unified/ (does not trust build's bookkeeping). Ported from the spike.

  uv run zhjudge validate --config ...   -> prints a report, writes data/unified/validation.json; exit 1 on any hard failure

Hard checks:
  1. every line parses and matches the schema (types, label range, options unique, noul has bool label and no options,
     a view, when present, is well formed for the question type)
  2. SHA256SUMS matches the files on disk
  3. no record id is duplicated
  4. no normalised text unit (>= 8 chars) of any TRAIN record occurs in any DEV/TEST record (all sources pooled), and
     none (>= 12 chars) occurs inside the text of an eval-only source's record
  5. no DEV/TEST record shares a text unit with any item of the probe sets (data/probes/*.jsonl)
  6. no TRAIN record shares a text unit with the probe items
  7. every source on disk is enabled in the allowlist and allowed by the licence audit under data.license_profile
     (unless the manifest says smoke override)
  8. no instruction the training-time renderings can produce (zhjudge.data.views) equals a probe instruction, even up
     to the field names, and no description or level text they add to v1's (format variants, the v1b data extras;
     only what this config turns on) equals a probe criterion text (label names shared with the probes are listed)
  9. no id repeats among the decisions `zhjudge eval` will score (same selection: zhjudge.eval.eval_records; eval matches
     rows to records by id), so an id problem stops the run here rather than after training
Warning only (eval_set_vs_v0): per-source dev/test counts that differ from v0's (eval.reference,
configs/eval_reference_v0.json) and sources with dev/test records dropped for overlapping the probes; either means
test:* of that source is not item-identical to v0. The owner may change an eval set on purpose, so it never fails.
Soft stats: per lang/split/type counts, label balance, context length percentiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict

from ..config import add_common_args, load_config, paths, repo_root
from .allowlist import check_licence, is_enabled, load_allowlist, load_licenses
from .common import ContainmentIndex, probe_units, record_units
from .views import check as check_view, probe_instruction_collisions, probe_option_collisions, tasks_for


def eval_set_vs_reference(cfg: dict, manifest: dict, by_src: Counter) -> dict:
    """Per-source dev/test counts on disk vs the reference run's (v0) and the sources that dropped dev/test records for
    overlapping the probes. Same counts and no such drop: the held-out files hold the records v0 had (every split is
    deterministic), so the capped test sample is v0's too; eval.json's fingerprints and `zhjudge compare` check items."""
    ref_path = repo_root() / (cfg["eval"].get("reference") or "configs/eval_reference_v0.json")
    if not ref_path.exists():
        return {"reference": None, "same_as_v0": None, "note": f"{ref_path.name} not found: nothing to compare with"}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    diff = {}
    for s in sorted(set(ref) | set(manifest["sources"])):
        for sp in ("dev", "test"):
            a, b = ref.get(s, {}).get(sp, 0), by_src.get((s, sp), 0)
            if a != b:
                diff.setdefault(s, {})[sp] = {"v0": a, "now": b}
    drop = {s: v["dropped_eval_overlapping_probes"] for s, v in sorted(manifest["sources"].items())
            if v.get("dropped_eval_overlapping_probes")}
    name = str(ref_path.relative_to(repo_root())) if ref_path.is_relative_to(repo_root()) else ref_path.name
    return {"reference": name, "same_as_v0": not diff and not drop, "count_differences": diff,
            "dropped_eval_overlapping_probes": drop}


def eval_set_lines(v: dict) -> list[str]:
    if v.get("same_as_v0") is None:
        return [f"[validate] eval sets vs v0: {v['note']}"]
    if v["same_as_v0"]:
        return [f"[validate] eval sets vs v0 ({v['reference']}): same per-source dev/test counts, no dev/test record "
                "dropped for overlapping the probes"]
    L = [f"[validate] WARNING eval sets differ from v0 ({v['reference']}): test:* of these sources is not "
         "item-identical to v0 (compare them with `zhjudge compare --allow-diff`, or leave them out of v0 comparisons)"]
    L += [f"  {s} {sp}: v0 {d['v0']}, now {d['now']}" for s, x in v["count_differences"].items() for sp, d in x.items()]
    L += [f"  {s}: {n} dev/test records dropped for overlapping the probes" for s, n in
          v["dropped_eval_overlapping_probes"].items()]
    return L


def validate(cfg: dict) -> dict:
    P = paths(cfg)
    uni = P["unified"]
    fail = []
    # 2) sha256
    sums = {}
    for line in (uni / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        sums[rel] = h
    on_disk = {str(p.relative_to(uni)) for p in uni.rglob("*.jsonl")}
    if set(sums) != on_disk:
        fail.append(f"SHA256SUMS file set mismatch: missing={sorted(on_disk - set(sums))} extra={sorted(set(sums) - on_disk)}")
    for rel, h in sums.items():
        if (uni / rel).exists() and hashlib.sha256((uni / rel).read_bytes()).hexdigest() != h:
            fail.append(f"sha256 mismatch {rel}")

    # 7) allowlist
    manifest = json.loads((uni / "manifest.json").read_text(encoding="utf-8"))
    smoke_override = manifest.get("allowlist", {}).get("smoke_allow_disabled", False)
    allow = load_allowlist(P["allowlist"])
    disabled = sorted({rel.split("/", 1)[1][:-6] for rel in on_disk} - {s for s in manifest["sources"] if is_enabled(allow, s)})
    if disabled and not smoke_override:
        fail.append(f"sources on disk that are not enabled in the allowlist: {disabled}")
    licenses = load_licenses(P["licenses"])
    lic_bad = []
    for s in sorted(manifest["sources"]):
        keep, problem = check_licence(cfg, allow, licenses, s)
        if not keep:
            lic_bad.append(problem or f"{s}: left out by licence profile {cfg['data'].get('license_profile')!r}")
    if lic_bad and not smoke_override:
        fail.append(f"sources on disk that the licence audit does not allow: {lic_bad}")

    # 1) schema + 3) ids
    ids = set()
    recs = []
    for rel in sorted(on_disk):
        for n, line in enumerate(open(uni / rel, encoding="utf-8")):
            r = json.loads(line)
            where = f"{rel}:{n + 1}"
            try:
                assert set(r) <= {"id", "lang", "type", "question", "options", "context", "label", "source", "split",
                                  "view", "group"}, set(r)
                assert "group" not in r or (r["split"] == "train" and isinstance(r["group"], str) and r["group"])
                assert r["lang"] in ("zh", "en") and r["split"] in ("train", "dev", "test") and r["type"] in ("choice", "score", "noul")
                assert isinstance(r["question"], str) and r["question"].strip()
                assert rel.startswith(r["split"] + "/") and rel.endswith(r["source"] + ".jsonl")
                if r["type"] == "noul":
                    assert isinstance(r["label"], bool) and "options" not in r
                else:
                    o = r["options"]
                    assert isinstance(o, list) and len(o) >= 2 and len(set(o)) == len(o) and all(isinstance(x, str) and x for x in o)
                    assert isinstance(r["label"], int) and not isinstance(r["label"], bool) and 0 <= r["label"] < len(o)
                if "view" in r:
                    check_view(r["view"], r["type"])
                assert r["id"] not in ids, "dup id"
            except AssertionError as e:
                fail.append(f"schema {where}: {e}")
                continue
            ids.add(r["id"])
            recs.append(r)

    # 4) train/eval leakage (pooled across sources)
    eval_units = defaultdict(set)
    for r in recs:
        if r["split"] != "train":
            for k in record_units(r):
                eval_units[k].add(r["source"])
    eval_only = {s for s, v in manifest["sources"].items() if v.get("eval_only")}
    inside = ContainmentIndex(r.get("context") or "" for r in recs
                              if r["source"] in eval_only and isinstance(r.get("context"), str))
    leak = Counter()
    leak_examples = []
    for r in recs:
        if r["split"] == "train":
            hit = [k for k in record_units(r) if k in eval_units or inside.hit(k)]
            if hit:
                leak[r["source"]] += 1
                if leak[r["source"]] <= 3:
                    leak_examples.append((r["id"], hit[0][:40], sorted(eval_units[hit[0]])))
    # 5/6) probe overlap
    bu, _ = probe_units(P["probes"])
    base_hits = Counter()
    for r in recs:
        if any(k in bu for k in record_units(r)):
            base_hits[f'{r["source"]}/{r["split"]}'] += 1

    counts = Counter((r["lang"], r["split"], r["type"]) for r in recs)
    by_src = Counter((r["source"], r["split"]) for r in recs)
    noul_rate = defaultdict(lambda: [0, 0])
    lab_dist = defaultdict(Counter)
    ctx_len = defaultdict(list)
    for r in recs:
        if r["type"] == "noul":
            noul_rate[(r["source"], r["split"])][0] += r["label"]
            noul_rate[(r["source"], r["split"])][1] += 1
        elif r["split"] == "train":
            lab_dist[r["source"] + ":" + r["type"]][r["options"][r["label"]] if r["type"] == "score" else r["label"]] += 1
        ctx_len[r["lang"]].append(len(r.get("context", "")) + len(r["question"]))

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0

    if leak:
        fail.append(f"train->eval leakage: {dict(leak)}")
    eval_base = {k: v for k, v in base_hits.items() if not k.endswith("/train")}
    if eval_base:
        fail.append(f"dev/test overlap with probe sets: {eval_base}")
    if any(k.endswith("/train") for k in base_hits):
        fail.append(f"train overlap with probe sets: {dict(base_hits)}")
    # 8) training-time instruction templates vs probe instructions
    collisions = probe_instruction_collisions(P["probes"], tasks_for(cfg))
    if collisions:
        fail.append(f"training instruction templates reproduce probe instructions: {collisions[:5]}")
    opt_coll = probe_option_collisions(P["probes"], cfg)
    if opt_coll["texts"]:
        fail.append(f"training option descriptions / level texts equal probe criteria: {opt_coll['texts'][:5]}")
    # 9) ids of the decisions eval will score (test sample + probes, and test:laya when eval.laya_test is on)
    from ..eval import duplicate_ids, eval_records, eval_seed, laya_records

    try:
        ev = eval_records(cfg, P, eval_seed(cfg))
    except Exception as e:  # noqa: BLE001  (eval would fail the same way, after training)
        fail.append(f"cannot assemble the eval set: {type(e).__name__}: {e}")
        ev = []
    try:
        ev += laya_records(cfg, P, eval_seed(cfg))
    except Exception as e:  # noqa: BLE001  (test:laya is a report addition: eval logs it and goes on)
        print(f"[validate] warning: test:laya records could not be built ({type(e).__name__}: {e})", flush=True)
    n_eval, dup = len(ev), duplicate_ids(ev)
    if dup:
        fail.append(f"{len(dup)} duplicate ids among the decisions eval will score, e.g. {dup[:5]}")
    try:
        vs_v0 = eval_set_vs_reference(cfg, manifest, by_src)
    except Exception as e:  # noqa: BLE001  (a warning, never a failure)
        vs_v0 = {"reference": None, "same_as_v0": None, "note": f"comparison failed: {type(e).__name__}: {e}"}
    report = {
        "records": len(recs),
        "totals": {f"{l}/{s}/{t}": n for (l, s, t), n in sorted(counts.items())},
        "totals_lang_split": {f"{l}/{s}": sum(n for (l2, s2, _), n in counts.items() if (l2, s2) == (l, s))
                              for l, s in sorted({(l, s) for l, s, _ in counts})},
        "per_source_split": {f"{a}/{b}": n for (a, b), n in sorted(by_src.items())},
        "noul_true_rate": {f"{a}/{b}": round(x / y, 3) for (a, b), (x, y) in sorted(noul_rate.items())},
        "score_level_dist_train": {k: dict(v) for k, v in lab_dist.items() if ":score" in k},
        "choice_label_index_max_share_train": {k: round(max(v.values()) / sum(v.values()), 3)
                                               for k, v in lab_dist.items() if ":choice" in k},
        "chars_question_plus_context": {l: {"p50": pct(v, .5), "p90": pct(v, .9), "p99": pct(v, .99)} for l, v in ctx_len.items()},
        "train_records_leaking_into_eval": dict(leak),
        "leak_examples": leak_examples,
        "records_overlapping_probes": dict(base_hits),
        "probe_units": len(bu),
        "probe_instruction_collisions": collisions,
        "probe_option_text_collisions": opt_coll["texts"],
        "probe_label_names_shared": opt_coll["labels"],
        "eval_decisions": n_eval,
        "eval_duplicate_ids": dup[:20],
        "eval_set_vs_v0": vs_v0,
        "smoke_allow_disabled": smoke_override,
        "hard_failures": fail,
    }
    (uni / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("records", "totals_lang_split", "train_records_leaking_into_eval",
                                             "records_overlapping_probes", "probe_units", "eval_decisions",
                                             "hard_failures")}, ensure_ascii=False, indent=1))
    print("\n".join(eval_set_lines(vs_v0)), flush=True)
    if opt_coll["labels"]:
        print(f"[validate] option label names this config adds that also name probe options (generic names, reported "
              f"only): {opt_coll['labels']}", flush=True)
    return report


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args(argv)
    rep = validate(load_config(a.config, a.set))
    sys.exit(1 if rep["hard_failures"] else 0)


if __name__ == "__main__":
    main()
