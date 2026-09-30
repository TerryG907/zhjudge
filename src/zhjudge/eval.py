"""Evaluate a Laya-format checkpoint: accuracy, Brier, ECE-15, NLL, per language / task / question type, next to the
laya-multilingual baseline.

  uv run zhjudge eval --config ...                     # evaluates runs/<name>/model (calibrated)
  uv run zhjudge eval --config ... --ckpt some/dir     # any Laya checkpoint directory

Evaluation sets:
  test:*   the unified corpus' held-out split (eval.split, default test) of every built source, including the
           eval-only sets (Belebele, XCOPA, XWinograd, Global-MMLU, MInDS-14), capped at eval.max_per_source; the
           capped sample is drawn with eval.seed (13, v0's run.seed), so run.seed no longer moves it
  probe:*  the spike's baseline probe sets (data/probes, built by `zhjudge probes`): zh probe with Chinese and
           English prompts, EN control, XNLI parallel zh/en, typed-decisions test
  test:laya:*  (eval.laya_test > 0) up to that many records per source of the test:* sample, written Laya-style
           (JSON state with named fields, field-aware instruction; zhjudge.data.views) with a fixed rendering per item
Baselines:
  recorded  the spike's measured laya-multilingual numbers on the probe sets (zhjudge/baselines.py)
  measured  (eval.baseline: true) laya-multilingual re-run here on exactly the same records, incl. test:*
Loading goes through upstream `laya.Agent` (same config and temperature handling users get); the forward pass is
batched. Outputs runs/<name>/eval.json, eval.md, eval_preds.jsonl (+ eval_preds_baseline.jsonl).

eval.json, eval.md and the preds are written first with exactly the v0 report; the additions follow, each on its own
(a failure is logged in report_errors and never costs the report or the exit code): eval-set fingerprints, Wilson
intervals, paired differences vs the re-measured baseline and vs eval.compare_preds (e.g. v0's eval_preds.jsonl, see
also `zhjudge compare`), macro averages, test:group:{trained,eval-only}, metrics at temperature 1, test:laya:* and how
the options were fitted into head_max_len. The intervals cover item sampling only, not training-seed variance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import device as D  # noqa: E402
from .baselines import HEADLINE, LAYA_MULTILINGUAL, SOURCE  # noqa: E402
from .config import add_common_args, load_config, paths  # noqa: E402
from .infer import agent_amp, batched_logits, probs_from_logits  # noqa: E402
from .metrics import summarize  # noqa: E402
from .model import BASELINE_REPO, BASELINE_REVISION, BASELINE_SUBFOLDER, baseline_dir, load_agent  # noqa: E402

BASELINE_LABEL = f"{BASELINE_REPO}@{BASELINE_REVISION[:7]}/{BASELINE_SUBFOLDER} (laya-multilingual)"
from .records import cap_per_source, encode_records, load_unified, option_fit, read_jsonl, to_laya  # noqa: E402
from .stats import Z95, ece_floor, fingerprint, fmt_d, paired, score_table, wilson  # noqa: E402


def gold_index(q, gold):
    if q["type"] == "choice":
        return list(q["criteria"]).index(gold)
    if q["type"] == "score":
        return int(gold)
    return int(bool(gold))


def td_gold(qdef, g):
    if qdef["type"] == "choice":
        keys = list(qdef["criteria"])
        return keys.index(g["label"]), [g["probabilities"][k] for k in keys]
    if qdef["type"] == "score":
        n = len(qdef["criteria"])
        return int(g["label"]), [g["probabilities"].get(str(i), 0.0) for i in range(n)]
    pt = g.get("noul", g["probabilities"]["true"])
    return int(str(g["label"]).lower() == "true"), [1 - pt, pt]


def n_options(q):
    return 2 if q["type"] == "noul" else len(q["criteria"])


def eval_seed(cfg) -> int:
    """Seed of the capped test sample: eval.seed (13 = v0's run.seed), whatever run.seed is."""
    return int(cfg["eval"].get("seed", 13))


def test_sample(cfg, P, seed):
    """The capped held-out sample test:* scores (unified records, file order); None without a built corpus."""
    if not (P["unified"] / "manifest.json").exists():
        return None
    e = cfg["eval"]
    return cap_per_source(load_unified(P["unified"], e["split"]), e["max_per_source"], seed)


def eval_only_sources(P) -> frozenset:
    man = P["unified"] / "manifest.json"
    if not man.exists():
        return frozenset()
    return frozenset(k for k, v in json.loads(man.read_text(encoding="utf-8"))["sources"].items() if v.get("eval_only"))


def eval_records(cfg, P, seed):
    """Every decision to score: Laya record + slice keys."""
    out = []
    e = cfg["eval"]
    man_path = P["unified"] / "manifest.json"
    recs = test_sample(cfg, P, seed)
    if recs is not None:
        for r in recs:
            lr = to_laya(r, 0.0, 0.0)
            lr["_slices"] = ["test:ALL", f"test:lang:{r['lang']}", f"test:source:{r['source']}",
                             f"test:qtype:{r['type']}", f"test:lang:{r['lang']}/qtype:{r['type']}"]
            out.append(lr)
    else:
        print(f"[eval] {man_path} missing: skipping test:* sets", flush=True)
    pdir = P["probes"]
    if e["probes"] and (pdir / "zh_probe.jsonl").exists():
        def add_items(fname, variant, name, keep=lambda it: True):
            kq, kg = ("q_native", "gold_native") if variant == "native" else ("q_en", "gold_en")
            for it in read_jsonl(pdir / fname):
                if not keep(it):
                    continue
                qq = it[kq]
                g = gold_index(qq["question"], it[kg])
                nm = name(it) if callable(name) else name
                out.append({"id": f"{it['id']}#{variant}", "source": it["source"], "lang": it["lang"],
                            "state": qq["state"], "question": qq["question"], "gold": g,
                            "target": [0.0] * n_options(qq["question"]),
                            "_slices": [nm, f"{nm}/family:{it['family']}", f"{nm}/qtype:{it['qtype']}"]})

        add_items("zh_probe.jsonl", "native", "probe:zh/zh-prompt")
        add_items("zh_probe.jsonl", "en", "probe:zh/en-prompt")
        add_items("en_control.jsonl", "native", "probe:en-control")
        add_items("xnli_parallel.jsonl", "native",
                  lambda it: "probe:xnli/en" if it["lang"] == "en" else "probe:xnli/zh/zh-prompt")
        add_items("xnli_parallel.jsonl", "en", "probe:xnli/zh/en-prompt", keep=lambda it: it["lang"] == "zh")
        for c in read_jsonl(pdir / "typed_decisions_test.jsonl"):
            for qid, qdef in c["questions"].items():
                gi, gsoft = td_gold(qdef, c["gold"][qid])
                nm = "probe:typed-decisions"
                out.append({"id": f"{c['id']}/{qid}", "source": "LocalLLaMA/typed-decisions", "lang": "en",
                            "state": c["state"], "question": qdef, "gold": gi, "gold_soft": gsoft,
                            "gold_score": c["gold"][qid].get("score"), "target": [0.0] * n_options(qdef),
                            "_slices": [nm, f"{nm}/family:{c['workflow']}", f"{nm}/qtype:{qdef['type']}"]})
    elif e["probes"]:
        print(f"[eval] no probe sets in {pdir} (run `zhjudge probes`): skipping probe:* sets", flush=True)
    return out


def laya_records(cfg, P, seed, sample=None):
    """test:laya: up to eval.laya_test records per source of the test:* sample that have a view, written Laya-style by
    zhjudge.data.views.render with a fixed rng per record id (30% other-language instruction and field names, 30% bare
    NLI labels, options in the test order), ids '<id>#laya'. `render` hashes the rendered state + question, so a
    changed template makes a different item (fingerprints, `zhjudge compare`)."""
    from .data import views

    n = int(cfg["eval"].get("laya_test") or 0)
    sample = test_sample(cfg, P, seed) if sample is None else sample
    if n <= 0 or not sample:
        return []
    out = []
    for r in cap_per_source([r for r in sample if r.get("view")], n, f"{seed}:laya"):
        lr = to_laya(views.render(r, random.Random(f"laya-test:{r['id']}"), 0.3, 0.3), 0.0, 0.0)
        lr["id"] = f"{r['id']}#laya"
        rendered = json.dumps([lr["state"], lr["question"]], ensure_ascii=False)
        lr["render"] = hashlib.sha256(rendered.encode()).hexdigest()[:12]
        lr["_slices"] = [f"test:laya:{k}" for k in ("ALL", f"lang:{r['lang']}", f"source:{r['source']}",
                                                     f"qtype:{r['type']}")]
        out.append(lr)
    return out


def duplicate_ids(records) -> list[str]:
    seen, dup = set(), []
    for r in records:
        if r["id"] in seen:
            dup.append(r["id"])
        seen.add(r["id"])
    return dup


def unique_ids(records):
    """The first record of each id: rows are matched to records by id (`zhjudge validate` fails on duplicates)."""
    dup = duplicate_ids(records)
    if not dup:
        return records
    print(f"[eval] warning: {len(dup)} duplicate eval ids (e.g. {dup[:3]}); scoring the first record of each",
          flush=True)
    seen, out = set(), []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def display_path(p) -> str:
    """Path for reports that may be published: relative to the repo, never an absolute local path."""
    from .config import repo_root

    p = Path(p).resolve()
    for base in (repo_root(), Path(os.environ.get("ZHJUDGE_RUNS_DIR") or repo_root() / "runs").resolve()):
        try:
            return str(p.relative_to(base))
        except ValueError:
            pass
    return p.name


def score_checkpoint(ckpt, records, dev, batch_size, label=None):
    t0 = time.time()
    agent = load_agent(ckpt, dev.type)
    items, dropped = encode_records(records, agent.tok, agent.cfg.get("max_len", 512),
                                    agent.cfg.get("head_max_len", 192), log_every=0)
    by_id = {r["id"]: r for r in records}
    logits = batched_logits(agent.model, items, agent.tok.pad_token_id, agent.device, agent_amp(agent), batch_size)
    probs = probs_from_logits(logits, [it["qtype"] for it in items], agent.temperature, agent.temperature_by_options)
    rows = []
    for it, p in zip(items, probs):
        r = by_id[it["id"]]
        rows.append({"id": it["id"], "probs": [round(float(x), 6) for x in p], "gold": r["gold"],
                     "qtype": r["question"]["type"], "gold_soft": r.get("gold_soft"), "gold_score": r.get("gold_score"),
                     "_slices": r["_slices"]})
        if "render" in r:
            rows[-1]["render"] = r["render"]
    slices = defaultdict(list)
    for row in rows:
        for s in row["_slices"]:
            slices[s].append(row)
    res = {s: summarize(v) for s, v in sorted(slices.items())}
    meta = {"checkpoint": label or display_path(ckpt), "decisions": len(rows), "dropped_options_do_not_fit": dropped,
            "temperature": agent.temperature, "temperature_by_options": agent.temperature_by_options,
            "device": str(agent.device), "seconds": round(time.time() - t0, 1)}
    # for the report additions: the same logits at temperature 1, and what the option-fit report needs
    x = {"logits": logits, "items": items, "tok": agent.tok, "head_max_len": agent.cfg.get("head_max_len", 192)}
    del agent
    return res, rows, meta, x


def fmt(m, keys=("acc", "brier", "ece15")):
    return " / ".join("-" if m is None or m.get(k) is None else f"{m[k]:.4f}" for k in keys) if m else "-"


def report_md(ours, measured, name):
    L = [f"# Eval: {name}", "",
         "acc / Brier (one-hot, lower is better) / ECE-15. `recorded` = laya-multilingual as measured by the spike "
         f"({SOURCE}); `re-measured` = laya-multilingual run here on the same records.", "",
         "## Baseline probe sets", "", "| set | n | ours | laya-multilingual recorded | laya-multilingual re-measured |",
         "|---|---|---|---|---|"]
    for key, label in HEADLINE:
        m = ours.get(key)
        L.append(f"| {label} | {m['n'] if m else '-'} | {fmt(m)} | {fmt(LAYA_MULTILINGUAL.get(key))} | "
                 f"{fmt((measured or {}).get(key))} |")
    fam = sorted(k for k in ours if k.startswith("probe:zh/zh-prompt/"))
    if fam:
        L += ["", "### zh probe (Chinese prompt) by family / question type", "",
              "| slice | n | ours | recorded | re-measured |", "|---|---|---|---|---|"]
        for k in fam:
            L.append(f"| {k.split('/', 2)[2]} | {ours[k]['n']} | {fmt(ours[k])} | {fmt(LAYA_MULTILINGUAL.get(k))} | "
                     f"{fmt((measured or {}).get(k))} |")
    tk = [k for k in ours if k.startswith("test:")]
    if tk:
        L += ["", "## Unified held-out split (test:*)", "", "| slice | n | ours | laya-multilingual re-measured |",
              "|---|---|---|---|"]
        def rank(k):
            if k == "test:ALL":
                return 0
            if k.startswith("test:lang:") and "/" not in k:
                return 1
            if k.startswith("test:qtype:"):
                return 2
            if k.startswith("test:lang:"):
                return 3
            return 4  # per source
        for k in sorted(tk, key=lambda k: (rank(k), k)):
            L.append(f"| {k[5:]} | {ours[k]['n']} | {fmt(ours[k])} | {fmt((measured or {}).get(k))} |")
    return "\n".join(L) + "\n"


REPORT_NOTE = ("The 95% intervals cover item sampling only (which items were drawn), not training-seed variance: one "
               "training run per model cannot show how much a retrain would move a number.")
PROBE_FILES = ("zh_probe.jsonl", "en_control.jsonl", "xnli_parallel.jsonl", "typed_decisions_test.jsonl")
HEADLINE_MORE = [("test:ALL", "held-out test, all"), ("test:lang:zh", "held-out test, zh"),
                 ("test:lang:en", "held-out test, en"), ("test:group:trained", "held-out test, trained sources"),
                 ("test:group:eval-only", "held-out test, eval-only sources (never trained on)"),
                 ("test:laya:ALL", "held-out test written Laya-style (test:laya)")]
# (label, slice prefix, which parts) -> unweighted mean over the parts
MACROS = ([("test: sources", "test:source:", lambda k, eo: True),
           ("test: zh sources", "test:source:", lambda k, eo: k.endswith("_zh")),
           ("test: en sources", "test:source:", lambda k, eo: k.endswith("_en")),
           ("test: trained sources", "test:source:", lambda k, eo: k not in eo),
           ("test: eval-only sources", "test:source:", lambda k, eo: k in eo),
           ("test:laya: sources", "test:laya:source:", lambda k, eo: True)]
          + [(f"{label}: families", f"{key}/family:", lambda k, eo: True) for key, label in HEADLINE])
K3 = ("brier", "ece15", "nll")


def extra_slices(row, eval_only) -> list[str]:
    """test:group:{trained,eval-only} of a plain test row (not in v0's slice set, so reported next to it)."""
    s = row["_slices"]
    if s[0] != "test:ALL":
        return []
    return [f"test:group:{'eval-only' if s[2][len('test:source:'):] in eval_only else 'trained'}"]


def by_slice(rows, eval_only) -> dict:
    out = defaultdict(list)
    for r in rows:
        for s in r["_slices"] + extra_slices(r, eval_only):
            out[s].append(r)
    return out


def write_preds(path, rows, eval_only, mode="w"):
    """eval_preds rows: v0's fields (id, probs, gold, qtype, gold_soft, gold_score), plus the slices (`zhjudge compare`)
    and, on test:laya rows, the rendering hash."""
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**{k: v for k, v in r.items() if not k.startswith("_")},
                                "slices": r["_slices"] + extra_slices(r, eval_only)}, ensure_ascii=False) + "\n")


def score_laya(cfg, P, seed, ckpt, dev, bs, run_base, eval_only):
    """Score test:laya (ours, and the baseline when it runs) and append the rows to the preds files."""
    from .data.views import VIEWS_VERSION

    recs = laya_records(cfg, P, seed)
    if not recs:
        return None
    print(f"[eval] test:laya: {len(recs)} decisions (views v{VIEWS_VERSION})", flush=True)
    _, rows, meta, x = score_checkpoint(ckpt, recs, dev, bs)
    _, brows, bmeta, bx = (score_checkpoint(baseline_dir(), recs, dev, bs, label=BASELINE_LABEL) if run_base
                           else (None, [], None, None))
    write_preds(P["run"] / "eval_preds.jsonl", rows, eval_only, "a")
    if brows:
        write_preds(P["run"] / "eval_preds_baseline.jsonl", brows, eval_only, "a")
    info = {"views_version": VIEWS_VERSION, "per_source": int(cfg["eval"]["laya_test"]), "records": len(recs),
            "decisions": len(rows), "dropped_options_do_not_fit": meta["dropped_options_do_not_fit"],
            "fingerprint": fingerprint(recs, f"views v{VIEWS_VERSION}"),
            "seconds": round(meta["seconds"] + (bmeta or {}).get("seconds", 0), 1)}
    return {"info": info, "records": recs, "rows": rows, "x": x, "brows": brows, "bx": bx}


def rows_t1(rows, x):
    """The rows with their probabilities at temperature 1 (the same logits, calibration undone)."""
    probs = probs_from_logits(x["logits"], [it["qtype"] for it in x["items"]])
    return [{**r, "probs": p.tolist()} for r, p in zip(rows, probs)]


def intervals(sl, table) -> dict:
    return {s: {"n": len(v), "acc_wilson95": wilson(sum(table[r["id"]][0] for r in v), len(v)),
                "ece15_floor": round(ece_floor([max(r["probs"]) for r in v]), 4)} for s, v in sorted(sl.items())}


def laya_vs_plain(sl, table) -> dict:
    """Per test:laya slice, the same model on the same items: Laya-style minus plain."""
    plain = {r["id"]: r for r in sl.get("test:ALL", [])}
    out = {}
    for s, v in sorted(sl.items()):
        if s.startswith("test:laya:"):
            lay = [{**r, "id": r["id"][:-len("#laya")]} for r in v]
            out[s] = paired(lay, [plain[r["id"]] for r in lay if r["id"] in plain], None, table)
    return out


def macro_rows(allm, pairs, eval_only) -> list[dict]:
    out = []
    for label, pre, keep in MACROS:
        keys = sorted(k for k in allm if k.startswith(pre) and "/" not in k[len(pre):]
                      and keep(k[len(pre):], eval_only))
        if len(keys) < 2:
            continue
        acc = sum(allm[k]["acc"] for k in keys) / len(keys)
        se = sum(allm[k]["acc"] * (1 - allm[k]["acc"]) / allm[k]["n"] for k in keys) ** 0.5 / len(keys)
        m = {"average": label, "parts": len(keys), "acc": round(acc, 4),  # normal approximation, clipped to [0, 1]
             "acc_ci95": [round(max(0.0, acc - Z95 * se), 4), round(min(1.0, acc + Z95 * se), 4)]}
        ps = [pairs.get(k) for k in keys]
        if all(p and p.get("d_acc_ci95") for p in ps):  # parts are disjoint item sets: variances add
            d = sum(p["d_acc"] for p in ps) / len(ps)
            se = sum(((p["d_acc_ci95"][1] - p["d_acc_ci95"][0]) / (2 * Z95)) ** 2 for p in ps) ** 0.5 / len(ps)
            m.update(baseline_acc=round(sum(p["acc_b"] for p in ps) / len(ps), 4), d_acc=round(d, 4),
                     d_acc_ci95=[round(d - Z95 * se, 4), round(d + Z95 * se, 4)])
        out.append(m)
    return out


def eval_set_info(cfg, P, seed, records, sl) -> dict:
    """What was scored, to tell mechanically whether two runs saw the same items: corpus and probe file hashes, the
    sample settings and per-slice fingerprints (sha256 of the sorted (id, gold[, rendering]) of the scored items)."""
    from .data.views import VIEWS_VERSION

    e, man, pdir = cfg["eval"], P["unified"] / "manifest.json", P["probes"]
    return {"split": e["split"], "eval_seed": seed, "max_per_source": e["max_per_source"],
            "corpus_sha256": json.loads(man.read_text(encoding="utf-8")).get("corpus_sha256") if man.exists() else None,
            "probes_sha256": {f: hashlib.sha256((pdir / f).read_bytes()).hexdigest() for f in PROBE_FILES
                              if e["probes"] and (pdir / f).exists()},
            "records": len(records), "fingerprint": fingerprint(records),
            "slices": {s: fingerprint(v, f"views v{VIEWS_VERSION}" if s.startswith("test:laya:") else "")
                       for s, v in sorted(sl.items())}}


def report_md_extra(rep) -> str:
    """The additions to eval.md (rep: the report sections; a table whose section failed is left out)."""
    allm, allb, meta = rep["allm"], rep["allb"], rep["meta"]
    ci, pairs = rep.get("intervals") or {}, rep.get("paired_vs_baseline") or {}
    cim = ci.get("model") or {}
    L = ["", "## Report additions", "",
         REPORT_NOTE + " `acc` in %; `95%` = Wilson interval of our accuracy (macro averages: normal approximation); "
         "both treat every decision as independent, so for typed-decisions (several questions per case) they are too "
         "narrow; `Δ` = ours minus laya-multilingual re-measured on the same items, in points (Brier x100), with a "
         "paired 95% interval (typed-decisions: the questions of one case resampled together), `*` = the interval "
         "excludes 0; `McNemar p` = exact test on the items only one of the two gets right; `ECE floor` = the ECE-15 "
         "a perfectly calibrated model with our confidences would still show at this n (an ECE near it is noise)."]

    def acc(m):
        return "-" if not m else f"{100 * m['acc']:.1f}"

    def wil(c):
        return "-" if not c or not c.get("acc_wilson95") else "%.1f–%.1f" % tuple(100 * v for v in c["acc_wilson95"])

    def mcn(p):
        return "-" if not p or p.get("mcnemar_p") is None else f"{p['mcnemar_p']:.3g}"

    def headline():
        out = ["", "### Headline sets", "", "| set | n | ours acc | 95% | ECE-15 | ECE floor | re-measured acc | "
               "Δ acc | Δ Brier | McNemar p |", "|---|---|---|---|---|---|---|---|---|---|"]
        for key, label in HEADLINE + HEADLINE_MORE:
            m, c, p = allm.get(key), cim.get(key) or {}, pairs.get(key)
            if m:
                out.append(f"| {label} | {m['n']} | {acc(m)} | {wil(c)} | {m['ece15']:.3f} | "
                           f"{c.get('ece15_floor', float('nan')):.3f} | {acc(allb.get(key))} | {fmt_d(p)} | "
                           f"{fmt_d(p, 'd_brier')} | {mcn(p)} |")
        return out

    def families():
        out = []
        for key, label in HEADLINE:
            sub = sorted(k for k in allm if k.startswith(key + "/"))
            if sub:
                out += ["", f"### {label}: by family / question type", "",
                        "| slice | n | ours acc | 95% | re-measured acc | Δ acc | McNemar p |",
                        "|---|---|---|---|---|---|---|"]
                out += [f"| {k[len(key) + 1:]} | {allm[k]['n']} | {acc(allm[k])} | {wil(cim.get(k))} | "
                        f"{acc(allb.get(k))} | {fmt_d(pairs.get(k))} | {mcn(pairs.get(k))} |" for k in sub]
        return out

    def macro():
        rows = rep.get("macro") or []
        out = ["", "### Macro averages (each source / family weighs 1)", "",
               "| average | parts | ours acc | 95% (normal approx.) | re-measured acc | Δ acc |", "|---|---|---|---|---|---|"]
        out += [f"| {m['average']} | {m['parts']} | {acc(m)} | {wil({'acc_wilson95': m['acc_ci95']})} | "
                f"{acc({'acc': m['baseline_acc']}) if 'baseline_acc' in m else '-'} | {fmt_d(m)} |" for m in rows]
        return out if rows else []

    def laya():
        info, keys = rep.get("laya_test"), [k for k in allm if k.startswith("test:laya:")]
        if not info or not keys:
            return []
        lvp = rep.get("laya_vs_plain") or {}
        rank = {"ALL": 0, "lang": 1, "qtype": 2, "source": 3}
        out = ["", f"### Held-out test written Laya-style (test:laya, views v{info['views_version']})", "",
               f"{info['decisions']} decisions: up to {info['per_source']} records per source of the test:* sample "
               "that have a view, written as a JSON state with named fields and a field-aware instruction (the "
               "training templates of zhjudge.data.views, fixed per item: 30% other-language instruction and field "
               "names, 30% bare NLI labels, options in the test order). Each item is also in test:* in plain form: "
               "`Laya − plain` is the same model on the same items.", "",
               "| slice | n | ours acc | 95% | re-measured acc | Δ acc | ours: Laya − plain | "
               "re-measured: Laya − plain |", "|---|---|---|---|---|---|---|---|"]
        for k in sorted(keys, key=lambda k: (rank.get(k.split(":")[2], 9), k)):
            out.append(f"| {k[len('test:laya:'):]} | {allm[k]['n']} | {acc(allm[k])} | {wil(cim.get(k))} | "
                       f"{acc(allb.get(k))} | {fmt_d(pairs.get(k))} | {fmt_d((lvp.get('model') or {}).get(k))} | "
                       f"{fmt_d((lvp.get('baseline_measured') or {}).get(k))} |")
        return out

    def t1():
        u = rep.get("uncalibrated") or {}
        um, ub = u.get("model") or {}, u.get("baseline_measured") or {}
        out = ["", "### Without temperatures (T = 1)", "",
               f"`zhjudge calibrate` fitted T = {meta.get('temperature')} (by number of options: "
               f"{meta.get('temperature_by_options') or '{}'}). The same logits at T = 1 show what training alone "
               "gives, and compare across runs whose calibration differs (v0 fitted its temperatures to the "
               "label-smoothed training targets, 1.09 / 1.20 / 1.20; v1 fits them to the gold label). "
               "Brier / ECE-15 / NLL:", "",
               "| set | n | ours | ours T=1 | re-measured | re-measured T=1 |", "|---|---|---|---|---|---|"]
        out += [f"| {label} | {um[key]['n']} | {fmt(allm.get(key), K3)} | {fmt(um[key], K3)} | "
                f"{fmt(allb.get(key), K3)} | {fmt(ub.get(key), K3)} |" for key, label in HEADLINE + HEADLINE_MORE
                if key in um]
        return out if um else []

    def previous():
        pv = rep.get("paired_vs_previous")
        if not pv:
            return []
        from .compare import slice_order

        pp = pv["slices"]
        same = sum(p.get("status") == "same items" for p in pp.values())
        one = sum(p.get("status", "").startswith("only in") for p in pp.values())
        out = ["", f"### Versus {pv['label']} (eval.compare_preds), item by item", "",
               f"{same} of {len(pp)} slices have exactly the same items (ids, golds, renderings) in both runs; {one} "
               "exist in one run only (e.g. test:laya:* when the other run did not score it) and are left out. "
               "`only ours` / `only prev` count items the other run did not score: then the Δ covers the common "
               "items only.", "",
               "| slice | n common | only ours | only prev | prev acc | ours acc | Δ acc | Δ Brier | McNemar p |",
               "|---|---|---|---|---|---|---|---|---|"]
        for k in slice_order(pp):
            p = pp[k]
            if p.get("n"):
                out.append(f"| {k} | {p['n']} | {p['only_a']} | {p['only_b']} | {acc({'acc': p['acc_b']})} | "
                           f"{acc({'acc': p['acc_a']})} | {fmt_d(p)} | {fmt_d(p, 'd_brier')} | {mcn(p)} |")
        return out

    def fit():
        f = meta.get("option_fit")
        if not f:
            return []
        bad = {s: v for s, v in f["sources"].items()
               if v["options_cut"] or v["options_colliding"] or v["records_instruction_cut"] or v["dropped"]}
        out = ["", f"### How the options fitted into the question head (head_max_len {f['head_max_len']})", "",
               "Report only (the inputs are what Laya's runtime builds): laya.common.build_sequence keeps at most 48 "
               "tokens per option; once the options overflow head_max_len it keeps [MASK] + max(4, (head_max_len − "
               "16) // K) − 1 tokens of each and cuts the instruction to max(8, what is left) tokens. Options cut to "
               "the same tokens cannot be told apart.", ""]
        if not bad:
            return out + ["Every option and instruction of every scored question fitted."]
        out += ["| source | records | with cut options | options cut / all | with identical options (e.g.) | "
                "instruction cut (tokens kept) | dropped |", "|---|---|---|---|---|---|---|"]
        out += [f"| {s} | {v['records']} | {v['records_options_cut']} | {v['options_cut']} / {v['options']} | "
                f"{v['records_colliding']} ({'; '.join(v['examples'][:2]) or '-'}) | {v['records_instruction_cut']}"
                + (f" ({v['instruction_kept_min']})" if v["records_instruction_cut"] else "") + f" | {v['dropped']} |"
                for s, v in bad.items()]
        return out

    for block in (headline, families, macro, laya, t1, previous, fit):
        try:
            L += block()
        except Exception as e:  # noqa: BLE001
            L += ["", f"(table {block.__name__} failed: {type(e).__name__}: {e})"]
    return "\n".join(L) + "\n"


def add_report(cfg, P, seed, ckpt, dev, bs, eval_only, records, out, md, rows, x, brows, bx) -> dict:
    """The additions to the v0 report already on disk. Each section runs on its own: a failure is logged in
    report_errors and leaves the other sections, the v0 report and the exit code alone."""
    errors = {}
    ours, measured, meta, bmeta = out["model"], out["baseline_measured"], out["meta"], out["baseline_meta"]

    def section(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (report code never fails an eval)
            errors[name] = f"{type(e).__name__}: {e}"
            print(f"[eval] warning: report section {name!r} failed ({errors[name]}); the rest is unaffected",
                  flush=True)
            return None

    def fit(xx):
        return {"head_max_len": xx["head_max_len"],
                "sources": option_fit(xx["items"], records, xx["tok"], xx["head_max_len"])}

    def uncalibrated(rr, xx, lrows, lx):
        rr = rows_t1(rr, xx) + (rows_t1(lrows, lx) if lrows else [])
        return {s: summarize(v, bootstrap=False) for s, v in sorted(by_slice(rr, eval_only).items())}

    laya = section("test:laya", lambda: score_laya(cfg, P, seed, ckpt, dev, bs, measured is not None, eval_only))
    laya = laya or {}
    all_rows, all_brows = rows + laya.get("rows", []), brows + laya.get("brows", [])
    sa = section("scores", lambda: score_table(all_rows)) or {}
    sb = section("scores (baseline)", lambda: score_table(all_brows)) or {}
    A = section("slices", lambda: by_slice(all_rows, eval_only)) or {}
    B = section("slices (baseline)", lambda: by_slice(all_brows, eval_only)) or {}
    extra = section("extra slices", lambda: {s: summarize(v) for s, v in sorted(A.items()) if s not in ours}) or {}
    bextra = None
    if measured is not None:
        bextra = section("extra slices (baseline)",
                         lambda: {s: summarize(v) for s, v in sorted(B.items()) if s not in measured})
    rep = {"model_extra": extra, "baseline_measured_extra": bextra, "meta": meta, "laya_test": laya.get("info"),
           "allm": {**ours, **extra}, "allb": {**(measured or {}), **(bextra or {})}}
    if laya:
        meta["laya_test"] = laya["info"]
    section("eval_set", lambda: meta.update(eval_set=eval_set_info(cfg, P, seed, records, A)))
    section("option_fit", lambda: meta.update(option_fit=fit(x)))
    if bx:
        section("option_fit (baseline)", lambda: bmeta.update(option_fit=fit(bx)))
    rep["intervals"] = section("intervals", lambda: {
        "model": intervals(A, sa), "baseline_measured": intervals(B, sb) if measured is not None else None})
    rep["uncalibrated"] = section("uncalibrated", lambda: {
        "model": uncalibrated(rows, x, laya.get("rows"), laya.get("x")),
        "baseline_measured": uncalibrated(brows, bx, laya.get("brows"), laya.get("bx")) if bx else None})
    if measured is not None:
        rep["paired_vs_baseline"] = section("paired_vs_baseline",
                                            lambda: {s: paired(v, B.get(s, []), sa, sb) for s, v in sorted(A.items())})
    if laya:
        rep["laya_vs_plain"] = section("laya_vs_plain", lambda: {
            "model": laya_vs_plain(A, sa), "baseline_measured": laya_vs_plain(B, sb) if measured is not None else None})
    rep["macro"] = section("macro", lambda: macro_rows(rep["allm"], rep.get("paired_vs_baseline") or {}, eval_only))
    if cfg["eval"].get("compare_preds"):
        def previous():
            from .compare import compare_rows

            pth = Path(cfg["eval"]["compare_preds"])
            mine = [{**r, "slices": r["_slices"] + extra_slices(r, eval_only)} for r in all_rows]
            return {"label": cfg["eval"].get("compare_label") or display_path(pth), "file": display_path(pth),
                    "slices": compare_rows(mine, read_jsonl(pth), eval_only, allow_diff=True)}
        rep["paired_vs_previous"] = section("paired_vs_previous", previous)
    full = dict(out)
    full.update({k: rep.get(k) for k in ("model_extra", "baseline_measured_extra", "intervals", "uncalibrated",
                                         "paired_vs_baseline", "laya_vs_plain", "macro", "paired_vs_previous")})
    full["report_note"] = REPORT_NOTE
    md_extra = section("eval.md", lambda: report_md_extra(rep))
    full["report_errors"] = errors

    def write():
        tmp = P["run"] / "eval.json.tmp"
        tmp.write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, P["run"] / "eval.json")
        if md_extra:
            (P["run"] / "eval.md").write_text(md + md_extra, encoding="utf-8")
            print(md_extra, flush=True)
    section("write", write)
    if errors:
        print(f"[eval] report sections that failed: {errors}", flush=True)
    return full


def evaluate(cfg: dict, ckpt=None, baseline=None) -> dict:
    P = paths(cfg)
    D.set_threads(cfg["train"]["cpu_threads"])
    dev = D.select_device(cfg["train"]["device"])
    ckpt = Path(ckpt) if ckpt else P["model"]
    seed = eval_seed(cfg)
    records = eval_records(cfg, P, seed)
    if not records:
        raise SystemExit("nothing to evaluate: build the corpus (`zhjudge build`) and/or probes (`zhjudge probes`)")
    records = unique_ids(records)
    print(f"[eval] {len(records)} decisions on {dev.type}", flush=True)
    bs = cfg["eval"]["batch_size"]
    ours, rows, meta, x = score_checkpoint(ckpt, records, dev, bs)
    run_base = cfg["eval"]["baseline"] if baseline is None else baseline
    measured, brows, bmeta, bx = None, [], None, None
    if run_base:
        measured, brows, bmeta, bx = score_checkpoint(baseline_dir(), records, dev, bs, label=BASELINE_LABEL)
    P["run"].mkdir(parents=True, exist_ok=True)
    out = {"model": ours, "meta": meta, "baseline_recorded": LAYA_MULTILINGUAL, "baseline_recorded_source": SOURCE,
           "baseline_measured": measured, "baseline_meta": bmeta, "smoke": bool(cfg["run"].get("smoke"))}
    (P["run"] / "eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        eo = eval_only_sources(P)
    except Exception:  # noqa: BLE001  (only the test:group slices need it)
        eo = frozenset()
    write_preds(P["run"] / "eval_preds.jsonl", rows, eo)
    (P["run"] / "eval_preds_baseline.jsonl").unlink(missing_ok=True)  # no stale baseline preds from an earlier eval
    if brows:
        write_preds(P["run"] / "eval_preds_baseline.jsonl", brows, eo)
    md = report_md(ours, measured, cfg["run"]["name"])
    (P["run"] / "eval.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    try:
        return add_report(cfg, P, seed, ckpt, dev, bs, eo, records, out, md, rows, x, brows, bx)
    except Exception as e:  # noqa: BLE001  (the v0 report above is complete; the additions never fail an eval)
        print(f"[eval] warning: report additions failed ({type(e).__name__}: {e})", flush=True)
        return out


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ckpt", default=None, help="Laya checkpoint dir (default runs/<name>/model)")
    ap.add_argument("--baseline", dest="baseline", action="store_true", default=None,
                    help="also re-measure laya-multilingual (downloads ~650 MB once)")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false")
    a = ap.parse_args(argv)
    evaluate(load_config(a.config, a.set), a.ckpt, a.baseline)


if __name__ == "__main__":
    main()
