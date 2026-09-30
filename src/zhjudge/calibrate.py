"""Fit temperatures on the held-out calibration slice and write them into the Laya checkpoint's rl_agent_config.json.

  uv run zhjudge calibrate --config ...

Per question type (choice / score / noul) like the spike, plus per (type, #options) bucket
(`laya.common.temp_bucket`: 2 / 3-5 / 6-10 / 11+ options) when the bucket has >= calibrate.min_bucket rows; the
Laya runtime uses a bucket temperature first and falls back to the per-type one. Every value is clamped to
[0.5, 5.0], the range all Laya runtimes accept. The calibration records (runs/<name>/calib.jsonl) were split off the
training data before training and never trained on; they are rendered like the training records (train.format_aug /
option_shuffle), so the temperatures fit the same mix of plain and Laya-style inputs. Fitting uses the fp16 weights
actually shipped. calibration.json also reports the effect out of sample (calib_crossfit: fitted on one half of the
slice, scored on the other, both ways).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import device as D  # noqa: E402
from .config import add_common_args, load_config, paths  # noqa: E402
from .infer import agent_amp, batched_logits, probs_from_logits  # noqa: E402
from .metrics import fit_temperature, summarize  # noqa: E402
from .model import load_agent  # noqa: E402
from .records import QTYPE_NAMES, encode_records, read_jsonl  # noqa: E402


def fit_temps(logits, items, min_b):
    """Temperatures per question type and per (type, #options) bucket with >= min_b rows."""
    from laya.common import temp_bucket

    # fit to the gold option, not to the label-smoothed training targets: the smoothed targets pull every
    # temperature above 1 (v0: [1.087, 1.201, 1.201], in-sample ECE 0.0350 -> 0.0354), while calibration is judged
    # against the gold label
    by_q, by_b = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
    for z, it in zip(logits, items):
        gold = np.eye(len(z))[it["gold"]]
        by_q[it["qtype"]][0].append(z)
        by_q[it["qtype"]][1].append(gold)
        b = by_b[temp_bucket(it["qtype"], len(z))]
        b[0].append(z)
        b[1].append(gold)
    temps = [round(fit_temperature(*by_q[q]), 4) if q in by_q else 1.0 for q in range(3)]
    by_opt = {k: round(fit_temperature(*v), 4) for k, v in sorted(by_b.items()) if len(v[0]) >= min_b}
    return temps, by_opt, by_q, by_b


def rows_of(probs, items):
    return [{"probs": p.tolist(), "gold": it["gold"], "qtype": QTYPE_NAMES[it["qtype"]]} for p, it in zip(probs, items)]


def crossfit(logits, items, min_b) -> dict:
    """Out-of-sample effect of the temperatures (report only): fit on one half of the calibration records (split by
    id hash), score the other half, and the reverse; metrics over all records, each scored with temperatures it was
    not fitted on. The in-sample numbers above always favour the fitted temperatures."""
    import hashlib

    fold = [int(hashlib.sha256(it["id"].encode()).hexdigest(), 16) % 2 for it in items]
    after = [None] * len(items)
    for f in (0, 1):
        fit = [j for j in range(len(items)) if fold[j] != f]
        held = [j for j in range(len(items)) if fold[j] == f]
        temps, by_opt, _, _ = fit_temps([logits[j] for j in fit], [items[j] for j in fit], min_b)
        for j, p in zip(held, probs_from_logits([logits[j] for j in held], [items[j]["qtype"] for j in held], temps,
                                                by_opt)):
            after[j] = p
    before = probs_from_logits(logits, [it["qtype"] for it in items])
    return {"before": summarize(rows_of(before, items), bootstrap=False),
            "after": summarize(rows_of(after, items), bootstrap=False)}


def calibrate(cfg: dict) -> dict:
    P = paths(cfg)
    D.set_threads(cfg["train"]["cpu_threads"])
    dev = D.select_device(cfg["train"]["device"])
    cfg_path = P["model"] / "rl_agent_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} not found: run `zhjudge train` first")
    agent = load_agent(P["model"], dev.type)
    recs = read_jsonl(P["run"] / "calib.jsonl")
    items, dropped = encode_records(recs, agent.tok, agent.cfg["max_len"], agent.cfg["head_max_len"], log_every=0)
    logits = batched_logits(agent.model, items, agent.tok.pad_token_id, agent.device, agent_amp(agent),
                            batch_size=cfg["calibrate"]["batch_size"])
    min_b = int(cfg["calibrate"]["min_bucket"])
    temps, by_opt, by_q, by_b = fit_temps(logits, items, min_b)
    qts = [it["qtype"] for it in items]
    before = summarize(rows_of(probs_from_logits(logits, qts), items), bootstrap=False)
    after = summarize(rows_of(probs_from_logits(logits, qts, temps, by_opt), items), bootstrap=False)
    acfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    acfg["temperature"], acfg["temperature_by_options"] = temps, by_opt
    acfg.setdefault("training", {})["calibration"] = {
        "records": len(items), "split": "held-out slice of the training sources (never trained on)",
        "per_type_rows": {QTYPE_NAMES[q]: len(v[0]) for q, v in sorted(by_q.items())},
        "bucket_rows": {k: len(v[0]) for k, v in sorted(by_b.items())}, "min_bucket": min_b}
    cfg_path.write_text(json.dumps(acfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rep = {"temperature": temps, "temperature_by_options": by_opt, "records": len(items), "dropped": dropped,
           "calib_in_sample_before": before, "calib_in_sample_after": after}
    (P["run"] / "calibration.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("temperature", "temperature_by_options", "records")}, ensure_ascii=False))
    print("calib (in-sample) ECE-15 %.4f -> %.4f, NLL %.4f -> %.4f" % (before["ece15"], after["ece15"],
                                                                      before["nll"], after["nll"]), flush=True)
    try:
        cv = rep["calib_crossfit"] = crossfit(logits, items, min_b)
        (P["run"] / "calibration.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("calib (2-fold cross-fitted, out of sample) ECE-15 %.4f -> %.4f, NLL %.4f -> %.4f, Brier %.4f -> %.4f" % (
            cv["before"]["ece15"], cv["after"]["ece15"], cv["before"]["nll"], cv["after"]["nll"], cv["before"]["brier"],
            cv["after"]["brier"]), flush=True)
    except Exception as e:  # noqa: BLE001  (a report: the temperatures above are written already)
        print(f"[calibrate] warning: cross-fitted report failed ({type(e).__name__}: {e})", flush=True)
    return rep


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args(argv)
    calibrate(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
