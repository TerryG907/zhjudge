"""Metrics, with the same definitions the spike's baseline scorer used (baselines/scripts/score.py), so our numbers
are comparable with the recorded laya-multilingual baseline.

  acc     argmax of the returned distribution == gold (noul: P(true) >= 0.5 -> true)
  brier   multi-class Brier vs the one-hot gold: sum_k (p_k - y_k)^2, range [0, 2]; uniform guessing = 1 - 1/K
  ece15   expected calibration error, 15 equal-width bins over top-1 confidence vs correctness
  nll     -log p(gold), p clipped at 1e-6
  score_mae  |E[level] - gold level| for score questions
  soft_brier (typed-decisions) sum_k (p_k - gold_soft_k)^2 over choice + noul
"""
from __future__ import annotations

import math

import numpy as np


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def pred_index(p, qtype):
    return int(p[1] >= 0.5) if qtype == "noul" else int(np.argmax(p))


def summarize(rows, bootstrap=True):
    """rows: dicts with probs (list), gold (int), qtype ('choice'|'score'|'noul'), optional gold_soft, gold_score."""
    if not rows:
        return None
    P = []
    for r in rows:
        p = np.asarray(r["probs"], float)
        P.append(p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p)))
    y = [r["gold"] for r in rows]
    corr = np.array([int(pred_index(p, r["qtype"]) == g) for p, r, g in zip(P, rows, y)])
    conf = np.array([p.max() for p in P])
    brier = np.array([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in zip(P, y)])
    nll = np.array([-math.log(max(p[g], 1e-6)) for p, g in zip(P, y)])
    k = np.array([len(p) for p in P])
    m = {"n": len(rows), "acc": round(float(corr.mean()), 4), "brier": round(float(brier.mean()), 4),
         "ece15": round(ece(conf, corr), 4), "nll": round(float(nll.mean()), 4),
         "mean_conf": round(float(conf.mean()), 4),
         "uni_acc": round(float((1 / k).mean()), 4), "uni_brier": round(float((1 - 1 / k).mean()), 4)}
    if bootstrap and len(rows) >= 20:
        rng = np.random.default_rng(0)
        boots = [corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(1000)]
        m["acc_ci95"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    sc = [(p, r) for p, r in zip(P, rows) if r["qtype"] == "score"]
    if sc:
        ev = [float(np.dot(np.arange(len(p)), p)) for p, _ in sc]
        gs = [r["gold_score"] if r.get("gold_score") is not None else r["gold"] for _, r in sc]
        m["score_mae"] = round(float(np.mean([abs(a - b) for a, b in zip(ev, gs)])), 4)
    soft = [(p, r) for p, r in zip(P, rows) if r.get("gold_soft") is not None and r["qtype"] in ("choice", "noul")]
    if soft:
        m["soft_brier"] = round(float(np.mean([((p - np.asarray(r["gold_soft"])) ** 2).sum() for p, r in soft])), 4)
    return m


def fit_temperature(logit_rows, target_rows, lo=0.5, hi=5.0):
    """Per-bucket temperature by minimising soft cross-entropy on held-out rows (grid + refine; numpy only).
    Clamped to the [0.5, 5.0] range every Laya runtime enforces at load (spike)."""
    if len(logit_rows) < 10:
        return 1.0

    def nll(t):
        tot = 0.0
        for z, y in zip(logit_rows, target_rows):
            z = np.asarray(z, dtype=np.float64) / t
            z = z - z.max()
            lp = z - np.log(np.exp(z).sum())
            tot -= float((np.asarray(y) * lp).sum())
        return tot / len(logit_rows)

    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 61))
    best = min(grid, key=nll)
    for _ in range(3):  # local refine
        fine = np.linspace(best / 1.08, best * 1.08, 21)
        best = min(fine, key=nll)
    return float(min(hi, max(lo, best)))
