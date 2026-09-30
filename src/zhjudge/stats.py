"""Uncertainty for eval reports (numpy only): how much of a difference between two numbers is noise.

  wilson(k, n)             95% Wilson interval of one accuracy (good for small n and for accuracies near 0 or 1)
  paired(rows_a, rows_b)   model A vs model B on the SAME decisions (matched by id): accuracy / Brier / NLL differences
                           with a 95% interval, and the exact McNemar p for accuracy. Decisions asked about one input
                           (the questions of a typed-decisions case) are resampled together (cluster-robust error).
  ece_floor(conf)          the ECE-15 a perfectly calibrated model with exactly these confidences still shows at this
                           n (mean over draws of correct ~ Bernoulli(conf)): an ECE near it is no evidence of
                           miscalibration
  fingerprint(rows)        sha256 of the sorted (id, gold[, rendering]) of a slice: equal fingerprints = the same items
  slices_from_id(id, ...)  the slices zhjudge.eval gives an item, for eval_preds rows written without them (v0)
The intervals cover item sampling only (which test items were drawn), never training-seed variance: one training run
per model cannot show how much a retrain would move a number.
"""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from .metrics import ece, pred_index

Z95 = 1.959964
_TEST_ID = re.compile(r"^([a-z0-9_]+)/(train|dev|test)/")
# probe id prefix (zh-<set>-..., en-<set>-...) -> family, as written by zhjudge.data.probes
PROBE_FAMILY = {"c3": "reading_mc", "dream": "reading_mc", "ocnli": "nli", "mnli": "nli", "tnews": "topic",
                "iflytek": "topic", "agnews": "topic", "chnsenti": "sentiment", "sst2": "sentiment", "amazon": "rating",
                "lcqmc": "paraphrase", "afqmc": "paraphrase", "qqp": "paraphrase"}


def wilson(k: int, n: int, z: float = Z95) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def item_scores(row: dict) -> tuple[int, float, float]:
    """(correct, one-hot Brier, NLL) of one eval_preds row, with zhjudge.metrics' definitions."""
    p = np.asarray(row["probs"], float)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
    g = int(row["gold"])
    return (int(pred_index(p, row["qtype"]) == g), float(((p - np.eye(len(p))[g]) ** 2).sum()),
            -math.log(max(float(p[g]), 1e-6)))


def score_table(rows) -> dict:
    """{id: item_scores(row)}: computed once, shared by every slice the row is in."""
    return {r["id"]: item_scores(r) for r in rows}


def cluster_of(item_id: str) -> str:
    """Decisions about the same input: a typed-decisions '<case>/<question>' -> '<case>'. Test ids
    ('<source>/<split>/<row>...') and probe ids ('...#<variant>') are single decisions."""
    if "#" in item_id or "/" not in item_id or _TEST_ID.match(item_id):
        return item_id
    return item_id.rpartition("/")[0]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts (A right and B wrong = b, the reverse = c): the binomial
    tail P(X <= min(b, c)), X ~ Bin(b + c, 1/2), summed in log space (fast and exact enough for any n)."""
    n = b + c
    if n == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(min(b, c) + 1)]
    top = max(logs)
    return float(min(1.0, 2 * math.exp(top) * sum(math.exp(x - top) for x in logs)))


def mean_ci(d, clusters=None):
    """Mean of per-item differences and its 95% interval (normal approximation; cluster-robust with clusters)."""
    d = np.asarray(d, float)
    n = len(d)
    m = float(d.mean()) if n else float("nan")
    if n < 2:
        return m, None
    if clusters is None or len(set(clusters)) == n:
        se = float(d.std(ddof=1) / math.sqrt(n))
    else:
        sums: dict[str, float] = {}
        for c, x in zip(clusters, d):
            sums[c] = sums.get(c, 0.0) + x - m
        g = len(sums)
        if g < 2:
            return m, None
        se = math.sqrt(g / (g - 1) * sum(s * s for s in sums.values())) / n
    return m, [round(m - Z95 * se, 4), round(m + Z95 * se, 4)]


def paired(rows_a, rows_b, sa: dict | None = None, sb: dict | None = None) -> dict:
    """A minus B on the decisions both have (matched by id; sa / sb: score_table()s to reuse). Positive d_acc = A more
    accurate; negative d_brier / d_nll = A better. gold_mismatch / input_mismatch count common ids whose gold or
    rendering differ (then the two sides did not score the same item)."""
    by_b = {r["id"]: r for r in rows_b}
    common = [(r, by_b[r["id"]]) for r in rows_a if r["id"] in by_b]
    out = {"n": len(common), "only_a": len(rows_a) - len(common), "only_b": len(by_b) - len(common)}
    if not common:
        return out
    sa, sb = sa or {}, sb or {}
    A = np.array([sa.get(a["id"]) or item_scores(a) for a, _ in common])
    B = np.array([sb.get(b["id"]) or item_scores(b) for _, b in common])
    clusters = [cluster_of(a["id"]) for a, _ in common]
    out.update(gold_mismatch=sum(int(a["gold"]) != int(b["gold"]) for a, b in common),
               input_mismatch=sum(a.get("render") != b.get("render") for a, b in common), clusters=len(set(clusters)),
               acc_a=round(float(A[:, 0].mean()), 4), acc_b=round(float(B[:, 0].mean()), 4))
    for j, name in enumerate(("acc", "brier", "nll")):
        m, ci = mean_ci(A[:, j] - B[:, j], clusters)
        out[f"d_{name}"], out[f"d_{name}_ci95"] = round(m, 4), ci
    b = int(((A[:, 0] == 1) & (B[:, 0] == 0)).sum())
    c = int(((A[:, 0] == 0) & (B[:, 0] == 1)).sum())
    out["a_right_b_wrong"], out["a_wrong_b_right"] = b, c
    # McNemar assumes independent decisions: not for clustered (typed-decisions) slices
    out["mcnemar_p"] = round(mcnemar_exact(b, c), 4) if out["clusters"] == len(common) else None
    return out


def fmt_d(p, key="d_acc", scale=100) -> str:
    """A paired difference in points ('+2.1 [-0.4, +4.6]'); * when its 95% interval excludes 0."""
    if not p or p.get(f"{key}_ci95") is None:
        return "-"
    lo, hi = p[f"{key}_ci95"]
    return f"{p[key] * scale:+.1f} [{lo * scale:+.1f}, {hi * scale:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"


def ece_floor(conf, bins=15, reps=100, seed=0) -> float:
    """Expected ECE-15 of a perfectly calibrated model with exactly these top-1 confidences."""
    conf = np.asarray(conf, float)
    if len(conf) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    return float(np.mean([ece(conf, rng.random(len(conf)) < conf, bins) for _ in range(reps)]))


def fingerprint(rows, salt: str = "") -> str:
    """Rows: dicts with id, gold and optionally render (a hash of a rendered input)."""
    h = hashlib.sha256(salt.encode())
    for line in sorted(f"{r['id']}\t{int(r['gold'])}\t{r.get('render') or ''}" for r in rows):
        h.update(line.encode() + b"\n")
    return h.hexdigest()[:16]


def slices_from_id(item_id: str, qtype: str, eval_only=frozenset()) -> list[str]:
    """The slices zhjudge.eval gives an item, recovered from its id (probe families from the id prefix; the workflow of a
    typed-decisions case is not in its id, so those rows get no family slice)."""
    base, _, variant = item_id.partition("#")
    m = _TEST_ID.match(base)
    if m:
        src = m.group(1)
        lang = "zh" if "_zh" in src else "en"
        if variant == "laya":
            return [f"test:laya:{k}" for k in ("ALL", f"lang:{lang}", f"source:{src}", f"qtype:{qtype}")]
        return ["test:ALL", f"test:lang:{lang}", f"test:source:{src}", f"test:qtype:{qtype}",
                f"test:lang:{lang}/qtype:{qtype}", f"test:group:{'eval-only' if src in eval_only else 'trained'}"]
    if variant:
        head = base.split("-")
        if head[0] == "xnli":
            nm = ("probe:xnli/en" if head[1] == "en" else
                  f"probe:xnli/zh/{'zh' if variant == 'native' else 'en'}-prompt")
            return [nm, f"{nm}/family:nli", f"{nm}/qtype:{qtype}"]
        if head[0] in ("zh", "en") and len(head) > 1:
            nm = "probe:en-control" if head[0] == "en" else f"probe:zh/{'zh' if variant == 'native' else 'en'}-prompt"
            return [nm, f"{nm}/family:{PROBE_FAMILY.get(head[1], 'unknown')}", f"{nm}/qtype:{qtype}"]
    return ["probe:typed-decisions", f"probe:typed-decisions/qtype:{qtype}"]
