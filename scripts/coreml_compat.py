# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "laya-coreml[convert,compare] @ git+https://github.com/mizorewww/laya-coreml@4619e0483f07adf39068532e85b42ec2347edb83",
# ]
# ///
"""Optional, Apple only: does an exported checkpoint run in the unmodified laya-coreml runtime
(git 4619e04 = v0.1.1; PyPI only has 0.1.0)? Ported from the phase-0 spike's coreml_compat.py; not yet run
against a checkpoint exported by this repo, so treat laya-coreml support as unverified until it passes.

Isolated env on purpose: laya-coreml[convert] pins torch==2.7.0 and numpy<2.2, which conflict with the training env
(torch 2.14), so this runs as a standalone uv script with its own dependencies. Steps (recorded in a JSON report):
  1. convert : `laya_coreml.convert.convert(ckpt, out)` with the CLI defaults (fp16, enumerated lengths up to the
               checkpoint's max_len, batch 1, 32 option slots)
  2. predict : `laya_coreml.load(out, compute_units=...)` answers N records
  3. parity  : per-record probability diff vs `laya_mlx.load(ckpt, dtype="float32")` (the `compare` extra pins
               laya-mlx 0.2.0), plus p50 latency per question

Run (from the repo root, after `zhjudge export`):
  uv run --script scripts/coreml_compat.py --ckpt runs/<name>/export/<export-name> --records runs/<name>/calib.jsonl
`--records` is any JSONL of Laya records ({"state": ..., "question": {...}} per line), e.g. the calibration slice.
"""
import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np


def probs_of(ans):
    if ans["type"] == "noul":
        return np.array([1 - ans["noul"], ans["noul"]])
    return np.array(list(ans["probabilities"].values()), dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--units", default="cpu_gpu", help="cpu | cpu_gpu | all | cpu_ne (laya-coreml names)")
    ap.add_argument("--records", required=True, help="JSONL of Laya records (state + question)")
    a = ap.parse_args()
    ckpt = Path(a.ckpt).resolve()
    out = ckpt.parent / "coreml_fp16"
    recs = [json.loads(l) for l in open(a.records, encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(a.n, len(recs)), replace=False))]
    rep = {"ckpt": str(ckpt), "n_records": len(recs), "compute_units": a.units}

    t0 = time.time()
    try:
        import coremltools
        import torch
        from laya_coreml.convert import convert
        if out.exists():
            shutil.rmtree(out)
        convert(str(ckpt), str(out), precision="float16")
        rep["convert"] = {"pass": True, "out": str(out), "seconds": round(time.time() - t0, 1),
                          "coremltools": coremltools.__version__, "torch": torch.__version__,
                          "mlpackage_mb": round(sum(p.stat().st_size for p in (out / "model.mlpackage").rglob("*")
                                                    if p.is_file()) / 2**20, 1)}
    except Exception as e:
        rep["convert"] = {"pass": False, "error": "%s: %s" % (type(e).__name__, e),
                          "trace": traceback.format_exc()[-2000:], "seconds": round(time.time() - t0, 1)}
    print("convert", json.dumps({k: v for k, v in rep["convert"].items() if k != "trace"}), flush=True)

    ref = []
    try:
        import laya_mlx
        ag = laya_mlx.load(str(ckpt), dtype="float32")
        ref = [probs_of(ag.predict(r["state"], {"q": r["question"]})["answers"]["q"]) for r in recs]
    except Exception as e:
        rep["laya_mlx_reference_error"] = "%s: %s" % (type(e).__name__, e)

    if rep["convert"]["pass"]:
        t0 = time.time()
        try:
            import laya_coreml
            cm = laya_coreml.load(str(out), compute_units=a.units)
            got, lat = [], []
            for r in recs:
                s = time.perf_counter()
                ans = cm.system_one(r["state"], {"q": r["question"]})["answers"]["q"]
                lat.append((time.perf_counter() - s) * 1000)
                got.append(probs_of(ans))
            res = {"pass": True, "latency_ms_p50_one_question": round(float(np.median(lat)), 2),
                   "latency_ms_p90": round(float(np.percentile(lat, 90)), 2)}
            if ref:
                res["parity_vs_laya_mlx_fp32"] = {
                    "max_abs_dprob": round(max(float(np.abs(x - y).max()) for x, y in zip(got, ref)), 5),
                    "mean_abs_dprob": round(float(np.mean([np.abs(x - y).mean() for x, y in zip(got, ref)])), 6),
                    "argmax_agreement": float(np.mean([x.argmax() == y.argmax() for x, y in zip(got, ref)]))}
            ex = cm.system_one({"标题": "央行宣布下调存款准备金率0.5个百分点"}, {
                "material": {"type": "noul", "instructions": "这条新闻会对A股市场产生实质影响吗？"},
                "impact": {"type": "score", "instructions": "对银行股的影响方向？",
                           "criteria": ["明显利空", "中性", "明显利好"]}})
            res["example_answer"] = ex
            rep["predict"] = res
        except Exception as e:
            rep["predict"] = {"pass": False, "error": "%s: %s" % (type(e).__name__, e),
                              "trace": traceback.format_exc()[-2000:]}
        rep["predict"]["seconds"] = round(time.time() - t0, 1)
        print("predict", json.dumps({k: v for k, v in rep["predict"].items() if k != "trace"},
                                    ensure_ascii=False)[:1500], flush=True)

    p = ckpt.parent / ("coreml_compat_%s.json" % a.units)
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
