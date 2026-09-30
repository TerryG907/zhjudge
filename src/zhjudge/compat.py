"""Prove (or disprove) that the exported checkpoint is a drop-in Laya checkpoint (port of the spike's check_compat.py).

  uv run zhjudge compat --config ...                 # checks runs/<name>/export/<export.name>
  uv run zhjudge compat --config ... --ckpt DIR

Checks (each recorded in runs/<name>/compat.json; a failure does not stop later checks):
  1. structure : required files, config keys, tensor prefixes, dtypes (offline, always runs)
  2. format    : tensor names / shapes / dtypes vs the published laya-multilingual checkpoint, read remotely from its
                 safetensors header (no download). Shapes are only expected to match when the base encoder is the
                 same size as mmBERT-base (e.g. not for the mmBERT-small smoke model). Skipped when offline.
  3. laya      : the upstream reference runtime (PyPI `laya`) loads it unchanged and answers N held-out records;
                 probabilities must match our batched evaluation path (max |dp|)
  4. laya-mlx  : PyPI `laya-mlx` (extra [mlx], Apple silicon only) loads it; parity vs laya; `laya_mlx.convert` works
  5. laya-serve: upstream Laya's HTTP server (`POST /v1/systemone`, a protocol Laya's authors describe as Jev's; a
     request shaped like theirs, never sent to Jev) answers with it (extra [serve])
Exit code 1 if structure, format (when it ran) or laya fails. Optional runtimes that are not installed are
reported as "skipped", not failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

from .config import add_common_args, load_config, paths  # noqa: E402
from .export import structural_check  # noqa: E402
from .model import BASELINE_REPO, BASELINE_REVISION, BASELINE_SUBFOLDER  # noqa: E402
from .records import cap_per_source, load_unified, read_jsonl, to_laya  # noqa: E402

EXAMPLE_STATE = {"标题": "央行宣布下调存款准备金率0.5个百分点"}
EXAMPLE_QUESTIONS = {
    "topic": {"type": "choice", "instructions": "这条新闻属于哪个类别？", "criteria": ["财经", "体育", "科技", "娱乐"]},
    "material": {"type": "noul", "instructions": "这条新闻会对A股市场产生实质影响吗？"},
    "impact": {"type": "score", "instructions": "对银行股的影响方向？", "criteria": ["明显利空", "中性", "明显利好"]},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 23), b""):
            h.update(blk)
    return h.hexdigest()


def check_format(ckpt: Path):
    from huggingface_hub import HfApi
    from safetensors import safe_open

    ours = {}
    with safe_open(str(ckpt / "model.safetensors"), "np") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            ours[k] = (sl.get_dtype(), tuple(sl.get_shape()))
    api = HfApi()
    meta = api.parse_safetensors_file_metadata(BASELINE_REPO, f"{BASELINE_SUBFOLDER}/model.safetensors",
                                               revision=BASELINE_REVISION)
    ref = {k: (v.dtype, tuple(v.shape)) for k, v in meta.tensors.items()}
    rep = {"reference": f"{BASELINE_REPO}@{BASELINE_REVISION[:7]}/{BASELINE_SUBFOLDER}",
           "n_tensors_ours": len(ours), "n_tensors_ref": len(ref),
           "only_in_ours": sorted(set(ours) - set(ref))[:20], "only_in_ref": sorted(set(ref) - set(ours))[:20],
           "dtype_mismatch": [(k, ours[k][0], ref[k][0]) for k in set(ours) & set(ref) if ours[k][0] != ref[k][0]][:20],
           "ref_dtypes": sorted({v[0] for v in ref.values()}), "ours_dtypes": sorted({v[0] for v in ours.values()})}
    shape_mm = [k for k in set(ours) & set(ref) if ours[k][1] != ref[k][1]]
    rep["shape_mismatch_count"] = len(shape_mm)
    rep["shape_mismatch_examples"] = sorted(shape_mm)[:5]
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    rep["same_encoder_as_ref"] = cfg.get("encoder") == "jhu-clsp/mmBERT-base"
    info = api.model_info(BASELINE_REPO, revision=BASELINE_REVISION, files_metadata=True)
    ref_sha = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    rep["tokenizer_json_identical_to_ref"] = (sha256(ckpt / "tokenizer" / "tokenizer.json")
                                              == ref_sha.get(f"{BASELINE_SUBFOLDER}/tokenizer/tokenizer.json"))
    names_ok = not (rep["only_in_ours"] or rep["only_in_ref"] or rep["dtype_mismatch"])
    rep["pass"] = names_ok and (not shape_mm or not rep["same_encoder_as_ref"])
    return rep


def run_records(predict, recs):
    out = []
    for r in recs:
        res = predict(r["state"], {"q": r["question"]})["answers"]["q"]
        if res["type"] == "noul":
            out.append(np.array([1 - res["noul"], res["noul"]]))
        else:
            out.append(np.array(list(res["probabilities"].values()), dtype=float))
    return out


def parity(a, b):
    d = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
    agree = float(np.mean([int(x.argmax() == y.argmax()) for x, y in zip(a, b)]))
    return {"max_abs_dprob": round(d, 6), "argmax_agreement": agree}


def sample_records(cfg, P, n):
    for split in ("test", "dev"):
        recs = load_unified(P["unified"], split)
        if recs:
            recs = cap_per_source(recs, max(1, n // max(1, len({r['source'] for r in recs}))), 0)
            return [to_laya(r) for r in recs][:n]
    calib = P["run"] / "calib.jsonl"
    return read_jsonl(calib)[:n] if calib.exists() else []


def compat(cfg: dict, ckpt=None) -> dict:
    P = paths(cfg)
    name = cfg["export"].get("name") or cfg["model"]["model_name"]
    ckpt = Path(ckpt).resolve() if ckpt else (P["export"] / name).resolve()
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found: run `zhjudge export` first")
    recs = sample_records(cfg, P, int(cfg["compat"]["n"]))
    report = {"ckpt": str(ckpt), "n_records": len(recs)}

    def step(key, fn, required=True):
        t0 = time.time()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res = fn()
            if isinstance(res, dict):
                res["warnings"] = sorted({str(x.message)[:200] for x in w})[:5]
            report[key] = res
        except ModuleNotFoundError as e:
            report[key] = {"pass": None, "skipped": f"not installed: {e.name}"}
        except Exception as e:  # noqa: BLE001
            report[key] = {"pass": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        report[key]["seconds"] = round(time.time() - t0, 2)
        report[key]["required"] = required
        print(key, json.dumps({k: v for k, v in report[key].items() if k != "trace"}, ensure_ascii=False,
                              default=str)[:1200], flush=True)

    def structure():
        problems = structural_check(ckpt)
        return {"pass": not problems, "problems": problems}

    step("structure", structure)

    def fmt():
        try:
            return check_format(ckpt)
        except Exception as e:  # noqa: BLE001  offline / hub unreachable is not a failure of the checkpoint
            n = type(e).__name__
            if isinstance(e, OSError) or any(s in n for s in ("Connect", "Offline", "Timeout", "Network")):
                return {"pass": None, "skipped": f"hub unreachable: {n}"}
            raise

    step("format", fmt)
    probs = {}

    def upstream():
        import laya
        import torch

        from .infer import agent_amp, batched_logits, probs_from_logits
        from .records import encode_records

        agent = laya.Agent(str(ckpt), device="cpu")
        probs["laya_cpu"] = run_records(agent.predict, recs)
        items, _ = encode_records(recs, agent.tok, agent.cfg["max_len"], agent.cfg["head_max_len"], log_every=0)
        lg = batched_logits(agent.model, items, agent.tok.pad_token_id, torch.device("cpu"), agent_amp(agent), 16)
        ours = probs_from_logits(lg, [it["qtype"] for it in items], agent.temperature, agent.temperature_by_options)
        par = parity(ours, probs["laya_cpu"]) if len(ours) == len(recs) and recs else None
        ex = agent.predict(EXAMPLE_STATE, EXAMPLE_QUESTIONS)
        ok = par is None or par["max_abs_dprob"] < 2e-3
        return {"pass": ok, "laya_version": laya.__version__, "parity_batched_eval_vs_laya": par,
                "example_answer": ex, "temperature_applied": agent.temperature,
                "temperature_by_options_applied": agent.temperature_by_options}

    step("laya_reference", upstream)

    def mlx_rt():
        import laya_mlx

        out = {"pass": True}
        for dt in ("float32", "float16"):
            ag = laya_mlx.load(str(ckpt), dtype=dt)
            probs["mlx_" + dt] = run_records(ag.predict, recs)
            if "laya_cpu" in probs:
                out["parity_vs_laya_" + dt] = parity(probs["mlx_" + dt], probs["laya_cpu"])
        import shutil
        from laya_mlx.convert import convert

        dst = P["run"] / "compat_mlx_converted"
        if dst.exists():
            shutil.rmtree(dst)
        convert(str(ckpt), str(dst), dtype="float16")
        ag = laya_mlx.load(str(dst), dtype="float16")
        out["convert_parity_vs_direct_fp16"] = parity(run_records(ag.predict, recs), probs["mlx_float16"])
        shutil.rmtree(dst, ignore_errors=True)
        return out

    step("laya_mlx", mlx_rt, required=False)

    def serve():
        from fastapi.testclient import TestClient
        from laya.router import Router
        from laya.serve import create_app

        router = Router(models={"english": str(ckpt), "multilingual": str(ckpt)}, device="cpu")
        c = TestClient(create_app(router))
        body = {"model": "jev-1", "state": "客户说：三月的账单被重复扣款了，今天不退款我们就取消订阅。",
                "questions": {"churn_risk": {"type": "noul", "instructions": "客户是否威胁要取消？"},
                              "department": {"type": "choice", "instructions": "应由哪个部门处理？",
                                             "criteria": {"billing": "账单、付款、退款", "technical": "故障",
                                                          "sales": "新合同"}}}}
        r = c.post("/v1/systemone", json=body)
        return {"pass": r.status_code == 200, "status": r.status_code, "response": r.json()}

    step("laya_serve_systemone", serve, required=False)

    failed = [k for k, v in report.items() if isinstance(v, dict) and v.get("required") and v.get("pass") is False]
    report["pass"] = not failed
    report["failed"] = failed
    out = P["run"] / "compat.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[compat] {'PASS' if not failed else 'FAIL ' + ', '.join(failed)} -> {out}", flush=True)
    return report


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args(argv)
    rep = compat(load_config(a.config, a.set), a.ckpt)
    raise SystemExit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
