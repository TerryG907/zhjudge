"""Train a Laya-format typed-decision model on the unified corpus (PyTorch; CUDA, MPS or CPU).

  uv run zhjudge train --config configs/train_base.yaml
  ZHJUDGE_DEVICE=cpu uv run zhjudge train --config configs/train_smoke.yaml
  uv run torchrun --nproc_per_node=2 -m zhjudge train --config configs/train_base.yaml   # optional: 2 GPUs (DDP)

Recipe = the spike's train_torch.py, which follows the upstream Laya Kaggle notebook: RLCD Gaussian policy gradient
on a strictly proper score (log + 0.75 spherical, minus RPS for score questions; a group of train.rl_group noisy
logit samples, 4 by default and in v1, sigma 0.4 -> 0.1) + soft cross-entropy + a small act-head term (act iff the
argmax is correct). AdamW with separate encoder/head learning rates, linear warmup + cosine decay, grad clip 1.0,
length-grouped token-capped batches.

v1b recipe keys (off by default and in v1; on in configs/train_v1b.yaml): train.rl_group 64;
train.format_aug_per_epoch: every epoch after the first gets a fresh rendering and option order of each training
record (epoch 0 as without it; the calibration records stay out in every epoch; one token cache per epoch);
train.format_variants: a share of the Laya-style renderings in another Laya question shape (zhjudge.data.views.variant);
data.train_sample (set at build time; train refuses a corpus built with other values).

Added for cloud runs: device auto-select, bf16 / fp16+GradScaler / fp32 by GPU generation, gradient checkpointing
(auto on < 20 GiB GPUs), resumable checkpoints (rerun the same command to continue), periodic dev eval, a wall-clock
budget per invocation (train.max_hours) that saves and exits cleanly, and a NaN guard.

Measurement and speed only (none of it changes what is trained; all in OPERATIONAL): dev results per source plus
their macro average; train.monitor_calib_per_source > 0 scores a per-source capped slice of the held-out calibration
records (every training source and question type, about half written Laya-style) at update 0, with every dev eval
and at the end, per source, type and rendering (fmt:laya / fmt:plain); every log line has |dL/dlogits| of the RL, CE
and act terms (glogit_*). A monitor that raises is logged once and switched off: monitors never stop training.
train.optimizer_impl (auto: fused AdamW on CUDA), train.pad_multiple (training batches only; same batch plan, same
logits; only the head's dropout draws follow the padded shape), train.memprobe (`zhjudge memprobe` in a separate
process before training: GPU peak without gradient checkpointing, report only).

Multi-GPU (optional): under torchrun each process takes every WORLD_SIZE-th batch of the same batch plan, so one
optimizer update sees WORLD_SIZE batches (learning rates unchanged; updates per epoch divided by WORLD_SIZE). Rank 0
tokenizes first, logs, runs dev eval and writes every file. A checkpoint can be resumed with a different number of
processes (the position in the batch plan and the learning-rate schedule carry over).

Outputs in runs/<name>/:
  model/                 Laya-format checkpoint (fp16, temperatures 1.0 until `zhjudge calibrate`)
  ckpt/last.pt           resumable state (fp32 weights + optimizer; large: ~4 GB for mmBERT-base)
  ckpt/last.json         small progress summary of last.pt (step, updates, world size, why it was saved)
  calib.jsonl            held-out calibration records (never trained on)
  memprobe.json          train.memprobe on CUDA (zhjudge.memprobe)
  train_log.jsonl, dev_log.jsonl (with calib_slice), train_stats.json, config.resolved.yaml
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import device as D  # noqa: E402
from .config import add_common_args, config_hash, load_config, paths, save_resolved  # noqa: E402
from .data.allowlist import is_enabled, load_allowlist  # noqa: E402
from .infer import batched_logits  # noqa: E402
from .model import build_from_base, write_checkpoint  # noqa: E402
from .data.views import VIEWS_VERSION  # noqa: E402
from .records import (QTYPE_NAMES, augment, cached_items, cap_per_source, cap_total, collate_np,  # noqa: E402
                      encode_records, length_grouped_batches, load_unified, option_fit, option_fit_lines,
                      split_calibration, to_laya, write_jsonl)

EXIT_BUDGET = 75  # "temporary failure": the time budget ran out, state saved, rerun to resume
# keys that do not change what is trained (may differ between a run and its resume)
OPERATIONAL = {"device", "precision", "cpu_threads", "max_hours", "log_every", "eval_every", "save_every", "resume",
               "max_bad_steps", "keep_last_ckpt", "optimizer_impl", "pad_multiple", "monitor_calib_per_source",
               "memprobe"}
# behaviour keys added after v1 started, per config section, at the value that keeps v1's behaviour: left out of the
# resume hash while they hold it, so a v1 checkpoint still resumes under code that knows them
NEUTRAL = {"data": {"train_sample": {}}, "train": {"format_aug_per_epoch": False, "format_variants": 0}}
# rank 0 tokenizes (and saves checkpoints) while the other ranks wait in a collective
DIST_TIMEOUT = datetime.timedelta(hours=1)
MEMPROBE_TIMEOUT = 900  # s; under DDP the other ranks wait for it (and rank 0's tokenizing) in that collective


def log(msg):
    print(msg, flush=True)


def resume_hash(cfg: dict) -> str:
    """Hash of everything that decides what is trained: a checkpoint resumes only under the same hash."""
    sub = {**cfg, "train": {k: v for k, v in cfg["train"].items() if k not in OPERATIONAL}}
    for sec, neutral in NEUTRAL.items():
        if isinstance(sub.get(sec), dict):
            sub[sec] = {k: v for k, v in sub[sec].items() if not (k in neutral and v == neutral[k])}
    return config_hash(sub)


def dist_env() -> tuple[int, int, int]:
    """(world_size, rank, local_rank) set by torchrun; (1, 0, 0) for a plain single-process run."""
    world = int(os.environ.get("WORLD_SIZE") or 1)
    if world <= 1:
        return 1, 0, 0
    rank = int(os.environ["RANK"])
    return world, rank, int(os.environ.get("LOCAL_RANK") or rank)


def any_rank(flag: bool, world: int, dev) -> bool:
    """True if `flag` is true on any process (every process must call this at the same point)."""
    if world == 1:
        return flag
    import torch.distributed as dist

    x = torch.tensor([1.0 if flag else 0.0], device=dev)
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return bool(x.item() > 0)


def barrier(world: int, dev):
    if world > 1:
        import torch.distributed as dist

        dist.barrier(device_ids=[dev.index] if dev.type == "cuda" else None)


def trainable_sources(cfg, P):
    man_path = P["unified"] / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"{man_path} not found: run `zhjudge build` first")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    allow = load_allowlist(P["allowlist"])
    smoke_ok = manifest.get("allowlist", {}).get("smoke_allow_disabled") and cfg["run"].get("smoke")
    srcs = []
    for name, info in manifest["sources"].items():
        if info.get("eval_only") or "train" not in info.get("files", {}):
            continue
        if not is_enabled(allow, name) and not smoke_ok:
            raise SystemExit(f"{name} is in data/unified but not enabled in the allowlist; rebuild the corpus")
        srcs.append(name)
    if not srcs:
        raise SystemExit("no trainable sources in data/unified (all eval-only or none enabled)")
    want, have = cfg["data"].get("train_sample") or {}, manifest.get("train_sample") or {}
    if want != have:
        raise SystemExit(f"{P['unified']} was built with data.train_sample={have}, but the config asks for {want}: "
                         f"rebuild the corpus with `zhjudge build` and the same config")
    return sorted(srcs), manifest


def augment_all(recs, t, seed):
    """augment() every record; a rendering never changes an id (the calibration records are excluded by id)."""
    out = [augment(r, t, seed) for r in recs]
    assert all(a["id"] == r["id"] for a, r in zip(out, recs)), "augment changed a record id"
    return out


def prepare(cfg, P, tok, seed, write=True, report=None):
    """Select, convert and tokenize train / calibration / dev data (deterministic for a given corpus + config).
    Returns train_items as one item list per rendering: one, or one per epoch with train.format_aug_per_epoch.
    report: a dict that gets report["option_fit"] (records.option_fit of train / dev / calib; report only)."""
    t = cfg["train"]
    m = cfg["model"]
    srcs, manifest = trainable_sources(cfg, P)
    recs = load_unified(P["unified"], "train", set(srcs))
    recs = cap_per_source(recs, t["max_train_per_source"], seed)
    recs = cap_total(recs, t["max_train_records"], seed)
    if float(t.get("format_aug") or 0) and not any(r.get("view") for r in recs):
        raise SystemExit(f"train.format_aug={t['format_aug']} but {P['unified']} has no record views (built by older "
                         f"code): rebuild the corpus with `zhjudge build`")
    base = recs
    recs = augment_all(base, t, seed)
    n_struct = sum(isinstance(r.get("context"), dict) for r in recs)
    if n_struct and write:
        print(f"[train] {n_struct}/{len(recs)} training records written Laya-style (train.format_aug)", flush=True)
    laya_recs = [to_laya(r, t["label_smoothing"], t["score_neighbor_mass"]) for r in recs]
    # records of one example (the unified `group`) go to the calibration slice together
    groups = [r.get("group") or r["id"] for r in recs] if any("group" in r for r in recs) else None
    train_recs, calib_recs = split_calibration(laya_recs, t["calib_frac"], t["calib_cap"], t["calib_min"], seed,
                                               groups)
    if write:
        write_jsonl(P["run"] / "calib.jsonl", calib_recs)
    dev = load_unified(P["unified"], "dev", set(srcs))
    dev = cap_total(dev, t["dev_max_records"], seed)
    dev_recs = [to_laya(r, t["label_smoothing"], t["score_neighbor_mass"]) for r in dev]
    key = hashlib.sha256(json.dumps({
        "corpus": manifest.get("corpus_sha256"), "sources": srcs, "base": m["base"], "rev": m.get("revision"),
        "max_len": m["max_len"], "head_max_len": m["head_max_len"],
        "sel": [t["max_train_per_source"], t["max_train_records"], t["calib_frac"], t["calib_cap"], t["calib_min"],
                t["dev_max_records"], t["label_smoothing"], t["score_neighbor_mass"], seed],
        "aug": [t.get("format_aug"), t.get("format_cross_lingual"), t.get("format_bare_labels"), t.get("option_shuffle"),
                VIEWS_VERSION] + ([float(t["format_variants"])] if float(t.get("format_variants") or 0) else []),
    }, sort_keys=True).encode()).hexdigest()
    cache = P["run"] / "cache"
    train_items = [cached_items(cache / "train_items.pkl", train_recs, tok, m["max_len"], m["head_max_len"], key)]
    if t.get("format_aug_per_epoch") and (float(t.get("format_aug") or 0) or float(t.get("option_shuffle") or 0)):
        # epochs 1.. get a fresh rendering and option order of the same training records (epoch 0 is the one above)
        assert len({r["id"] for r in base}) == len(base), "training record ids are not unique"
        calib_ids = {r["id"] for r in calib_recs}
        keep = [r for r in base if r["id"] not in calib_ids]
        for ep in range(1, int(t["epochs"])):
            recs_ep = [to_laya(r, t["label_smoothing"], t["score_neighbor_mass"])
                       for r in augment_all(keep, t, f"{seed}:ep{ep}")]
            train_items.append(cached_items(cache / f"train_items_ep{ep}.pkl", recs_ep, tok, m["max_len"],
                                            m["head_max_len"], f"{key}:ep{ep}"))
        if write:
            print(f"[train] train.format_aug_per_epoch: {len(train_items)} renderings of the {len(keep)} training "
                  f"records, one per epoch", flush=True)
    dev_items = cached_items(cache / "dev_items.pkl", dev_recs, tok, m["max_len"], m["head_max_len"], key) if dev_recs else []
    if report is not None:
        try:  # which options / instructions build_sequence cut (the inputs stay what they are)
            calib_items, _ = encode_records(calib_recs, tok, m["max_len"], m["head_max_len"], log_every=0)
            parts = (("train", train_items[0], train_recs), ("dev", dev_items, dev_recs),
                     ("calib", calib_items, calib_recs))
            report["option_fit"] = {"head_max_len": m["head_max_len"],
                                    **{k: option_fit(i, r, tok, m["head_max_len"]) for k, i, r in parts}}
        except Exception as e:  # noqa: BLE001  (a report never stops training)
            report["option_fit"] = {"error": f"{type(e).__name__}: {e}"}
    return train_items, calib_recs, dev_items, srcs, manifest, key


def dev_eval(model, items, pad_id, dev, amp, bs, max_tokens=0, fmt=False):
    """acc / mean CE (vs the training target): ALL, per question type, language, source (source:<name>) and, with
    fmt, rendering (fmt:laya / fmt:plain); macro_src = the unweighted mean over sources."""
    if not items:
        return {}
    logits = batched_logits(model, items, pad_id, dev, amp, batch_size=bs, max_tokens=max_tokens)
    by = {}
    for z, it in zip(logits, items):
        z = z - z.max()
        lp = z - np.log(np.exp(z).sum())
        ce = -float((np.asarray(it["target"]) * lp).sum())
        acc = float(int(np.argmax(z)) == it["gold"])
        for key in ("ALL", QTYPE_NAMES[it["qtype"]], it["lang"], "source:" + it["source"]) + (
                ("fmt:" + it["fmt"],) if fmt else ()):
            d = by.setdefault(key, {"n": 0, "ce": 0.0, "acc": 0.0})
            d["n"] += 1
            d["ce"] += ce
            d["acc"] += acc
    keys = [k for k in by if ":" not in k] + sorted(k for k in by if ":" in k)  # ALL, types, languages first (v0)
    out = {k: {"n": by[k]["n"], "ce": round(by[k]["ce"] / by[k]["n"], 4), "acc": round(by[k]["acc"] / by[k]["n"], 4)}
           for k in keys}
    src = [v for k, v in by.items() if k.startswith("source:")]
    out["macro_src"] = {"sources": len(src), "ce": round(sum(v["ce"] / v["n"] for v in src) / len(src), 4),
                        "acc": round(sum(v["acc"] / v["n"] for v in src) / len(src), 4)}
    return out


def monitor_line(d) -> str:
    """One-line summary of a dev_eval() result."""
    def acc(k):
        return f"{d[k]['acc']:.3f}" if k in d else "-"

    return (f"acc {acc('ALL')} ce {d['ALL']['ce']:.3f} | " + (f"laya {acc('fmt:laya')} plain {acc('fmt:plain')} | "
            if any(k.startswith("fmt:") for k in d) else "") + " ".join(f"{q} {acc(q)}" for q in QTYPE_NAMES.values())
            + f" | macro over {d['macro_src']['sources']} sources {d['macro_src']['acc']:.3f}")


def adamw_kwargs(impl: str, dev) -> dict:
    """train.optimizer_impl -> torch.optim.AdamW kwargs. fused: one kernel per step; GradScaler hands it found_inf, so
    a skipped fp16 step leaves the weights, the moments and the step count unchanged (as the unfused step does)."""
    if impl not in ("auto", "fused", "foreach", "default"):
        raise SystemExit(f"train.optimizer_impl must be auto | fused | foreach | default, not {impl!r}")
    if impl == "fused" or (impl == "auto" and dev.type == "cuda"):
        return {"fused": True}
    return {"foreach": True} if impl == "foreach" else {}


def run_memprobe(P, chash, dev):
    """`zhjudge memprobe` in a separate process (an OOM or a sticky CUDA error there never reaches training). Returns
    memprobe.json; a failure or a timeout is logged and recorded there instead. A result or a failure is kept for
    later resumes with the same config and GPU (not measured, or waited for, again)."""
    out = P["run"] / "memprobe.json"
    t0 = time.time()
    who = {"config_hash": chash, "device": D.describe(dev)}
    try:
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
        if all(old.get(k) == v for k, v in who.items()):
            log(f"[train] memprobe: {out} is from this config and GPU ({old.get('error') or 'ok'}); not run again")
            return old
        out.unlink(missing_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")}
        cmd = [sys.executable, "-m", "zhjudge", "memprobe", "--config", str(P["run"] / "config.resolved.yaml")]
        log(f"[train] memprobe: {' '.join(cmd[1:])} (separate process, timeout {MEMPROBE_TIMEOUT} s, report only)")
        rc = subprocess.run(cmd, env=env, timeout=MEMPROBE_TIMEOUT).returncode
        if out.exists():
            log(f"[train] memprobe: done in {time.time() - t0:.0f} s -> {out}")
            return json.loads(out.read_text(encoding="utf-8"))
        err = f"exit {rc}, no result"
    except subprocess.TimeoutExpired:  # the child has been killed
        err = f"timeout: no result within {MEMPROBE_TIMEOUT} s"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    log(f"[train] memprobe failed ({err}) after {time.time() - t0:.0f} s; training is unaffected")
    res = {**who, "error": err}
    with contextlib.suppress(Exception):
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def epoch_items(train_items, ep):
    """The items of epoch `ep` (train_items: one list per rendering, see prepare)."""
    return train_items[ep % len(train_items)]


def plan_batches(train_items, t, seed):
    """(epoch, indices into that epoch's items) for every batch of the run."""
    plan = []
    for ep in range(t["epochs"]):
        plan += [(ep, b) for b in length_grouped_batches(epoch_items(train_items, ep), t["batch_size"], seed + ep,
                                                         max_tokens=t["max_tokens"])]
    if t["max_steps"]:
        plan = (plan * (t["max_steps"] // max(1, len(plan)) + 1))[: t["max_steps"]]
    return plan


def save_state(path: Path, state: dict, progress: dict | None = None):
    """Write the resumable state atomically, then ckpt/last.json (a small summary tools can read without torch)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    if progress is not None:
        side = path.with_suffix(".json")
        tmp = side.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(progress, indent=1), encoding="utf-8")
        os.replace(tmp, side)


def train(cfg: dict, force=False) -> dict:
    t_start = time.time()
    P = paths(cfg)
    t, m = cfg["train"], cfg["model"]
    seed = int(cfg["run"]["seed"])
    world, rank, local_rank = dist_env()
    main = rank == 0
    if main:
        P["run"].mkdir(parents=True, exist_ok=True)
        save_resolved(cfg, P["run"])
    chash = resume_hash(cfg)
    done_path = P["run"] / "train_done.json"
    if done_path.exists() and not force and (P["model"] / "model.safetensors").exists():
        try:
            done = json.loads(done_path.read_text())
        except FileNotFoundError:  # DDP: rank 0 deletes a stale one below while another rank checks it
            done = {}
        if done.get("config_hash") == chash:
            if main:
                log(f"[train] {P['model']} already trained with this config (hash {chash}); skipping (use --force)")
            return done
    if main:
        done_path.unlink(missing_ok=True)  # stale: written for another config, or retraining with --force
    D.set_threads(t["cpu_threads"])
    dev = D.select_device(t["device"])
    if world > 1:
        import torch.distributed as dist

        if dev.type == "cuda":
            torch.cuda.set_device(local_rank)
            dev = torch.device("cuda", local_rank)
        elif dev.type != "cpu":
            raise SystemExit(f"multi-process training (WORLD_SIZE={world}) needs CUDA GPUs (or CPU), not {dev.type}")
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo", timeout=DIST_TIMEOUT)
    precision = D.select_precision(dev, t["precision"])
    amp = D.amp_dtype(precision)
    gckpt = D.want_grad_checkpointing(t["grad_checkpointing"], dev)
    pad = int(t["pad_multiple"])
    if pad < 1:
        raise SystemExit(f"train.pad_multiple must be >= 1, not {pad}")
    if main:
        log(f"[train] device {D.describe(dev)} | precision {precision} | grad checkpointing {gckpt}"
            + (f" | {world} processes (DDP)" if world > 1 else ""))
    probe = None
    if main and t.get("memprobe"):  # rank 0, before any model exists here (the other ranks wait in the barrier below)
        if dev.type == "cuda":
            probe = run_memprobe(P, chash, dev)
        else:
            log("[train] memprobe: skipped (no CUDA)")

    torch.manual_seed(seed)
    if world > 1 and not main:
        barrier(world, dev)  # rank 0 first: it downloads the base model and writes the tokenizer and caches
    model, tok, acfg, ecfg = build_from_base(cfg, P["run"], seed)
    pad_id = tok.pad_token_id
    fit = {} if main else None
    train_items, calib_recs, dev_items, srcs, manifest, data_key = prepare(cfg, P, tok, seed, write=main, report=fit)
    mon_items, n_mon = [], int(t.get("monitor_calib_per_source") or 0)
    if main and n_mon and calib_recs:
        try:  # held out like calib.jsonl: every source and question type, about half written Laya-style
            mon_items = cached_items(P["run"] / "cache" / "monitor_items.pkl", cap_per_source(calib_recs, n_mon, seed),
                                     tok, m["max_len"], m["head_max_len"], f"{data_key}:monitor:{n_mon}")
            log(f"[monitor] calibration slice: {len(mon_items)} records (<= {n_mon} per source, "
                f"{sum(it['fmt'] == 'laya' for it in mon_items)} written Laya-style), scored before training, with "
                f"every dev eval and at the end (dev_log.jsonl calib_slice)")
        except Exception as e:  # noqa: BLE001  (a monitor never stops training)
            mon_items = []
            log(f"[monitor] calibration slice failed ({type(e).__name__}: {e}); switched off, training continues")
    if world > 1 and main:
        barrier(world, dev)
    if main:
        try:
            of = fit["option_fit"]
            if "error" in of:
                raise RuntimeError(of["error"])
            log(f"[train] how the options fitted into head_max_len {m['head_max_len']} (report only; train_stats.json "
                f"option_fit), training items:\n" + "\n".join(option_fit_lines(of["train"], m["head_max_len"]))
                + "".join(f"\n  {k}: {sum(bool(v['options_cut'] or v['options_colliding']) for v in of[k].values())} of "
                          f"{len(of[k])} sources with cut or identical options" for k in ("dev", "calib") if of[k]))
        except Exception as e:  # noqa: BLE001  (a report never stops training)
            log(f"[train] option-fit report failed ({type(e).__name__}: {e}); training is unaffected")
    if gckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    if t["freeze_embeddings"]:
        model.encoder.embeddings.tok_embeddings.weight.requires_grad_(False)
    model.to(dev).train()

    enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": t["lr_encoder"]}, {"params": head_p, "lr": t["lr_head"]}],
                            weight_decay=t["weight_decay"], **adamw_kwargs(str(t["optimizer_impl"]), dev))
    plan = plan_batches(train_items, t, seed)
    plan_hash = hashlib.sha256(json.dumps(plan).encode()).hexdigest()[:16]
    n_plan = len(plan)
    accum = max(1, int(t["grad_accum"]))
    total = -(-n_plan // world)  # micro-steps; each process takes one batch of the plan per micro-step
    total_updates = math.ceil(total / accum)
    warm = max(1, int(t["warmup"] * total_updates))

    def lr_lambda(s):
        return s / warm if s < warm else 0.5 * (
            1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_updates - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and dev.type == "cuda"))

    ckpt_path = P["ckpt"] / "last.pt"
    step0, updates, elapsed_before = 0, 0, 0.0
    counters = {"seen_ex": 0, "seen_real": 0, "seen_pad": 0}  # summed over processes
    if ckpt_path.exists() and t["resume"]:
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if st.get("plan_hash") != plan_hash or st.get("config_hash") != chash:
            raise SystemExit(f"{ckpt_path} was written for a different config/data plan; delete it or set "
                             f"train.resume: false to start over")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        st_world = int(st.get("world_size", 1))
        if st_world == world:
            sched.load_state_dict(st["scheduler"])
            step0, updates = st["step"], st["updates"]
        else:
            # different number of processes: continue at the same place in the batch plan and the same point of
            # the learning-rate schedule (as a fraction of this run's number of updates)
            done_batches = int(st.get("batches_done", st["step"] * st_world))
            step0 = done_batches // world // accum * accum
            updates = step0 // accum
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=updates - 1)
        if main:
            torch.set_rng_state(st["rng_cpu"])
            if dev.type == "cuda" and st.get("rng_cuda") is not None:
                with contextlib.suppress(Exception):
                    torch.cuda.set_rng_state_all(st["rng_cuda"])
        elapsed_before = st.get("elapsed_s", 0.0)
        counters = st.get("counters", counters)
        if main:
            log(f"[train] resumed from {ckpt_path} at micro-step {step0}/{total} (update {updates})"
                + (f"; saved with {st_world} process(es), now {world}" if st_world != world else ""))
        del st
    # a resumed run keeps the implementation its checkpoint was written with (load_state_dict restores the flags, not the
    # constructor's GradScaler flag: an fp16 step would hand found_inf to a restored foreach/default kernel, which fails)
    g0 = opt.param_groups[0]
    opt._step_supports_amp_scaling = bool(g0.get("fused"))
    opt_impl = "fused" if g0.get("fused") else "foreach" if g0.get("foreach") else "default"
    if world > 1 and not (main and step0):
        torch.manual_seed(seed + 7919 * rank + step0)  # distinct dropout noise per process
    fwd = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        fwd = DDP(model, device_ids=[local_rank] if dev.type == "cuda" else None)
    n_params = sum(p.numel() for p in model.parameters())
    if main:
        log("[train] params %.1fM | sources %d | train %d calib %d dev %d | micro-steps %d x %d process(es) "
            "(accum %d, updates %d) | bs<=%d max_tokens %d | pad to %d | AdamW %s | RL group %d" % (
                n_params / 1e6, len(srcs), len(train_items[0]), len(calib_recs), len(dev_items), total, world, accum,
                total_updates, t["batch_size"], t["max_tokens"], pad, opt_impl, int(t["rl_group"])))

    from laya.common import proper_reward

    G, W_ACT = int(t["rl_group"]), float(t["act_weight"])
    s0, s1 = float(t["rl_sigma_start"]), float(t["rl_sigma_end"])
    budget_s = float(t.get("max_hours") or 0) * 3600
    log_f = open(P["run"] / "train_log.jsonl", "a", encoding="utf-8") if main else None
    dev_f = open(P["run"] / "dev_log.jsonl", "a", encoding="utf-8") if main else None
    recent, bad = [], 0
    t_loop = time.time()
    base_counters, local = dict(counters), {k: 0 for k in counters}
    off = set()  # monitors switched off after an exception

    def guard(name, fn):
        """Run monitor `name` (rank 0 only, no collectives); an exception is logged once and switches it off."""
        if name in off:
            return None
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (a monitor never stops training)
            off.add(name)
            err = f"{type(e).__name__}: {e}"
        model.train()  # an eval that raised half-way would otherwise leave dropout off
        if dev.type == "cuda":
            torch.cuda.empty_cache()  # after the traceback (and the tensors it holds) is gone
        log(f"[monitor] {name} failed ({err}); switched off, training continues")
        return None

    def evaluate(update, final=False):
        """Dev eval + the calibration-slice monitor (rank 0): one dev_log.jsonl row, (dev, calib slice) back."""
        bs = cfg["calibrate"]["batch_size"]
        d = guard("dev", lambda: dev_eval(model, dev_items, pad_id, dev, amp, bs)) if dev_items else None
        c = guard("calib_slice", lambda: dev_eval(model, mon_items, pad_id, dev, amp, bs, t["max_tokens"], fmt=True)
                  ) if mon_items else None
        dev_f.write(json.dumps({"update": update, **({"final": True} if final else {}), **(d or {}),
                                **({"calib_slice": c} if c else {})}) + "\n")
        dev_f.flush()
        when = "before training" if update == 0 and not final else "final" if final else f"update {update}"
        if d:
            log(f"[dev] {when}: {json.dumps(d.get('ALL', {}))} | macro over {d['macro_src']['sources']} sources "
                f"acc {d['macro_src']['acc']:.4f}")
        if c:
            log(f"[monitor] {when}: calibration slice {monitor_line(c)}")
        return d or {}, c

    def global_counters():
        """Counters summed over processes (a collective: every process calls it at the same point)."""
        if world == 1:
            return {k: base_counters[k] + local[k] for k in local}
        x = torch.tensor([local[k] for k in local], dtype=torch.float64, device=dev)
        dist.all_reduce(x)
        return {k: base_counters[k] + int(v) for k, v in zip(local, x.tolist())}

    def state(step, cnt):
        return {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None, "step": step, "updates": updates,
                "plan_hash": plan_hash, "config_hash": chash, "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                "elapsed_s": elapsed_before + time.time() - t_loop, "counters": cnt, "world_size": world,
                "batches_done": min(n_plan, step * world)}

    def progress(step, reason):
        return {"run": cfg["run"]["name"], "reason": reason, "step": step, "total_steps": total,
                "batches_done": min(n_plan, step * world), "total_batches": n_plan, "updates": updates,
                "total_updates": total_updates, "world_size": world, "epochs": t["epochs"],
                "train_elapsed_h": round((elapsed_before + time.time() - t_loop) / 3600, 3),
                "plan_hash": plan_hash, "config_hash": chash,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "saved_unix": round(time.time(), 1)}

    if main and step0 == 0 and (dev_items or mon_items):
        evaluate(0)
    opt.zero_grad(set_to_none=True)
    for step in range(step0, total):
        pi = step * world + rank  # this process's batch; the last micro-step wraps around to fill every process
        ep, bidx = plan[pi % n_plan]
        is_update = (step + 1) % accum == 0 or step == total - 1
        t0 = time.time()
        b = collate_np([epoch_items(train_items, ep)[i] for i in bidx], pad_id, pad)
        x = {k: torch.from_numpy(b[k]).to(dev) for k in ("input_ids", "attention_mask", "marker_pos",
                                                           "marker_mask", "qtype", "target")}
        sigma = s0 + (s1 - s0) * step / max(1, total - 1)
        # DDP: no gradient all-reduce on micro-steps that only accumulate
        with fwd.no_sync() if world > 1 and not is_update else contextlib.nullcontext():
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                logits, act = fwd(x["input_ids"], x["attention_mask"], x["marker_pos"], x["marker_mask"], x["qtype"])
            logits, act = logits.float(), act.float()
            mask, target = x["marker_mask"], x["target"]
            k = mask.sum(-1, keepdim=True).float()
            g = torch.Generator().manual_seed(seed * 1_000_003 + pi)
            eps = torch.randn((G,) + tuple(logits.shape), generator=g).to(dev) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), x["qtype"], mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
            correct = logits.detach().argmax(-1) == target.argmax(-1)
            loss_act = torch.nn.functional.cross_entropy(act, (~correct).long())
            loss = t["rl_weight"] * loss_rl + loss_ce + W_ACT * loss_act
            # every process skips the batch together (DDP would otherwise wait for a backward that never comes)
            if any_rank(not bool(torch.isfinite(loss)), world, dev):
                bad += 1
                opt.zero_grad(set_to_none=True)
                if main:
                    log(f"[train] non-finite loss at micro-step {step} ({bad} in a row); batch skipped")
                if bad >= t["max_bad_steps"]:
                    raise SystemExit("too many non-finite losses. On a T4/V100 (fp16) try ZHJUDGE_PRECISION=fp32, "
                                     "or lower train.lr_encoder")
                continue
            bad = 0
            show = main and (step % t["log_every"] == 0 or step == total - 1)
            # |dL/dz| of each weighted loss term at the head's outputs (the graph above them only; no parameter grads)
            gl = guard("glogit", lambda: [0.0 if g_ is None else float(g_.norm()) for v, y in (
                (t["rl_weight"] * loss_rl, logits), (loss_ce, logits), (W_ACT * loss_act, act))
                for g_ in torch.autograd.grad(v, y, retain_graph=True, allow_unused=True)]) if show else None
            scaler.scale(loss / accum).backward()
        gv = float("nan")
        if is_update:
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            updates += 1
            gv = float(gnorm)
        lv, cv, rv = float(loss.detach()), float(loss_ce.detach()), float(r.mean())
        dt = time.time() - t0
        local["seen_ex"] += len(bidx)
        local["seen_real"] += b["real_tokens"]
        local["seen_pad"] += b["padded_tokens"]
        recent = (recent + [(dt, len(bidx), b["real_tokens"])])[-200:]
        if main:
            rec = {"step": step, "update": updates, "epoch": ep, "dt": round(dt, 4), "n": len(bidx),
                   "real_tokens": b["real_tokens"], "padded_tokens": b["padded_tokens"], "loss": round(lv, 5),
                   "ce": round(cv, 5), "reward": round(rv, 5), "gnorm": round(gv, 4) if math.isfinite(gv) else None,
                   "lr_enc": sched.get_last_lr()[0], "sigma": round(sigma, 4),
                   "qtypes": np.bincount(b["qtype"], minlength=3).tolist()}
            if gl:
                rec["glogit_rl"], rec["glogit_ce"], rec["glogit_act"] = (round(v, 6) for v in gl)
            log_f.write(json.dumps(rec) + "\n")
        if show:
            st_dt = sum(x_[0] for x_ in recent)
            ex_s = world * sum(x_[1] for x_ in recent) / max(st_dt, 1e-9)
            eta_h = (total - step - 1) * (st_dt / len(recent)) / 3600
            log("step %6d/%d ep %d  loss %.4f  ce %.4f  reward %.3f  gnorm %s%s  %.2fs  L=%d  %.1f ex/s  "
                "peak %.2f GiB  ETA %.2f h" % (step, total, ep, lv, cv, rv, "%.2f" % gv if math.isfinite(gv) else
                                              ("-" if math.isnan(gv) else "inf (fp16 step skipped)"),
                                              "  |dL/dz| rl %.3g ce %.3g act %.3g" % tuple(gl) if gl else "",
                                              dt, b["input_ids"].shape[1], ex_s, D.peak_memory_gib(dev), eta_h))
            log_f.flush()
        if (main and is_update and (dev_items or mon_items) and t["eval_every"] and updates % t["eval_every"] == 0
                and step != total - 1):
            evaluate(updates)
        if is_update and t["save_every"] and updates % t["save_every"] == 0 and step != total - 1:
            cnt = global_counters()
            if main:
                save_state(ckpt_path, state(step + 1, cnt), progress(step + 1, "periodic"))
                log(f"[train] saved {ckpt_path} (update {updates})")
        # the budget counts this invocation only (a resumed run gets the full budget again)
        if is_update and budget_s and step != total - 1 and any_rank(time.time() - t_start > budget_s, world, dev):
            cnt = global_counters()
            if main:
                save_state(ckpt_path, state(step + 1, cnt), progress(step + 1, "time_budget"))
                log(f"[train] time budget train.max_hours={t['max_hours']} reached at micro-step {step + 1}/{total}; "
                    f"state saved to {ckpt_path}. Rerun the same command to resume.")
                log_f.close()
                dev_f.close()
            barrier(world, dev)  # nobody exits before the state is on disk
            if world > 1:
                dist.destroy_process_group()
            sys.exit(EXIT_BUDGET)
    t_train = elapsed_before + time.time() - t_loop
    cnt = global_counters()
    if world > 1:
        dist.destroy_process_group()
    if not main:
        return {}  # rank 0 writes the model and the stats
    log_f.close()

    dev_final, calib_final = evaluate(updates, final=True) if dev_items or mon_items else ({}, None)
    dev_f.close()

    steady = recent[3:] if len(recent) > 6 else recent
    st = sum(x_[0] for x_ in steady) or 1e-9
    stats = {
        "device": D.describe(dev), "precision": precision, "grad_checkpointing": gckpt, "world_size": world,
        "base": m["base"], "base_revision": acfg["base_revision"], "sources": srcs,
        "corpus_sha256": manifest.get("corpus_sha256"),
        "train_records": len(train_items[0]), "calib_records": len(calib_recs), "dev_records": len(dev_items),
        "renderings": len(train_items), "rl_group": G,
        "micro_steps": total, "updates": updates, "epochs": t["epochs"], "examples_seen": cnt["seen_ex"],
        "train_loop_wall_s": round(t_train, 1),
        "examples_per_s_recent": round(world * sum(x_[1] for x_ in steady) / st, 2),
        "real_tokens_per_s_recent": round(world * sum(x_[2] for x_ in steady) / st, 1),
        "mean_real_tokens_per_example": round(cnt["seen_real"] / max(1, cnt["seen_ex"]), 1),
        "padding_overhead": round(cnt["seen_pad"] / max(1, cnt["seen_real"]) - 1, 4),
        "peak_memory_gib": round(D.peak_memory_gib(dev), 2), "pad_multiple": pad,
        "optimizer": opt_impl, "optimizer_impl": t["optimizer_impl"], "dev_final": dev_final,
        "calib_slice_final": calib_final, "monitors_off": sorted(off), "memprobe": probe,
        "config_hash": chash, "data_key": data_key, "smoke": bool(cfg["run"].get("smoke")),
        "option_fit": fit.get("option_fit"),
    }
    per_s = max(stats["examples_per_s_recent"], 1e-9)
    stats["hours_per_epoch_estimate"] = round(stats["train_records"] / per_s / 3600, 3)
    model.eval()
    sd = {k_: v.detach().float().cpu().numpy() for k_, v in model.state_dict().items()}
    ck = write_checkpoint(P["model"], sd, acfg, ecfg, P["run"] / "tokenizer", {
        "updates": updates, "epochs_completed": t["epochs"], "hours": round(t_train / 3600, 4), "world_size": world,
        "fine_tuned_from_checkpoint": False, "trainer": "zhjudge zhjudge.train", "device": D.describe(dev),
        "precision": precision, "base": m["base"], "base_revision": acfg["base_revision"],
        "corpus_sha256": manifest.get("corpus_sha256"), "sources": srcs, "smoke_test": bool(cfg["run"].get("smoke")),
        "jev_outputs_used": False})
    stats["checkpoint"] = str(ck)
    stats["total_wall_s"] = round(time.time() - t_start, 1)
    (P["run"] / "train_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    done_path.write_text(json.dumps({"config_hash": chash, "checkpoint": str(ck), "updates": updates}, indent=2))
    if not t.get("keep_last_ckpt"):
        for p in (ckpt_path, ckpt_path.with_suffix(".json")):
            p.unlink(missing_ok=True)  # the Laya checkpoint is written; the resumable state is no longer needed
    log(json.dumps({k_: v for k_, v in stats.items() if k_ not in ("dev_final", "sources", "option_fit",
                                                                  "calib_slice_final", "memprobe")}, indent=2))
    return stats


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--force", action="store_true", help="retrain even if runs/<name>/model exists for this config")
    a = ap.parse_args(argv)
    train(load_config(a.config, a.set), force=a.force)


if __name__ == "__main__":
    main()
