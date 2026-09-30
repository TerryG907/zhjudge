"""Measure the GPU memory of a training step without gradient checkpointing (report only: nothing is changed).

  uv run zhjudge memprobe --config configs/train_base.yaml

Prepares the data and the batch plan exactly like `zhjudge train` (so it writes the token caches train then reuses),
builds the model, and runs forward + backward on the plan's batch with the most padded tokens and on its longest
batch: without gradient checkpointing, then with it (what train uses on < 20 GiB GPUs; that estimate can be checked
against train_stats.json peak_memory_gib). Writes runs/<name>/memprobe.json: peak allocated / reserved memory, the
AdamW state and optimizer-step temporaries train adds on top, and whether a step without checkpointing would fit
(estimate <= 90% of the GPU - 0.75 GiB). One process, no DDP (DDP adds about one more gradient copy); timings are
single passes after one warm-up.

`zhjudge train` runs this in a separate process when train.memprobe is true (CUDA only, rank 0, before it builds its
own model), so an out-of-memory or a sticky CUDA error here never reaches training; a failure or timeout is only
logged. Without CUDA it prints "skipped (no CUDA)" and exits 0.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from . import device as D  # noqa: E402
from .config import add_common_args, load_config, paths  # noqa: E402
from .model import build_from_base  # noqa: E402
from .records import collate_np  # noqa: E402
from .train import adamw_kwargs, epoch_items, plan_batches, prepare, resume_hash  # noqa: E402

GiB = 2**30
KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "target")


def worst_batches(plan, train_items, pad) -> dict:
    """The plan's batch with the most padded tokens and its longest batch (padded length), as lists of items."""
    def shape(eb):
        items = epoch_items(train_items, eb[0])
        L = -(-max(len(items[i]["ids"]) for i in eb[1]) // pad) * pad
        return len(eb[1]) * L, L

    worst = {"most_padded_tokens": max(plan, key=shape), "longest": max(plan, key=lambda eb: shape(eb)[::-1])}
    return {k: [epoch_items(train_items, ep)[i] for i in b] for k, (ep, b) in worst.items()}


def set_checkpointing(model, on: bool):
    if on:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.encoder.gradient_checkpointing_disable()
    model.head_checkpointing = on


def fwd_bwd(model, b, dev, amp):
    x = {k: torch.from_numpy(b[k]).to(dev) for k in KEYS}
    with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
        logits, act = model(x["input_ids"], x["attention_mask"], x["marker_pos"], x["marker_mask"], x["qtype"])
    # train's CE and act terms; its RL term only adds (rl_group, batch, options) tensors
    loss = -(x["target"] * torch.log_softmax(logits.float(), -1)).sum(-1).mean() + act.float().logsumexp(-1).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def measure(model, batches, dev, amp, gckpt) -> dict:
    """Peak memory over forward + backward of each batch (gradients freed after each, as after an update)."""
    set_checkpointing(model, gckpt)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    out = {"seconds": {}, "oom": False}
    for name, b in batches.items():
        D.sync(dev)
        t0 = time.time()
        try:
            fwd_bwd(model, b, dev, amp)
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            out["oom"] = True
            out["seconds"][name] = None
            continue
        D.sync(dev)
        out["seconds"][name] = round(time.time() - t0, 3)
    if cuda:
        out["peak_allocated"] = torch.cuda.max_memory_allocated(dev)
        out["peak_reserved"] = torch.cuda.max_memory_reserved(dev)
        torch.cuda.empty_cache()
    return out


def memprobe(cfg: dict, allow_cpu=False) -> dict | None:
    """allow_cpu: run the whole probe on CPU without memory numbers (tests the code path)."""
    t_start = time.time()
    P = paths(cfg)
    t = cfg["train"]
    seed = int(cfg["run"]["seed"])
    D.set_threads(t["cpu_threads"])
    dev = D.select_device(t["device"])
    if dev.type != "cuda" and not allow_cpu:
        print("[memprobe] skipped (no CUDA)", flush=True)
        return None
    cuda = dev.type == "cuda"
    precision = D.select_precision(dev, t["precision"])
    amp = D.amp_dtype(precision)
    torch.manual_seed(seed)
    model, tok, _, _ = build_from_base(cfg, P["run"], seed)
    train_items = prepare(cfg, P, tok, seed, write=False)[0]
    pad = int(t["pad_multiple"])
    plan = plan_batches(train_items, t, seed)
    worst = worst_batches(plan, train_items, pad)
    batches = {k: collate_np(b, tok.pad_token_id, pad) for k, b in worst.items()}
    if t["freeze_embeddings"]:
        model.encoder.embeddings.tok_embeddings.weight.requires_grad_(False)
    free0, total = torch.cuda.mem_get_info(dev) if cuda else (None, None)
    model.to(dev).train()
    weights = torch.cuda.memory_allocated(dev) if cuda else None
    trainable = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    fused = bool(adamw_kwargs(str(t["optimizer_impl"]), dev).get("fused"))
    adam, step_tmp = 2 * trainable, 0 if fused else trainable  # fp32 moments; foreach step() temporaries
    set_checkpointing(model, True)
    ep0, b0 = plan[0]  # warm-up
    fwd_bwd(model, collate_np([epoch_items(train_items, ep0)[i] for i in b0], tok.pad_token_id, pad), dev, amp)
    runs = {"without_checkpointing": measure(model, batches, dev, amp, False),
            "with_checkpointing": measure(model, batches, dev, amp, True)}

    def gib(x):
        return None if x is None else round(x / GiB, 3)

    usable = 0.9 * total - 0.75 * GiB if cuda else None
    res = {"device": D.describe(dev), "precision": precision, "config_hash": resume_hash(cfg),
           "pad_multiple": pad, "optimizer": "fused" if fused else "foreach/default",
           "batches": {k: {"n": int(b["input_ids"].shape[0]), "padded_len": int(b["input_ids"].shape[1]),
                           "padded_tokens": int(b["padded_tokens"]), "real_tokens": int(b["real_tokens"])}
                       for k, b in batches.items()},
           "gpu_total_gib": gib(total), "gpu_free_at_start_gib": gib(free0), "usable_gib": gib(usable),
           "weights_gib": gib(weights), "trainable_gib": gib(trainable), "adam_state_gib": gib(adam),
           "optimizer_step_temp_gib": gib(step_tmp)}
    for k, r in runs.items():
        est = None
        if cuda and not r["oom"]:  # the forward/backward peak with the moments resident, or the optimizer step
            est = max(r["peak_allocated"] + adam, weights + trainable + adam + step_tmp)
        res[k] = {"peak_allocated_gib": gib(r.get("peak_allocated")), "peak_reserved_gib": gib(r.get("peak_reserved")),
                  "oom": r["oom"], "seconds": r["seconds"], "estimate_train_peak_gib": gib(est)}
    no = res["without_checkpointing"]
    res["would_fit_without_checkpointing"] = (not no["oom"] and no["estimate_train_peak_gib"] is not None
                                              and no["estimate_train_peak_gib"] <= res["usable_gib"])
    res["seconds_total"] = round(time.time() - t_start, 1)
    out = P["run"] / "memprobe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    w = res["with_checkpointing"]
    print(f"[memprobe] {res['device']}, {precision}, pad {pad}, AdamW {res['optimizer']}; batches "
          + ", ".join(f"{k} {v['n']} x {v['padded_len']}" for k, v in res["batches"].items())
          + "\n[memprobe] without checkpointing: " + ("out of memory" if no["oom"] else
                                                      f"peak {no['peak_allocated_gib']} GiB allocated "
                                                      f"({no['peak_reserved_gib']} reserved)")
          + f"; with: peak {w['peak_allocated_gib']} GiB; + AdamW state {res['adam_state_gib']} GiB -> estimated "
          f"train peak {no['estimate_train_peak_gib']} / {w['estimate_train_peak_gib']} GiB of {res['usable_gib']} "
          f"usable: {'would' if res['would_fit_without_checkpointing'] else 'would NOT'} fit without checkpointing "
          f"(report only) -> {out}", flush=True)
    return res


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args(argv)
    memprobe(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
