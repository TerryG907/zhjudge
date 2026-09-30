"""Batched inference on a Laya DecisionModel, applying temperatures exactly like `laya.Agent.system_one`
(per (qtype, #options) bucket first, then per qtype, both clamped to [0.5, 5.0])."""
from __future__ import annotations

import numpy as np
import torch

from .metrics import softmax
from .records import collate_np, eval_batches


@torch.no_grad()
def batched_logits(model, items, pad_id, device, amp=None, batch_size=64, max_tokens=0):
    """Raw (pre-temperature) logits per item, in item order: list of np.ndarray of length k."""
    was_training = model.training
    model.eval()
    out = [None] * len(items)
    for bidx in eval_batches(items, batch_size, max_tokens):
        b = collate_np([items[i] for i in bidx], pad_id)
        t = {k: torch.from_numpy(b[k]).to(device) for k in ("input_ids", "attention_mask", "marker_pos",
                                                             "marker_mask", "qtype")}
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            lg, _ = model(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
        lg = lg.float().cpu().numpy()
        for r, i in enumerate(bidx):
            out[i] = lg[r, : len(items[i]["markers"])].copy()
    if was_training:
        model.train()
    return out


def agent_amp(agent):
    """The autocast dtype laya.Agent uses for this device (CUDA only; fp32 elsewhere)."""
    return agent.dtype if agent.device.type == "cuda" else None


def temperature_for(qtype: int, k: int, temps, temps_by_options):
    from laya.common import clamp_temperature, temp_bucket

    t = temps_by_options.get(temp_bucket(qtype, k), temps[qtype])
    return clamp_temperature(t)


def probs_from_logits(logits, qtypes, temps=(1.0, 1.0, 1.0), temps_by_options=None):
    temps_by_options = temps_by_options or {}
    return [softmax(np.asarray(z) / temperature_for(q, len(z), temps, temps_by_options))
            for z, q in zip(logits, qtypes)]
