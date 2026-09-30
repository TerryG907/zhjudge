"""Laya-format model: a pretrained ModernBERT-family encoder + Laya's typed decision head.

The module object is upstream laya's own `laya.common.DecisionModel` (PyPI `laya`, Apache-2.0), so parameter names,
shapes, head construction and default initialisation are exactly upstream's and the state_dict IS the Laya
checkpoint format by construction (the spike verified this with laya, laya-mlx and laya-serve; `zhjudge compat` re-checks
laya and laya-serve, and scripts/coreml_compat.py is the laya-coreml check).

Laya checkpoint directory:
  model.safetensors      fp16, upstream PyTorch parameter names (+ `temperature` buffer kept fp32)
  rl_agent_config.json   encoder id, head_layers, max_len, head_max_len, act_costs, temperature[3],
                         temperature_by_options{}, training{...}
  encoder/config.json    Hugging Face ModernBERT config (model_type "modernbert")
  tokenizer/             tokenizer.json + tokenizer_config.json (cls/sep/pad/mask tokens)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

# Known bases (all MIT/Apache-2.0). Any ModernBERT-family encoder id + revision works via the config.
BASES = {
    "jhu-clsp/mmBERT-base": "c5955035435e2bf121cde7f3c8863ef52ff35d82",     # MIT, 307M, multilingual (default)
    "jhu-clsp/mmBERT-small": "abc32620dd4f6ab06f5fbe905dc25f310618e09f",    # MIT, 140M, smoke tests
    "answerdotai/ModernBERT-large": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",  # Apache-2.0, English-centric
}

# the baseline we must beat: laya-multilingual, at the revision the spike measured
BASELINE_REPO = "convaiinnovations/laya"
BASELINE_REVISION = "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"
BASELINE_SUBFOLDER = "multilingual"


def agent_config(encoder_id, max_len=1024, head_max_len=256, model_name="zhjudge", head_layers=2):
    """rl_agent_config.json with exactly the keys the shipped laya-multilingual checkpoint uses."""
    return {
        "encoder": encoder_id,
        "head_layers": head_layers,
        "max_len": max_len,
        "head_max_len": head_max_len,
        "max_prefixes": 6,
        "act_costs": {"escalate": 0.5},
        "cost_wrong_act": 3.0,
        "amp_dtype": "bf16",
        "model_name": model_name,
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {},
    }


def fix_tokenizer_config(tok_dir):
    """Same normalisation upstream laya applies at load (list-valued extra_special_tokens -> dict,
    TokenizersBackend -> PreTrainedTokenizerFast), done at save time so every runtime reads it."""
    p = Path(tok_dir) / "tokenizer_config.json"
    cfg = json.loads(p.read_text())
    if cfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg.pop("backend", None)
        cfg.pop("is_local", None)
    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        cfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")


def build_from_base(cfg: dict, run_dir: Path, seed: int):
    """Pretrained encoder + freshly initialised Laya heads. Writes run_dir/{tokenizer,encoder}/ once.

    Returns (model, tokenizer, agent_cfg, encoder_cfg)."""
    import torch
    from laya.common import DecisionModel
    from transformers import AutoModel, AutoTokenizer

    m = cfg["model"]
    repo, rev = m["base"], m.get("revision") or BASES.get(m["base"])
    torch.manual_seed(seed)
    enc = AutoModel.from_pretrained(repo, revision=rev, attn_implementation="sdpa", dtype=torch.float32)
    enc_cfg = json.loads(enc.config.to_json_string())  # == what config.save_pretrained() writes (upstream notebook)
    acfg = agent_config(repo, m["max_len"], m["head_max_len"], m["model_name"], m.get("head_layers", 2))
    model = DecisionModel(enc, acfg["head_layers"], len(acfg["act_costs"]) + 1)
    try:  # ModernBERT would otherwise torch.compile the encoder ("auto"); keep eager like upstream laya
        model.encoder.config.reference_compile = False
    except Exception:  # noqa: BLE001
        pass
    tok_dir = Path(run_dir) / "tokenizer"
    if not (tok_dir / "tokenizer_config.json").exists():
        AutoTokenizer.from_pretrained(repo, revision=rev).save_pretrained(str(tok_dir))
        fix_tokenizer_config(tok_dir)
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    (Path(run_dir) / "encoder").mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "encoder" / "config.json").write_text(json.dumps(enc_cfg, indent=2) + "\n")
    acfg["base_revision"] = rev
    return model, tok, acfg, enc_cfg


def write_checkpoint(out_dir, state_dict_np, agent_cfg, encoder_cfg, tokenizer_src_dir, training_meta):
    """Write a Laya-format checkpoint. state_dict_np: {upstream torch name: np.ndarray}.

    Weights are stored fp16 like the shipped laya-multilingual checkpoint; `temperature` stays fp32."""
    from safetensors.numpy import save_file

    out = Path(out_dir)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "encoder").mkdir(parents=True)
    sd = {k: np.ascontiguousarray(v.astype(np.float32 if k == "temperature" else np.float16))
          for k, v in state_dict_np.items()}
    save_file(sd, str(tmp / "model.safetensors"))
    (tmp / "encoder" / "config.json").write_text(json.dumps(encoder_cfg, indent=2) + "\n")
    shutil.copytree(tokenizer_src_dir, tmp / "tokenizer")
    fix_tokenizer_config(tmp / "tokenizer")
    cfg = {k: v for k, v in agent_cfg.items() if k != "base_revision"}
    cfg["training"] = training_meta
    (tmp / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def baseline_dir() -> str:
    """Local path of laya-multilingual at the measured revision (downloads ~650 MB once)."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(BASELINE_REPO, revision=BASELINE_REVISION,
                             allow_patterns=[f"{BASELINE_SUBFOLDER}/{p}" for p in
                                             ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    return str(Path(root) / BASELINE_SUBFOLDER)


def load_agent(path, device: str):
    """Load a Laya checkpoint with the upstream reference runtime (`laya.Agent`): identical loading, config and
    temperature handling to what users get. Our batched inference then runs on `agent.model`."""
    import laya

    return laya.Agent(str(path), device=device)
