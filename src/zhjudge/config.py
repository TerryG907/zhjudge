"""Config loading and repo paths.

A run is described by one YAML file (configs/train_*.yaml) merged over DEFAULTS below. A few environment
variables override it so the same config works on a laptop, Colab, Kaggle or a rented GPU:

  ZHJUDGE_DEVICE     auto | cuda | mps | cpu        -> train.device (also used by calibrate/eval/compat)
  ZHJUDGE_PRECISION  auto | bf16 | fp16 | fp32      -> train.precision
  ZHJUDGE_THREADS    int                            -> train.cpu_threads (CPU intra-op threads)
  RUN_NAME           str                            -> run.name
  ZHJUDGE_RUNS_DIR   path                           where runs/<name>/ live (e.g. a mounted Google Drive)
  ZHJUDGE_DATA_DIR   path                           where data/unified, data/probes live
  ZHJUDGE_ROOT       path                           repo root (default: nearest parent of cwd with configs/)

`--set a.b=value` on any command line overrides one key (value parsed as YAML).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "run": {"name": "dev", "seed": 13, "smoke": False},
    "model": {
        "base": "jhu-clsp/mmBERT-base",
        "revision": "c5955035435e2bf121cde7f3c8863ef52ff35d82",
        "max_len": 1024,
        "head_max_len": 256,
        "head_layers": 2,
        "model_name": "zhjudge",
    },
    "data": {
        "allowlist": "configs/datasets.yaml",
        "licenses": "configs/licenses.yaml",  # licence audit; build refuses enabled sources it does not allow
        "license_profile": "default",  # default (ALLOW + CONDITIONAL) | permissive_strict (also no share-alike)
        "sources": "enabled",        # "enabled" = every enabled converter in the allowlist, or an explicit list
        "decontaminate": "full",     # full | selected  (see zhjudge.data.build)
        "probes": True,              # build the baseline probe sets before the corpus
        "allow_disabled": False,     # smoke tests only (requires run.smoke: true)
        "clean": True,               # wipe data/unified before building (no stale sources)
        "train_sample": {},          # train-split sizes / extras per converter (data.common.TRAIN_SAMPLE_KEYS);
                                     # dev/test never change; train refuses a corpus built with other values
    },
    "train": {
        "device": "auto",
        "precision": "auto",
        "cpu_threads": 0,
        "epochs": 1,
        "max_steps": 0,
        "batch_size": 32,
        "max_tokens": 8192,
        "grad_accum": 1,
        "lr_encoder": 5.0e-5,
        "lr_head": 2.0e-4,
        "weight_decay": 0.01,
        "warmup": 0.06,
        "grad_clip": 1.0,
        "rl_weight": 1.0,
        "rl_group": 4,
        "rl_sigma_start": 0.4,
        "rl_sigma_end": 0.1,
        "act_weight": 0.1,
        "grad_checkpointing": "auto",
        "optimizer_impl": "auto",    # AdamW: auto (fused on CUDA, else torch's default) | fused | foreach | default
        "pad_multiple": 32,          # training batches padded to a multiple of this (the batch plan does not change)
        "freeze_embeddings": False,
        "max_train_records": 0,
        "max_train_per_source": 0,
        "label_smoothing": 0.05,
        "score_neighbor_mass": 0.1,
        "format_aug": 0.0,           # share of train records written Laya-style from their view (zhjudge.data.views)
        "format_cross_lingual": 0.3,  # of those: in the other language (instruction, field names, translatable options)
        "format_bare_labels": 0.3,   # of the rendered NLI records: bare labels ("entailment", not "entailment: ...")
        "option_shuffle": 0.0,       # share of train `choice` records whose options are re-ordered at random
        "format_variants": 0.0,      # of the Laya-style records: share in another Laya question shape (views.variant)
        "format_aug_per_epoch": False,  # a fresh rendering / option order of every train record in each epoch
        "calib_frac": 0.02,
        "calib_cap": 4000,
        "calib_min": 50,
        "dev_max_records": 2000,
        "eval_every": 1000,
        "save_every": 1000,
        "log_every": 50,
        "monitor_calib_per_source": 0,  # > 0: up to this many calib records per source, scored at every dev eval
        "memprobe": False,           # CUDA: `zhjudge memprobe` in a separate process before training (report only)
        "resume": True,
        "max_hours": 0,              # wall-clock budget per invocation; 0 = none. On expiry: save ckpt/last.pt, exit 75,
                                     # rerun resumes (with a fresh budget)
        "keep_last_ckpt": False,     # delete ckpt/last.pt once the final Laya checkpoint is written
        "max_bad_steps": 20,
    },
    "calibrate": {"min_bucket": 200, "batch_size": 64},
    "eval": {
        "split": "test",
        "max_per_source": 2000,
        "seed": 13,                  # seed of the capped test sample (13 = v0's run.seed); run.seed does not move it
        "probes": True,
        "baseline": True,
        "batch_size": 64,
        "laya_test": 0,              # per source: test records also scored written Laya-style (test:laya:*)
        "compare_preds": None,       # a previous run's eval_preds.jsonl (e.g. v0's): paired comparison in eval.md
        "compare_label": None,       # its name in the report (default: the file path)
        "reference": "configs/eval_reference_v0.json",  # v0's per-source dev/test counts (`zhjudge validate` warns)
    },
    "export": {"name": None, "license": "cc-by-sa-4.0",  # see DATA_LICENSES.md, release profiles
               "source_url": "https://github.com/TerryG907/zhjudge"},  # linked from the model card; null = none
    "compat": {"n": 200},
}

ENV_OVERRIDES = {
    "ZHJUDGE_DEVICE": ("train", "device"),
    "ZHJUDGE_PRECISION": ("train", "precision"),
    "ZHJUDGE_THREADS": ("train", "cpu_threads"),
    "RUN_NAME": ("run", "name"),
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def repo_root() -> Path:
    env = os.environ.get("ZHJUDGE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "configs").is_dir() and (p / "pyproject.toml").exists():
            return p
    # installed package without a checkout: fall back to the source tree layout
    return Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None, sets: list[str] | None = None) -> dict:
    root = repo_root()
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = root / p
        cfg = _merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        cfg["_config_path"] = str(p)
    for env, (sec, key) in ENV_OVERRIDES.items():
        v = os.environ.get(env)
        if v not in (None, ""):
            cfg[sec][key] = yaml.safe_load(v)
    for s in sets or []:
        k, _, v = s.partition("=")
        node = cfg
        parts = k.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(v)
    if cfg["data"].get("allow_disabled") and not cfg["run"].get("smoke"):
        raise SystemExit("data.allow_disabled is only allowed for smoke tests (run.smoke: true)")
    return cfg


def paths(cfg: dict) -> dict:
    root = repo_root()
    data = Path(os.environ.get("ZHJUDGE_DATA_DIR") or root / "data")
    runs = Path(os.environ.get("ZHJUDGE_RUNS_DIR") or root / "runs")
    run = runs / cfg["run"]["name"]
    return {
        "root": root,
        "data": data,
        "unified": data / "unified",
        "probes": data / "probes",
        "runs": runs,
        "run": run,
        "model": run / "model",          # Laya-format checkpoint written by train, calibrated in place
        "ckpt": run / "ckpt",            # resumable training state
        "export": run / "export",
        "allowlist": _under(root, cfg["data"]["allowlist"]),
        "licenses": _under(root, cfg["data"].get("licenses") or "configs/licenses.yaml"),
    }


def _under(root: Path, p) -> Path:
    return Path(p) if Path(p).is_absolute() else root / p


def config_hash(cfg: dict, sections=("model", "data", "train")) -> str:
    sub = {k: cfg.get(k) for k in sections}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:16]


def add_common_args(ap):
    ap.add_argument("--config", default=os.environ.get("CONFIG", "configs/train_base.yaml"),
                    help="YAML config (default: $CONFIG or configs/train_base.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one config key, e.g. --set train.epochs=2 (repeatable)")
    return ap


def save_resolved(cfg: dict, run_dir: Path, name="config.resolved.yaml"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                                               allow_unicode=True, sort_keys=False), encoding="utf-8")
