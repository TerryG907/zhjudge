"""`zhjudge <command> [--config FILE] [--set key=value ...]`

Commands, in pipeline order:
  probes     build the baseline probe sets (data/probes)            zhjudge.data.probes
  build      build the unified corpus from enabled datasets          zhjudge.data.build
  validate   independent corpus checks (leakage, schema, allowlist)  zhjudge.data.validate
  train      train (resumable)                                        zhjudge.train
  calibrate  fit temperatures on the held-out calibration slice      zhjudge.calibrate
  eval       metrics vs laya-multilingual                             zhjudge.eval
  export     Laya checkpoint + model card                             zhjudge.export
  compat     load the export in Laya runtimes                         zhjudge.compat
  all        every step above in order (what scripts/run_all.sh runs after `uv sync`)

Not part of `all`:
  compare    two runs' eval_preds.jsonl item by item (e.g. v1 vs v0)  zhjudge.compare
  memprobe   GPU peak of a training step without grad checkpointing   zhjudge.memprobe (train runs it: train.memprobe)
  publish    upload an export to the Hugging Face Hub (HF_TOKEN)      zhjudge.publish
"""
from __future__ import annotations

import importlib
import sys
import time

COMMANDS = {
    "probes": "zhjudge.data.probes",
    "build": "zhjudge.data.build",
    "validate": "zhjudge.data.validate",
    "train": "zhjudge.train",
    "calibrate": "zhjudge.calibrate",
    "eval": "zhjudge.eval",
    "export": "zhjudge.export",
    "compat": "zhjudge.compat",
    "compare": "zhjudge.compare",
    "memprobe": "zhjudge.memprobe",
    "publish": "zhjudge.publish",
}
PIPELINE = ["probes", "build", "validate", "train", "calibrate", "eval", "export", "compat"]


def run(cmd, argv):
    mod = importlib.import_module(COMMANDS[cmd])
    try:
        mod.main(argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "all":
        from .config import load_config

        # the probe step is skipped when the config turns probes off everywhere (e.g. the smoke test)
        cfg_path = next((rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "--config"), None)
        sets = [rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "--set"]
        cfg = load_config(cfg_path, sets)
        for step in PIPELINE:
            if step == "probes" and not (cfg["data"]["probes"] or cfg["eval"]["probes"]):
                print("== probes: skipped (data.probes and eval.probes are off)", flush=True)
                continue
            t0 = time.time()
            print(f"== {step}", flush=True)
            run(step, rest)
            print(f"== {step} done in {time.time() - t0:.1f}s", flush=True)
        return
    if cmd not in COMMANDS:
        raise SystemExit(f"unknown command {cmd!r}\n{__doc__}")
    run(cmd, rest)


if __name__ == "__main__":
    main()
