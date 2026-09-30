"""Upload an exported checkpoint (e.g. a Kaggle run's outputs/model/<name>/) to the Hugging Face Hub.

  HF_TOKEN=... uv run zhjudge publish EXPORT_DIR --repo-id <user>/<name> [--public] [--config configs/train_v1b.yaml]
  uv run zhjudge publish EXPORT_DIR --dry-run           # only prepare the folder and the model card

Works on an export that already exists, so nothing is retrained or re-evaluated: the folder is copied to --out (Kaggle
inputs are read-only), the model card and DATA_LICENSES.md are regenerated with the current code from the export's own
rl_agent_config.json, eval.json and training_data.json, the structure is checked, and the folder is uploaded. The
repository is created private unless --public; make it public on the Hub after checking the page.

Refuses a smoke-test model and a weights licence that does not follow the share-alike training sources. The token is
read from the HF_TOKEN environment variable only, never printed and never written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .config import add_common_args, load_config, paths
from .data.allowlist import load_allowlist
from .export import model_card, share_alike_sources, structural_check


def prepare(cfg: dict, src: Path, out: Path, repo_id: str | None = None) -> tuple[Path, str]:
    """Copy the export to `out` with a regenerated card; returns (folder, problem or ''). With `repo_id` the card is
    titled after the Hub repo and its usage example loads the model from it instead of a local path."""
    P = paths(cfg)
    src = Path(src)
    problems = structural_check(src)
    if problems:
        raise SystemExit(f"{src} is not a complete export: {problems}")
    acfg = json.loads((src / "rl_agent_config.json").read_text(encoding="utf-8"))
    ev = json.loads((src / "eval.json").read_text(encoding="utf-8")) if (src / "eval.json").exists() else None
    td = json.loads((src / "training_data.json").read_text(encoding="utf-8")) \
        if (src / "training_data.json").exists() else {}
    if (ev or {}).get("smoke") or acfg.get("training", {}).get("smoke"):
        raise SystemExit("refusing to publish a smoke-test model")
    # the card reads eval-only sources from a manifest; an export carries training_data.json instead
    allow = load_allowlist(P["allowlist"])
    eval_only = {v["converter"]: {"eval_only": True} for v in allow.values()
                 if v.get("enabled") and v.get("converter") and v.get("role") == "eval_only"}
    manifest = {"corpus_sha256": td.get("corpus_sha256"), "train_sample": td.get("train_sample") or {},
                "sources": eval_only}
    name = src.name
    dst = Path(out) / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    lic = P["root"] / "DATA_LICENSES.md"
    if lic.exists():
        shutil.copy2(lic, dst / "DATA_LICENSES.md")
    title = repo_id.split("/")[-1] if repo_id else name
    card = model_card(cfg, acfg, ev, manifest, title, P)
    if repo_id:
        card = card.replace(f'laya.load("path/to/{title}")', f'laya.load("{repo_id}")')
    (dst / "README.md").write_text(card, encoding="utf-8")
    problems = structural_check(dst)
    if problems:
        raise SystemExit(f"prepared folder {dst} failed the structural check: {problems}")
    weights_lic = str(cfg["export"].get("license") or "other")
    sa = share_alike_sources(cfg, P, acfg.get("training", {}).get("sources"))
    problem = ""
    if sa and not weights_lic.lower().startswith("cc-by-sa"):
        problem = f"export.license={weights_lic} but the model was trained on share-alike sources {sa}"
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**20
    print(f"[publish] prepared {dst} ({size:.0f} MB, weights licence {weights_lic})", flush=True)
    return dst, problem


def publish(cfg: dict, src: Path, repo_id: str | None, out: Path, public=False, dry_run=False) -> Path:
    dst, problem = prepare(cfg, src, out, repo_id)
    if problem:
        raise SystemExit(f"refusing to publish: {problem} (DATA_LICENSES.md section 6)")
    if dry_run:
        print("[publish] dry run: nothing uploaded", flush=True)
        return dst
    if not repo_id:
        raise SystemExit("--repo-id <user>/<name> is required (or --dry-run)")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN in the environment (a Hugging Face token with write access); it is never "
                         "written to disk")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    url = api.create_repo(repo_id, repo_type="model", private=not public, exist_ok=True)
    api.upload_folder(folder_path=str(dst), repo_id=repo_id, repo_type="model", commit_message=f"upload {dst.name}")
    print(f"[publish] uploaded to {url} ({'public' if public else 'private: make it public in the repo settings'})",
          flush=True)
    return dst


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("export_dir", help="an exported checkpoint folder (runs/<run>/export/<name> or outputs/model/<name>)")
    ap.add_argument("--repo-id", default=None, help="Hugging Face repo id <user>/<name> (token from $HF_TOKEN)")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    ap.add_argument("--out", default="publish", help="where the prepared copy is written (default ./publish)")
    ap.add_argument("--dry-run", action="store_true", help="prepare the folder and card only; upload nothing")
    a = ap.parse_args(argv)
    publish(load_config(a.config, a.set), Path(a.export_dir), a.repo_id, Path(a.out), a.public, a.dry_run)


if __name__ == "__main__":
    main()
