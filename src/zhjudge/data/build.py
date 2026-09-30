"""Build the unified typed-decision corpus from the sources enabled in configs/datasets.yaml.

  uv run zhjudge build --config configs/train_base.yaml

Steps (ported from the spike's data/build.py): download (pinned revisions) -> convert -> de-duplicate ->
decontaminate -> write data/unified/<split>/<source>.jsonl -> manifest.json + SHA256SUMS. Then run
`zhjudge validate`.

Decontamination modes (config `data.decontaminate`):
  full     (spike behaviour, use for real runs) train is decontaminated against every dev/test record of EVERY
           registered source (enabled or not; they are downloaded and converted for this purpose only, never
           written), CLUE dev/test, LocalLLaMA/typed-decisions train+test and every item of the baseline probe
           sets in data/probes/; dev/test records that overlap a probe item are dropped.
           A train record is also dropped when one of its texts (>= 12 normalised chars) occurs INSIDE a text of an
           eval-only source (Belebele passages are runs of FLORES-200 sentences, SIB-200 labels FLORES sentences).
  selected (smoke tests) only the selected sources' own dev/test records plus the probe items, if present.
Output files are byte-identical across rebuilds with the same inputs (the manifest records sha256 of each).

data.train_sample (default {}; configs/train_v1b.yaml) changes TRAIN records of single converters only (sizes and
extra train-only questions, see zhjudge.data.common.TRAIN_SAMPLE_KEYS); dev/test files are byte-identical with and without
it. The manifest records it (`train_sample`; `split_sources` names it per source, both go into the export's
training_data.json), and `zhjudge train` refuses a corpus built with other values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict

import pandas as pd

from ..config import add_common_args, load_config, paths
from . import common, en, zh
from .allowlist import load_allowlist, load_licenses, select_sources
from .common import (DOWNLOADED, MIN_UNIT, REVISIONS, ContainmentIndex, _strings, dl, norm_key, probe_units,
                     record_units, write_jsonl)

SOURCES = {
    # zh — trainable
    "pawsx_zh": zh.pawsx_zh, "massive_zh": zh.massive_zh, "crosswoz_zh": zh.crosswoz_zh, "cold_zh": zh.cold_zh,
    "csl_zh": None,  # needs CLUE-csl exclusion set, wired in main()
    "sib200_zh": zh.sib200_zh, "nli26_zh_mnli": None, "nli26_zh_wanli": zh.nli26_zh_wanli,
    "nli26_zh_fever": zh.nli26_zh_fever, "oasst2_zh": zh.oasst2_zh,
    # zh — eval only
    "belebele_zh": zh.belebele_zh, "xcopa_zh": zh.xcopa_zh, "xwinograd_zh": zh.xwinograd_zh,
    "globalmmlu_zh": zh.globalmmlu_zh, "minds14_zh": zh.minds14_zh,
    # en
    "banking77_en": en.banking77_en, "boolq_en": en.boolq_en, "dbpedia14_en": en.dbpedia14_en, "mnli_en": en.mnli_en,
    "paws_en": en.paws_en, "csqa_en": en.csqa_en, "arc_en": en.arc_en, "goemotions_en": en.goemotions_en,
    "civil_en": en.civil_en, "helpsteer2_en": en.helpsteer2_en, "massive_en": en.massive_en, "oasst2_en": en.oasst2_en,
}
EVAL_ONLY = {"belebele_zh", "xcopa_zh", "xwinograd_zh", "globalmmlu_zh", "minds14_zh"}
# 2: records carry a `view` (zhjudge.data.views) and record_units() includes the view texts
CORPUS_VERSION = 3

CLUE_CFGS = ["c3", "ocnli", "tnews", "iflytek", "afqmc", "cluewsc2020", "cmnli", "csl"]

# Where each of our splits comes from (official split names). The baseline probe evaluates on OFFICIAL VALIDATION
# splits of CLUE c3/ocnli/tnews/iflytek/afqmc, ChnSentiCorp, LCQMC, amazon_reviews_multi zh/en, XNLI zh/en, GLUE
# mnli validation_matched / sst2 / qqp, AG News test, DREAM dev and LocalLLaMA/typed-decisions test. None of those
# official splits is used for any of our splits (MNLI validation_matched deliberately skipped), and validate proves
# zero text overlap with every probe item.
SPLIT_DOC = {
    "pawsx_zh": "train<-train, dev<-validation, test<-test (official)",
    "massive_zh": "train<-train, dev<-validation, test<-test (official MASSIVE 1.1 zh-CN)",
    "crosswoz_zh": "train/dev/test <- official CrossWOZ data_split",
    "cold_zh": "train/dev/test <- official COLD train/dev/test csv",
    "csl_zh": "sha256(doc_id) hash split of neuclir/csl (~3.0% train, 0.25% dev, 0.4% test); CLUE-csl dev/test abstracts removed",
    "sib200_zh": "train/dev/test <- official SIB-200",
    "nli26_zh_mnli": "sha256(premise|hypothesis) hash split 97/1/2; rows whose English premise is in MNLI validation_* removed",
    "nli26_zh_wanli": "sha256(premise|hypothesis) hash split 97/1/2",
    "nli26_zh_fever": "sha256(premise|hypothesis) hash split 97/1/2",
    "oasst2_zh": "train<-train, test<-validation (official)",
    "belebele_zh": "eval-only: test", "xcopa_zh": "eval-only: dev<-validation, test<-test", "xwinograd_zh": "eval-only: test",
    "globalmmlu_zh": "eval-only: 1,000 seeded sample of test", "minds14_zh": "eval-only: all 502 (only split is train)",
    "banking77_en": "train<-train, test<-test (official)", "boolq_en": "train<-train, test<-validation (official test unlabelled)",
    "dbpedia14_en": "train<-10k sample of train, test<-2k sample of test",
    "mnli_en": "train<-12k sample of train, test<-2k sample of validation_mismatched; validation_matched NOT used (probe dev)",
    "paws_en": "labeled_final: train<-10k sample, dev<-1k of validation, test<-2k of test",
    "csqa_en": "train<-train, test<-validation (official test unlabelled)", "arc_en": "official train/validation/test (Challenge+Easy)",
    "goemotions_en": "simplified: train<-10k sample, dev<-1k validation, test<-2k test",
    "civil_en": "train<-bin-balanced 8k of train shard 0, test<-bin-balanced 2k of test",
    "helpsteer2_en": "train<-10k sample of train, test<-validation", "massive_en": "official train/validation/test (en-US)",
    "oasst2_en": "train<-6k sample of train, test<-1k sample of validation",
}


def split_doc(name, sample) -> str:
    """SPLIT_DOC[name] plus the data.train_sample settings of that converter (keys `<source>` or `<source>_*`)."""
    return SPLIT_DOC[name] + "".join(f"; train changed by data.train_sample.{k}={json.dumps(v)} "
                                     f"({common.TRAIN_SAMPLE_KEYS[k]})" for k, v in sorted(sample.items())
                                     if k == name or k.startswith(name + "_"))


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def external_eval_units():
    """Text units of external eval sets that our TRAIN split must not contain (CLUE dev/test, typed-decisions)."""
    units, csl_abs, per = set(), set(), Counter()
    for cfg in CLUE_CFGS:
        for sp in ("validation", "test"):
            try:
                df = pd.read_parquet(dl("clue/clue", f"{cfg}/{sp}-00000-of-00001.parquet"))
            except Exception as e:  # noqa: BLE001
                log(f"  [warn] clue/{cfg}/{sp}: {e}")
                continue
            for col in df.columns:
                if col in ("idx", "label", "id"):
                    continue
                for v in df[col]:
                    for s in _strings(v):
                        k = norm_key(s)
                        if len(k) >= MIN_UNIT:
                            units.add(k)
                            per[f"clue:{cfg}"] += 1
                            if cfg == "csl" and col == "abst":
                                csl_abs.add(k)
    for sp in ("train", "test"):
        df = pd.read_parquet(dl("LocalLLaMA/typed-decisions", f"all/{sp}-00000-of-00001.parquet"))
        for v in df["state"]:
            try:
                obj = json.loads(v)
            except Exception:  # noqa: BLE001
                obj = v
            for s in _strings(obj):
                k = norm_key(s)
                if len(k) >= MIN_UNIT:
                    units.add(k)
                    per["typed-decisions"] += 1
    return units, frozenset(csl_abs), per


def build(cfg: dict) -> dict:
    t0 = time.time()
    P = paths(cfg)
    allow = load_allowlist(P["allowlist"])
    licenses = load_licenses(P["licenses"])
    if licenses is None:
        log(f"  [warn] {P['licenses']} not found: sources are checked against the allowlist only")
    selected, notes = select_sources(cfg, allow, SOURCES, licenses)
    for n in notes:
        log(f"  [allowlist] {n}")
    if not selected:
        raise SystemExit(f"no sources selected: enable datasets in {P['allowlist']} (enabled: true) first")
    mode = cfg["data"]["decontaminate"]
    if mode not in ("full", "selected"):
        raise SystemExit(f"data.decontaminate must be full|selected, got {mode!r}")
    log(f"building {len(selected)} sources ({mode} decontamination): {', '.join(selected)}")
    sample = dict(cfg["data"].get("train_sample") or {})
    unknown = sorted(set(sample) - set(common.TRAIN_SAMPLE_KEYS))
    if unknown:
        raise SystemExit(f"data.train_sample: unknown keys {unknown}; known: {sorted(common.TRAIN_SAMPLE_KEYS)}")
    common.TRAIN_SAMPLE.clear()
    common.TRAIN_SAMPLE.update(sample)
    if sample:
        log(f"  data.train_sample (train records only; dev/test unchanged): {sample}")

    base_units, base_per = probe_units(P["probes"])
    if not base_units:
        msg = f"{P['probes']} has no probe sets; run `zhjudge probes` first so train can be decontaminated against them"
        if mode == "full":
            raise SystemExit(msg)
        log(f"  [warn] {msg}")

    fns = dict(SOURCES)
    ext_units, csl_abs, ext_per = set(), frozenset(), Counter()
    mnli_eval_premises = frozenset()
    if mode == "full":
        log("loading external eval sets for decontamination ...")
        ext_units, csl_abs, ext_per = external_eval_units()
    if mode == "full" or "nli26_zh_mnli" in selected:
        mnli_eval_premises = en.mnli_premises_eval()
    if mode == "full" or "csl_zh" in selected:
        if not csl_abs:
            _, csl_abs, _ = _clue_csl_abstracts()
    ext_units |= base_units
    ext_per.update(base_per)
    fns["csl_zh"] = lambda: zh.csl_zh(exclude_abstract_keys=csl_abs)
    fns["nli26_zh_mnli"] = lambda: zh.nli26_zh_mnli(exclude_en_premises=mnli_eval_premises)

    # 1) convert (full mode: every source, because eval units must come from ALL sources)
    to_convert = list(fns) if mode == "full" else selected
    all_recs = {}
    for name in to_convert:
        ts = time.time()
        all_recs[name] = list(fns[name]())
        log(f"  {name:18s} {len(all_recs[name]):7d} raw records  ({time.time() - ts:.1f}s)")

    def all_units(r, units):
        # raw converter units (e.g. untruncated text, English source premise) + units recoverable from the written
        # record (whole context and each role-prefixed segment); validate re-derives the latter from disk
        return {k for k in (norm_key(u) for u in units) if len(k) >= MIN_UNIT} | record_units(r)

    # 2) eval unit set = all dev/test units of converted sources + external eval sets + probe items
    eval_units = set(ext_units)
    for name, recs in all_recs.items():
        for r, units in recs:
            if r["split"] != "train":
                eval_units.update(all_units(r, units))
    # eval-only texts can CONTAIN a training text (Belebele passage = several FLORES sentences, one of them a SIB-200
    # train sentence): check containment, not only whole-unit equality
    inside_eval_only = ContainmentIndex(r.get("context") or "" for name, recs in all_recs.items() if name in EVAL_ONLY
                                        for r, _ in recs if isinstance(r.get("context"), str))

    # 3) de-dup + decontaminate, write
    out_dir = P["unified"]
    if cfg["data"].get("clean", True) and out_dir.exists():
        shutil.rmtree(out_dir)
    manifest_sources = {}
    for name in selected:
        recs = all_recs[name]
        seen, kept = set(), defaultdict(list)
        drop_dup = drop_contam = drop_base = drop_inside = 0
        inside_examples = []
        for r, units in recs:
            key = (r["split"], r["question"], r.get("context", ""), tuple(r.get("options", [])))
            if key in seen:
                drop_dup += 1
                continue
            seen.add(key)
            if r["split"] == "train":
                if name in EVAL_ONLY:
                    raise AssertionError(f"{name} is eval-only but produced train records")
                ru = all_units(r, units)
                inside = [u for u in ru if inside_eval_only.hit(u)]
                if not ru.isdisjoint(eval_units) or inside:
                    drop_contam += 1
                    if inside and ru.isdisjoint(eval_units):
                        drop_inside += 1
                        if len(inside_examples) < 3:
                            inside_examples.append(inside[0][:60])
                    continue
            elif not all_units(r, units).isdisjoint(base_units):
                drop_base += 1  # held-out record the probe also evaluates -> keep our test disjoint from it
                continue
            kept[r["split"]].append(r)
        files = {}
        for split in ("train", "dev", "test"):
            if not kept[split]:
                continue
            path = out_dir / split / f"{name}.jsonl"
            n = write_jsonl(path, kept[split])
            files[split] = {"path": str(path.relative_to(out_dir)), "records": n}
        all_kept = [r for s in kept.values() for r in s]
        manifest_sources[name] = {
            "lang": all_kept[0]["lang"] if all_kept else None,
            "eval_only": name in EVAL_ONLY,
            "files": files,
            "by_type": dict(Counter(r["type"] for r in all_kept)),
            "by_split_type": {s: dict(Counter(r["type"] for r in v)) for s, v in kept.items()},
            "noul_true_rate": {s: round(sum(r["label"] for r in v if r["type"] == "noul")
                                        / max(1, sum(r["type"] == "noul" for r in v)), 3) for s, v in kept.items()},
            "dropped_duplicates": drop_dup,
            "dropped_train_contaminated": drop_contam,
            "dropped_train_inside_eval_only": drop_inside,  # (subset of the above) found only by containment
            "inside_eval_only_examples": inside_examples,
            "dropped_eval_overlapping_probes": drop_base,
        }
        log(f"  {name:18s} dup-{drop_dup:<5d} contam-{drop_contam:<5d} probe-{drop_base:<3d} "
            f"kept {dict((s, len(v)) for s, v in kept.items())}")
        if drop_inside:
            log(f"  {'':18s} of which {drop_inside} only inside eval-only texts, e.g. {inside_examples}")

    # 4) manifest + SHA256SUMS
    manifest = {"corpus_version": CORPUS_VERSION, "sources": manifest_sources, "train_sample": sample}
    manifest["schema"] = {"id": "str", "lang": "zh|en", "type": "choice|score|noul", "question": "str",
                          "options": "list[str] (choice: candidates; score: ordered levels low->high; absent for noul)",
                          "context": "str (optional)", "label": "int index into options | bool for noul",
                          "source": "str", "split": "train|dev|test",
                          "view": "dict (optional): {kind, text[], task, arg?}, see zhjudge.data.views",
                          "group": "str (optional, train only): records the calibration slice takes or leaves together"}
    manifest["split_policy"] = ("train<-official train; dev<-official validation when a labelled official test exists; "
                                "test<-official labelled test, else official validation; hash split (sha256) when no "
                                "official split")
    manifest["split_sources"] = {k: split_doc(k, sample) for k in selected}
    manifest["allowlist"] = {"file": cfg["data"]["allowlist"], "notes": notes,
                             "smoke_allow_disabled": bool(cfg["data"].get("allow_disabled")),
                             "licenses": cfg["data"].get("licenses") if licenses else None,
                             "license_profile": cfg["data"].get("license_profile") if licenses else None}
    manifest["decontamination"] = {
        "mode": mode, "min_unit_norm_chars": MIN_UNIT, "external_eval_units": dict(ext_per),
        "external_sets": ([f"clue/clue:{c}:validation+test" for c in CLUE_CFGS] + ["LocalLLaMA/typed-decisions:all"]
                          if mode == "full" else [])
        + [f"data/probes/{p.name} (state strings; also removed from our dev/test)"
           for p in sorted(P["probes"].glob('*.jsonl'))],
        "containment": "train texts (>= 12 normalised chars) found inside an eval-only source's text are dropped",
        "also": "nli26_zh_mnli drops rows whose English premise is in MNLI validation_matched/mismatched; "
                "csl_zh drops papers whose abstract is in CLUE csl validation/test"}
    manifest["downloads"] = sorted(DOWNLOADED.values(), key=lambda d: (d["repo"], d["file"]))
    manifest["download_bytes_total"] = sum(d["bytes"] for d in DOWNLOADED.values())
    manifest["revisions"] = REVISIONS
    files = sorted(out_dir.rglob("*.jsonl"))
    sums, totals = {}, Counter()
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rel = str(p.relative_to(out_dir))
        sums[rel] = h
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                totals[(r["lang"], r["split"])] += 1
    manifest["sha256"] = sums
    manifest["corpus_sha256"] = hashlib.sha256("".join(f"{h}  {rel}\n" for rel, h in sums.items()).encode()).hexdigest()
    manifest["totals"] = {f"{lang}/{split}": n for (lang, split), n in sorted(totals.items())}
    manifest["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "SHA256SUMS").write_text("".join(f"{h}  {rel}\n" for rel, h in sums.items()), encoding="utf-8")
    print(json.dumps(manifest["totals"], indent=1))
    log(f"done in {time.time() - t0:.0f}s; downloads {manifest['download_bytes_total'] / 1e6:.0f} MB; "
        f"corpus sha256 {manifest['corpus_sha256'][:16]}")
    return manifest


def _clue_csl_abstracts():
    """Only the CLUE csl abstracts (selected mode with csl_zh)."""
    keys = set()
    for sp in ("validation", "test"):
        df = pd.read_parquet(dl("clue/clue", f"csl/{sp}-00000-of-00001.parquet"))
        keys.update(k for k in (norm_key(s) for s in df["abst"]) if len(k) >= MIN_UNIT)
    return set(), frozenset(keys), Counter()


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args(argv)
    build(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
