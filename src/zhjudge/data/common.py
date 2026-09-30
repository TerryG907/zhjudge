"""Shared helpers for the unified typed-decision JSONL (ported unchanged from the phase-0 spike,
except that output paths now come from the config instead of module constants).

Unified record (one JSON object per line):
  id       str   "<source>/<split>/<row>[/<variant>]" — stable across rebuilds
  lang     str   "zh" | "en"
  type     str   "choice" | "score" | "noul"
  question str   the instruction the model answers (Chinese for zh sources, English for en)
  options  list[str]  choice: candidate answers (order is the index space of `label`)
                      score : ordered levels, low -> high (index = level)
                      noul  : omitted
  context  str   the state/text being judged (omitted when the question is self-contained)
  label    int | bool   choice/score: index into options; noul: true/false
  source   str   converter name (see SOURCES in build.py / configs/datasets.yaml)
  split    str   "train" | "dev" | "test"
  view     dict  (optional) the example's texts and task, for Laya-style re-rendering at training time
                 (see zhjudge.data.views; the plain record above is what evaluation always uses)
  group    str   (optional, train only) records about the same example (e.g. one oasst2 message tree): the
                 calibration slice takes or leaves them together

Split policy:
  train <- official train split only
  dev   <- official validation split, only when a labelled official test split also exists
  test  <- official labelled test split; if the official test is unlabelled/absent, the official
           validation split becomes our test (and there is no dev)
  Sources without official splits use a deterministic sha256 hash split of a stable key.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from huggingface_hub import hf_hub_download

SEED = 20260924

# Pinned dataset revisions (repo sha observed 2026-09-24). Every download goes through dl().
REVISIONS = {
    "google-research-datasets/paws-x": "4cd8187c404bda33cb1f62b49b001115862acf37",
    "mteb/amazon_massive_intent": "940fd47a81eaa7f2cc7b129674d945d618ac38c2",
    "Davlan/sib200": "38977a667f6fc264d5c26ec57a01e16db040b358",
    "MoritzLaurer/multilingual-NLI-26lang-2mil7": "510a233972a0d7ff0f767d82f46e046832c10538",
    "facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    "cambridgeltl/xcopa": "042f78955ba48e6404616762fa6e05e839c3907a",
    "Muennighoff/xwinograd": "90b619ef5278605cd4f572e3e8506ab16afd44f2",
    "CohereLabs/Global-MMLU": "0e619dbeb34206cd48705a1a0ea7fb21cae09993",
    "PolyAI/minds14": "40ce77cb32a384e4d50a568e1ec39ac804019d33",
    "legacy-datasets/banking77": "f54121560de48f2852f90be299010d1d6dc612ec",
    "google/boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "fancyzhx/dbpedia_14": "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
    "nyu-mll/multi_nli": "da70db2af9d09693783c3320c4249840212ee221",
    "google-research-datasets/paws": "161ece9501cf0a11f3e48bd356eaa82de46d6a09",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "google-research-datasets/go_emotions": "add492243ff905527e67aeb8b80c082af02207c3",
    "google/civil_comments": "f2970eb3a55777454c94069077cc8d9b5866312d",
    "nvidia/HelpSteer2": "990b2711a36180dd19d9c94b8627844866f8982a",
    "OpenAssistant/oasst2": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
    "neuclir/csl": "b4e118b56ca678375f5072b6328ded1ecb766083",
    "ConvLab/crosswoz": "4a3e56082543ed9eecb9c76ef5eadc1aa0cc5ca0",
    "thu-coai/cold": "a31e56eb008ac3c6abe987094fb06a07fc53625f",
    "LocalLLaMA/typed-decisions": "c76749ec58bd8c3d2ea706b31c333a9059c38f90",
    # only used for decontamination (never converted); pinned to the revision the spike's probe set used
    "clue/clue": "28178267a609dd08bdc703dd6c931dfc2c2f4431",
}

DOWNLOADED: dict[str, dict] = {}

# data.train_sample (build sets it before converting): TRAIN-split sample sizes and switches of single converters.
# Held-out splits never depend on it, so dev/test files are byte-identical with and without it.
TRAIN_SAMPLE: dict = {}
TRAIN_SAMPLE_KEYS = {
    "mnli_en": "train pairs sampled from MNLI train (default 12000; a larger sample contains the default one)",
    "goemotions_en_valence": "GoEmotions train comments not used by goemotions_en, asked for polarity / valence "
                             "(0 = none)",
    "oasst2_zh_multi_attr": "also the other rated attributes (quality, helpfulness, creativity) of oasst2_zh replies",
    "oasst2_zh_prompts": "also the quality of rated oasst2_zh user prompts",
}


def train_n(key, default):
    return TRAIN_SAMPLE.get(key, default)


def dl(repo: str, filename: str) -> str:
    rev = REVISIONS.get(repo)
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=rev)
    DOWNLOADED[f"{repo}/{filename}"] = {"repo": repo, "file": filename, "revision": rev,
                                         "bytes": Path(path).stat().st_size}
    return path


def rng_for(*key) -> random.Random:
    h = hashlib.sha256(":".join(map(str, (SEED,) + key)).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def hash_split(key: str, test_frac: float = 0.02, dev_frac: float = 0.02) -> str:
    """Deterministic split for sources without official splits."""
    x = int.from_bytes(hashlib.sha256(f"split:{key}".encode()).digest()[:8], "big") / 2**64
    if x < test_frac:
        return "test"
    if x < test_frac + dev_frac:
        return "dev"
    return "train"


_ws = re.compile(r"\s+")


def clean(text) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("　", " ").replace("\r", "")
    return _ws.sub(" ", t).strip()


def clean_keep_lines(text) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text)).replace("\r", "")
    lines = [_ws.sub(" ", ln).strip() for ln in t.split("\n")]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def norm_key(text: str) -> str:
    """Normalisation used for de-duplication / contamination checks."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"[\s\W_]+", "", t)


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def make(source, split, row, lang, typ, question, label, options=None, context=None, variant=None, view=None,
         group=None):
    assert typ in ("choice", "score", "noul"), typ
    assert split in ("train", "dev", "test"), split
    assert group is None or (split == "train" and isinstance(group, str) and group), (split, group)
    rid = f"{source}/{split}/{row}" + (f"/{variant}" if variant is not None else "")
    rec = {"id": rid, "lang": lang, "type": typ, "question": question}
    if typ == "noul":
        assert isinstance(label, bool) and options is None, (rid, label)
    else:
        assert options and isinstance(label, int) and 0 <= label < len(options), (rid, label, options)
        assert len(set(options)) == len(options), (rid, options)
        rec["options"] = list(options)
    if context:
        rec["context"] = context
    rec["label"] = label
    rec["source"] = source
    rec["split"] = split
    if view is not None:
        from .views import check

        check(view, typ)
        rec["view"] = view
    if group is not None:
        rec["group"] = group
    return rec


def choice_subset(rng: random.Random, all_labels: list[str], gold: str, kmin=4, kmax=12):
    """Random candidate subset containing the gold label, shuffled. Returns (options, label_idx)."""
    k = min(len(all_labels), rng.randint(kmin, kmax))
    others = [x for x in all_labels if x != gold]
    opts = rng.sample(others, k - 1) + [gold]
    rng.shuffle(opts)
    return opts, opts.index(gold)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    return n


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 23), b""):
            h.update(blk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------
# Text units used for leakage / contamination checks (shared by build.py and validate.py)
# ----------------------------------------------------------------------------------------------------
MIN_UNIT = 8  # normalised chars; shorter units are too generic to count as overlap
_ROLE = re.compile(r"(?:^|\n)\s*(?:句子[AB]|前提|假设|用户|系统|助手|标题|摘要|Sentence [AB]|User|Assistant)\s*[:：]\s*")


def record_units(rec) -> set[str]:
    """Normalised text units of a WRITTEN record: the whole context plus each role-prefixed segment of it
    (句子A/句子B, 前提/假设, 系统/用户, 标题/摘要, User/Assistant ...) and each text of its view (e.g. an NLI
    hypothesis, a BoolQ question). For context-free multiple choice the question itself is the item."""
    ctx = rec.get("context", "") or ""
    parts = [ctx] + [p for p in _ROLE.split(ctx) if p] + list((rec.get("view") or {}).get("text", []))
    if rec["type"] == "choice" and not ctx:
        parts.append(rec["question"])
    return {k for k in (norm_key(p) for p in parts) if len(k) >= MIN_UNIT}


MIN_CONTAINED = 12  # normalised chars: a train text this long found INSIDE an eval-only text counts as leaked


class ContainmentIndex:
    """Normalised texts of EVAL-ONLY records (e.g. Belebele passages = runs of FLORES-200 sentences). hit(u) is True
    when unit u (>= MIN_CONTAINED chars) occurs anywhere inside one of them, whatever the sentence boundaries
    (exact whole-unit matching misses a SIB-200 train sentence that is one sentence of a Belebele passage)."""

    def __init__(self, texts, k=MIN_CONTAINED):
        self.k, self.docs, self.first = k, [], {}
        for t in texts:
            d = norm_key(t)
            if len(d) < k:
                continue
            j = len(self.docs)
            self.docs.append(d)
            for i in range(len(d) - k + 1):
                self.first.setdefault(d[i:i + k], []).append(j)

    def hit(self, unit: str) -> bool:
        if len(unit) < self.k:
            return False
        a, b = self.first.get(unit[:self.k]), self.first.get(unit[-self.k:])
        if not a or not b:
            return False
        return any(unit in self.docs[j] for j in set(a) & set(b))


def _strings(x):
    """Every string value inside a nested state (dict values only: keys are schema field names)."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)) or hasattr(x, "tolist"):
        for v in (x.tolist() if hasattr(x, "tolist") else x):
            yield from _strings(v)


def probe_units(probe_dir: Path):
    """Text units of every item of the baseline probe sets (data/probes/*.jsonl).

    Our train split must not contain them, and our dev/test splits must not overlap them either (the probe
    uses official validation splits as its dev sets; our held-out sets must stay disjoint)."""
    from collections import Counter

    units, per = set(), Counter()
    for p in sorted(Path(probe_dir).glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # zh_probe/en_control/xnli keep the state under q_native/q_en; typed_decisions_test at top level
                for st in [(r.get(fld) or {}).get("state") for fld in ("q_native", "q_en")] + [r.get("state")]:
                    for s in _strings(st):
                        for piece in {s, *s.split("\n")}:  # whole string and each of its lines
                            k = norm_key(piece)
                            if len(k) >= MIN_UNIT:
                                units.add(k)
                                per[f"probes:{p.name}"] += 1
    return units, per
