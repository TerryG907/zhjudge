"""Converters for PERMISSIVE English sources (bilingual mix) -> unified typed-decision records."""
from __future__ import annotations

import gzip
import json

import pandas as pd
import pyarrow.parquet as pq

from .common import choice_subset, clean, clean_keep_lines, dl, make, rng_for, train_n, truncate
from .views import view
from . import zh as _zh


def label_names(parquet_path, col):
    """Class names stored in the HF parquet schema metadata."""
    meta = pq.read_schema(parquet_path).metadata or {}
    info = json.loads(meta.get(b"huggingface", b"{}"))
    feat = info.get("info", {}).get("features", {}).get(col, {})
    if feat.get("_type") == "ClassLabel":
        return feat["names"]
    if feat.get("_type") == "Sequence" and feat.get("feature", {}).get("_type") == "ClassLabel":
        return feat["feature"]["names"]
    if "names" in feat:
        return feat["names"]
    raise KeyError(f"no ClassLabel for {col} in {parquet_path}")


def _sample(df, n, *key):
    """n rows in the order of one seeded permutation (pandas: RandomState.choice without replacement = a prefix of
    permutation(len(df))), so a larger n for the same key contains the smaller sample, in the same order."""
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=rng_for(*key).randint(0, 2**31 - 1))


# ---------------- banking77 (legacy-datasets/banking77) CC BY 4.0 ----------------
def banking77_en():
    src = "banking77_en"
    for off, split in (("train", "train"), ("test", "test")):
        p = dl("legacy-datasets/banking77", f"data/{off}-00000-of-00001.parquet")
        names = [n.replace("_", " ") for n in label_names(p, "label")]
        df = pd.read_parquet(p)
        for i, r in df.iterrows():
            t = clean(r["text"])
            rng = rng_for(src, split, i)
            gold = names[int(r["label"])]
            if split == "test" or rng.random() < 0.5:
                opts = list(names)
                rng.shuffle(opts)
                lab = opts.index(gold)
            else:
                opts, lab = choice_subset(rng, names, gold)
            yield make(src, split, i, "en", "choice", "Which banking intent best describes this customer message?", lab, opts, t,
                       view=view("message", [t], "banking.intent") if t else None), [t]


# ---------------- BoolQ (google/boolq) CC BY-SA 3.0 ----------------
def boolq_en():
    src = "boolq_en"
    for off, split in (("train", "train"), ("validation", "test")):
        df = pd.read_parquet(dl("google/boolq", f"data/{off}-00000-of-00001.parquet"))
        for i, r in df.iterrows():
            q = clean(r["question"]).rstrip("?")
            q = q[0].upper() + q[1:] + "?"
            ctx = clean(r["passage"])
            yield make(src, split, i, "en", "noul", q, bool(r["answer"]), context=ctx,
                       view=view("passage_q", [ctx, q], "boolq") if ctx else None), [ctx]


# ---------------- DBpedia-14 (fancyzhx/dbpedia_14) CC BY-SA 3.0 + GFDL ----------------
def dbpedia14_en():
    src = "dbpedia14_en"
    for off, split, n in (("train", "train", 10000), ("test", "test", 2000)):
        p = dl("fancyzhx/dbpedia_14", f"dbpedia_14/{off}-00000-of-00001.parquet")
        names = label_names(p, "label")
        pretty = {"Company": "company", "EducationalInstitution": "educational institution", "Artist": "artist",
                  "Athlete": "athlete", "OfficeHolder": "office holder / politician", "MeanOfTransportation": "means of transportation",
                  "Building": "building", "NaturalPlace": "natural place", "Village": "village", "Animal": "animal",
                  "Plant": "plant", "Album": "music album", "Film": "film", "WrittenWork": "written work"}
        names = [pretty.get(n, n) for n in names]
        df = _sample(pd.read_parquet(p), n, src, split)
        for i, r in df.iterrows():
            title, content = clean(r["title"]), clean(r["content"])
            ctx = f"{title}: {content}"
            rng = rng_for(src, split, i)
            opts = list(names)
            rng.shuffle(opts)
            yield make(src, split, i, "en", "choice", "What kind of entity is this encyclopedia entry about?",
                       opts.index(names[int(r["label"])]), opts, ctx,
                       view=view("entry", [title, content], "dbpedia") if title and content else None), [content]


# ---------------- MultiNLI (nyu-mll/multi_nli) OANC / CC BY 3.0 / CC BY-SA 3.0 / PD ----------------
NLI_OPTS_EN = ["entailment: if the premise is true, the hypothesis must be true",
               "neutral: the hypothesis may or may not be true",
               "contradiction: if the premise is true, the hypothesis must be false"]


def mnli_premises_eval():
    """Normalised premises of MNLI validation_matched + validation_mismatched (for zh-translation decontamination)."""
    from .common import norm_key
    keys = set()
    for off in ("validation_matched", "validation_mismatched"):
        df = pd.read_parquet(dl("nyu-mll/multi_nli", f"data/{off}-00000-of-00001.parquet"), columns=["premise"])
        keys.update(norm_key(x) for x in df["premise"])
    return frozenset(keys)


def mnli_en():
    src = "mnli_en"
    # validation_matched is the baselines probe's English NLI dev set (nyu-mll/glue:mnli validation_matched), so it is
    # never used here: test <- validation_mismatched, no dev.
    for off, split, n in (("train", "train", train_n("mnli_en", 12000)), ("validation_mismatched", "test", 2000)):
        df = pd.read_parquet(dl("nyu-mll/multi_nli", f"data/{off}-00000-of-00001.parquet"), columns=["premise", "hypothesis", "label"])
        df = _sample(df[df["label"].isin([0, 1, 2])], n, src, split)
        for i, r in df.iterrows():
            p, h, y = clean(r["premise"]), clean(r["hypothesis"]), int(r["label"])
            rng = rng_for(src, split, i)
            if split == "test" or rng.random() < 0.7:
                yield make(src, split, i, "en", "choice", f'Hypothesis: "{h}" How does it relate to the text?', y, NLI_OPTS_EN, p, "rel",
                           view("nli", [p, h], "nli.rel") if p and h else None), [p, h]
            else:
                yield make(src, split, i, "en", "noul", f'If the text is true, must "{h}" also be true?', y == 0, context=p,
                           variant="entail_yn", view=view("nli", [p, h], "nli.entail_yn") if p and h else None), [p, h]


# ---------------- PAWS (google-research-datasets/paws, labeled_final) "freely used for any purpose" ----------------
PAWS_Q = ["Do sentence A and sentence B mean the same thing?", "Is sentence B a paraphrase of sentence A?",
          "Do these two sentences have the same meaning?"]


def paws_en():
    src = "paws_en"
    for off, split, n in (("train", "train", 10000), ("validation", "dev", 1000), ("test", "test", 2000)):
        df = _sample(pd.read_parquet(dl("google-research-datasets/paws", f"labeled_final/{off}-00000-of-00001.parquet")), n, src, split)
        for _, r in df.iterrows():
            a, b = clean(r["sentence1"]), clean(r["sentence2"])
            rng = rng_for(src, split, r["id"])
            yield make(src, split, r["id"], "en", "noul", rng.choice(PAWS_Q), int(r["label"]) == 1,
                       context=f"Sentence A: {a}\nSentence B: {b}",
                       view=view("pair", [a, b], "paraphrase") if a and b else None), [a, b]


# ---------------- CommonsenseQA (tau/commonsense_qa) MIT ----------------
def csqa_en():
    src = "csqa_en"
    for off, split in (("train", "train"), ("validation", "test")):
        df = pd.read_parquet(dl("tau/commonsense_qa", f"data/{off}-00000-of-00001.parquet"))
        for _, r in df.iterrows():
            labels, texts = list(r["choices"]["label"]), [clean(t) for t in r["choices"]["text"]]
            if not r["answerKey"] or len(set(texts)) != len(texts):
                continue
            q = clean(r["question"])
            yield make(src, split, r["id"], "en", "choice", q, labels.index(r["answerKey"]), texts,
                       view=view("question", [q], "mc.answer") if q else None), [q]


# ---------------- ARC (allenai/ai2_arc) CC BY-SA 4.0 ----------------
def arc_en():
    src = "arc_en"
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        for off, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
            df = pd.read_parquet(dl("allenai/ai2_arc", f"{cfg}/{off}-00000-of-00001.parquet"))
            for _, r in df.iterrows():
                labels, texts = list(r["choices"]["label"]), [clean(t) for t in r["choices"]["text"]]
                if r["answerKey"] not in labels or len(set(texts)) != len(texts):
                    continue
                q = clean(r["question"])
                yield make(src, split, r["id"], "en", "choice", q, labels.index(r["answerKey"]), texts,
                           view=view("question", [q], "mc.answer") if q else None), [q]


# ---------------- GoEmotions (google-research-datasets/go_emotions, simplified) Apache-2.0 ----------------
GOEMO_TRAIN_N = 10000


def goemotions_en():
    src = "goemotions_en"
    for off, split, n in (("train", "train", GOEMO_TRAIN_N), ("validation", "dev", 1000), ("test", "test", 2000)):
        p = dl("google-research-datasets/go_emotions", f"simplified/{off}-00000-of-00001.parquet")
        names = label_names(p, "labels")
        df = _sample(pd.read_parquet(p), n, src, split)
        for _, r in df.iterrows():
            t = clean(r["text"])
            labs = [names[int(x)] for x in r["labels"]]
            rng = rng_for(src, split, r["id"])
            if len(labs) == 1 and (split == "test" or rng.random() < 0.6):
                opts = list(names)
                rng.shuffle(opts)
                yield make(src, split, r["id"], "en", "choice", "Which emotion does this comment express most clearly?",
                           opts.index(labs[0]), opts, t, "emotion", view("comment", [t], "goemo.emotion") if t else None), [t]
            else:
                pos = rng.random() < 0.5
                emo = rng.choice(labs) if pos else rng.choice([x for x in names if x not in labs])
                yield make(src, split, r["id"], "en", "noul", f"Does this comment express {emo}?", pos, context=t, variant="emo_yn",
                           view=view("comment", [t], "goemo.emo_yn", emo) if t else None), [t]
    if train_n("goemotions_en_valence", 0):
        yield from goemotions_valence(int(train_n("goemotions_en_valence", 0)))


# the dataset authors' grouping of their 27 emotions (google-research/goemotions data/sentiment_mapping.json)
GOEMO_SENTIMENT = {
    "positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration",
                 "gratitude", "relief", "approval"],
    "negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust",
                 "anger", "annoyance", "disapproval"],
    "ambiguous": ["realization", "surprise", "curiosity", "confusion"],
}
POLARITY_EN = ["positive", "negative", "neutral"]
VALENCE_EN = ["clearly negative", "neutral", "clearly positive"]  # score levels, low -> high


def goemo_polarity(labels):
    """+1 / -1 when every emotion of the comment is in the authors' positive / negative group, 0 for `neutral` alone,
    None otherwise (an ambiguous emotion, both groups, or neutral next to an emotion)."""
    s = set(labels)
    if s == {"neutral"}:
        return 0
    for pol, grp in ((1, "positive"), (-1, "negative")):
        if s and s <= set(GOEMO_SENTIMENT[grp]):
            return pol
    return None


def goemotions_valence(n):
    """TRAIN only: n GoEmotions train comments that goemotions_en does not use, asked for their overall polarity
    (choice) or valence (3-level score) from the authors' grouping of their human emotion labels."""
    src = "goemotions_en"
    p = dl("google-research-datasets/go_emotions", "simplified/train-00000-of-00001.parquet")
    names = label_names(p, "labels")
    df = pd.read_parquet(p)
    used = set(_sample(df, GOEMO_TRAIN_N, src, "train")["id"])  # goemotions_en's own train sample
    for _, r in _sample(df[~df["id"].isin(used)], n, src, "train", "valence").iterrows():
        t = clean(r["text"])
        pol = goemo_polarity([names[int(x)] for x in r["labels"]])
        if not t or pol is None:
            continue
        rng = rng_for(src, "train", r["id"], "valence")
        if rng.random() < 0.5:
            opts = list(POLARITY_EN)
            rng.shuffle(opts)
            gold = {1: "positive", -1: "negative", 0: "neutral"}[pol]
            yield make(src, "train", r["id"], "en", "choice", "Is the overall tone of this comment positive, negative "
                       "or neutral?", opts.index(gold), opts, t, "polarity",
                       view("comment", [t], "goemo.polarity")), [t]
        else:
            yield make(src, "train", r["id"], "en", "score", "How positive is the overall tone of this comment?",
                       pol + 1, VALENCE_EN, t, "valence", view("comment", [t], "goemo.valence")), [t]


# ---------------- Civil Comments (google/civil_comments) CC0 1.0 ----------------
CIVIL_LEVELS = ["almost no readers (<10%)", "a few readers (10-30%)", "some readers (30-50%)", "most readers (50-80%)",
                "nearly all readers (>=80%)"]


def _civil_bin(x):
    return 0 if x < 0.1 else 1 if x < 0.3 else 2 if x < 0.5 else 3 if x < 0.8 else 4


def civil_en():
    src = "civil_en"
    for fname, split, per_bin in (("data/train-00000-of-00002.parquet", "train", 1600), ("data/test-00000-of-00001.parquet", "test", 400)):
        df = pd.read_parquet(dl("google/civil_comments", fname), columns=["text", "toxicity"])
        df = df[df["text"].str.len().between(20, 1500)]
        df["bin"] = df["toxicity"].map(_civil_bin)
        parts = [_sample(df[df["bin"] == b], per_bin, src, split, b) for b in range(5)]
        for b, part in enumerate(parts):
            for i, r in part.iterrows():
                t = clean(r["text"])
                rng = rng_for(src, split, i)
                if split == "test" or rng.random() < 0.7:
                    yield make(src, split, i, "en", "score", "What share of readers would rate this comment as toxic?", b,
                               CIVIL_LEVELS, t, "tox", view("comment", [t], "civil.tox") if t else None), [t]
                else:
                    yield make(src, split, i, "en", "noul", "Would most readers consider this comment toxic?", float(r["toxicity"]) >= 0.5,
                               context=t, variant="tox_yn", view=view("comment", [t], "civil.tox_yn") if t else None), [t]


# ---------------- HelpSteer2 (nvidia/HelpSteer2) CC BY 4.0 — human 0-4 ratings ----------------
HS_Q = {
    "helpfulness": ("How helpful is the response to the user's prompt?",
                    ["not helpful", "slightly helpful", "moderately helpful", "very helpful", "extremely helpful"]),
    "correctness": ("How factually correct and complete is the response?",
                    ["mostly wrong", "several errors", "partly correct", "mostly correct", "fully correct"]),
    "coherence": ("How clear and coherent is the response?",
                  ["incoherent", "hard to follow", "somewhat clear", "clear", "perfectly clear"]),
    "complexity": ("How much expertise is needed to write this response?",
                   ["basic", "simple", "intermediate", "advanced", "expert"]),
    "verbosity": ("How verbose is the response relative to what was asked?",
                  ["very terse", "succinct", "moderate", "detailed", "very verbose"]),
}


def helpsteer2_en(cap_train=10000):
    src = "helpsteer2_en"
    for fname, split in (("train.jsonl.gz", "train"), ("validation.jsonl.gz", "test")):
        with gzip.open(dl("nvidia/HelpSteer2", fname), "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        idx = list(range(len(rows)))
        if split == "train" and len(idx) > cap_train:
            idx = sorted(rng_for(src, "cap").sample(idx, cap_train))
        for i in idx:
            r = rows[i]
            rng = rng_for(src, split, i)
            u = rng.random()
            attr = "helpfulness" if u < 0.5 else ["correctness", "coherence", "complexity", "verbosity"][int((u - 0.5) / 0.125) % 4]
            q, levels = HS_Q[attr]
            pt, rt = truncate(clean_keep_lines(r["prompt"]), 1500), truncate(clean_keep_lines(r["response"]), 2500)
            yield make(src, split, i, "en", "score", q, int(r[attr]), levels, f"User: {pt}\n\nAssistant: {rt}", attr,
                       view("chat", [pt, rt], f"hs.{attr}") if pt and rt else None), [rt]


# ---------------- MASSIVE en-US (CC BY 4.0) ----------------
def massive_en():
    yield from _zh._massive("en")


# ---------------- oasst2 en (Apache-2.0) ----------------
def oasst2_en():
    yield from _zh._oasst("en", cap_train=6000, cap_test=1000)
