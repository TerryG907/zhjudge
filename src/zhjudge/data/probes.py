"""Build the baseline probe sets (the ones the spike measured laya-multilingual on). Ported from the spike's
baselines/scripts/build_probes.py with every download pinned to the revision the spike recorded, so the items are
the same ones behind the baseline numbers in zhjudge/baselines.py.

  uv run zhjudge probes --config ...     -> data/probes/*.jsonl + provenance.json

Writes (JSONL, one decision per line):
  zh_probe.jsonl              600 Chinese items from gold-labelled public validation/dev splits
  en_control.jsonl            360 English items, same task families (control)
  xnli_parallel.jsonl         150 XNLI validation items x {en, zh}, identical items in both languages
  typed_decisions_test.jsonl  400 cases (2,000 decisions), LocalLLaMA/typed-decisions `all` test split

Every zh/en/xnli item carries two question variants:
  q_native : instructions/criteria/field names in the item's own language (zh items -> Chinese prompt)
  q_en     : English instructions/criteria/field names (for zh items: English prompt over Chinese content)

These sets are EVALUATION-ONLY. Several upstream sets are research-only / non-commercial or have no stated licence
(CLUE, LCQMC, ChnSentiCorp, Amazon reviews, XNLI, AG News, DREAM): they are downloaded at runtime, used locally to
measure the model and to decontaminate the training corpus, and never trained on or redistributed.
The spike's 64-case feishu_zh diagnostic (from the Laya GitHub repo) is not rebuilt here.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import urllib.request

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ..config import add_common_args, load_config, paths
from .common import sha256_file

SEED = 20260923

# revisions recorded in the spike's baselines/data/provenance.json
PINNED = {
    "clue/clue": "28178267a609dd08bdc703dd6c931dfc2c2f4431",
    "lansinuote/ChnSentiCorp": "b0c4c119c3fb33b8e735969202ef9ad13d717e5a",
    "SetFit/amazon_reviews_multi_zh": "184ac90d5511a7f6801cba99688892f440ece660",
    "C-MTEB/LCQMC": "17f9b096f80380fce5ed12a9be8be7784b337daf",
    "nyu-mll/glue": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "fancyzhx/ag_news": "eb185aade064a813bc0b7f42de02595523103ca4",
    "SetFit/amazon_reviews_multi_en": "ec73b665e4be0f567b69d39425355401cfe0d29b",
    "facebook/xnli": "b8dd5d7af51114dbda02c0e3f6133f332186418e",
    "LocalLLaMA/typed-decisions": "c76749ec58bd8c3d2ea706b31c333a9059c38f90",
}
# DREAM dev (github.com/nlpdata/dream); sha256 of the exact file the spike used
DREAM_URL = "https://raw.githubusercontent.com/nlpdata/dream/master/data/dev.json"
DREAM_SHA256 = "9d5af2e580d809c73872a7dd43fe93d0b07c6f6086b04a9a9a1917603009d961"
# CLUE label files the spike read from the CLUE repo (only the parts it used)
TNEWS_OFFICIAL = {"100": "news_story", "101": "news_culture", "102": "news_entertainment", "103": "news_sports",
                  "104": "news_finance", "106": "news_house", "107": "news_car", "108": "news_edu", "109": "news_tech",
                  "110": "news_military", "112": "news_travel", "113": "news_world", "114": "news_stock",
                  "115": "news_agriculture", "116": "news_game"}
IFLYTEK_ZH = {"95": "借贷", "106": "电商", "36": "小说", "20": "棋牌中心", "48": "音乐", "34": "新闻", "8": "公共交通",
              "9": "政务", "94": "股票", "91": "运动健身"}

PROVENANCE = {}


def pq_rows(repo, path):
    p = hf_hub_download(repo, path, repo_type="dataset", revision=PINNED[repo])
    PROVENANCE[f"{repo}/{path}"] = PINNED[repo]
    return pq.read_table(p).to_pylist()


def jsonl_rows(repo, path):
    p = hf_hub_download(repo, path, repo_type="dataset", revision=PINNED[repo])
    PROVENANCE[f"{repo}/{path}"] = PINNED[repo]
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def dream_dev(cache_dir):
    p = cache_dir / "dream_dev.json"
    if not p.exists() or sha256_file(p) != DREAM_SHA256:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DREAM_URL, timeout=60) as r:
            p.write_bytes(r.read())
    got = sha256_file(p)
    if got != DREAM_SHA256:
        raise SystemExit(f"DREAM dev.json changed upstream (sha256 {got}); the en_control set would differ")
    PROVENANCE[DREAM_URL] = DREAM_SHA256
    return json.load(open(p, encoding="utf-8"))


def balanced(rows, key, per_class, rng, classes=None):
    by = collections.defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    out = []
    for c in sorted(by) if classes is None else classes:
        pool = by[c]
        if len(pool) < per_class:
            raise ValueError(f"class {c} has {len(pool)} < {per_class}")
        out += rng.sample(pool, per_class)
    rng.shuffle(out)
    return out


def item(id_, lang, source, split, idx, family, qtype, q_native, gold_native, q_en, gold_en, note=""):
    return {"id": id_, "lang": lang, "source": source, "split": split, "source_idx": idx, "family": family,
            "qtype": qtype, "q_native": q_native, "gold_native": gold_native, "q_en": q_en, "gold_en": gold_en,
            "note": note}


def q(state, qtype, instructions, criteria=None):
    d = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        d["criteria"] = criteria
    return {"state": state, "question": d}


NLI_ZH = {"蕴含": "前提可以推出假设为真", "中立": "前提既不能推出假设为真，也不能推出假设为假", "矛盾": "前提可以推出假设为假"}
NLI_EN = {"entailment": "the premise implies the hypothesis is true",
          "neutral": "the premise neither implies nor contradicts the hypothesis",
          "contradiction": "the premise implies the hypothesis is false"}
NLI_MAP_ZH = {"entailment": "蕴含", "neutral": "中立", "contradiction": "矛盾"}


def nli_item(id_, lang, source, split, idx, prem, hyp, gold_en_key, native_zh):
    en = q({"premise": prem, "hypothesis": hyp}, "choice",
           "What is the relationship between `premise` and `hypothesis`?", dict(NLI_EN))
    if native_zh:
        nat = q({"前提": prem, "假设": hyp}, "choice", "`前提`和`假设`之间是什么关系？", dict(NLI_ZH))
        g = NLI_MAP_ZH[gold_en_key]
    else:
        nat, g = en, gold_en_key
    return item(id_, lang, source, split, idx, "nli", "choice", nat, g, en, gold_en_key)


SENT2_ZH = {"正面": "评价积极、满意", "负面": "评价消极、不满"}
SENT2_EN = {"positive": "favourable, satisfied", "negative": "unfavourable, dissatisfied"}
STARS_ZH = ["非常负面（1星）", "负面（2星）", "中性（3星）", "正面（4星）", "非常正面（5星）"]
STARS_EN = ["very negative (1 star)", "negative (2 stars)", "neutral (3 stars)", "positive (4 stars)",
            "very positive (5 stars)"]


def build_zh(rng):
    out = []
    # 1. C3 multiple-choice reading comprehension (CLUE c3 validation)
    rows = pq_rows("clue/clue", "c3/validation-00000-of-00001.parquet")
    for i, r in enumerate(rows):
        r["_i"] = i  # the parquet `id` is per passage and repeats across its questions
    cand = [r for r in rows if len(set(r["choice"])) == len(r["choice"]) and r["answer"] in r["choice"]]
    for r in rng.sample(cand, 100):
        ctx = "\n".join(r["context"])
        letters = "ABCD"[: len(r["choice"])]
        crit = {c: None for c in r["choice"]}
        nat = q({"材料": ctx, "问题": r["question"]}, "choice", "根据`材料`，哪个选项正确回答了`问题`？", crit)
        en = q({"passage": ctx, "question": r["question"]}, "choice",
               "Based on `passage`, which option correctly answers `question`?", dict(crit))
        out.append(item(f"zh-c3-{r['_i']}", "zh", "clue/clue:c3", "validation", r["_i"], "reading_mc", "choice",
                        nat, r["answer"], en, r["answer"], note=f"{len(letters)} options"))
    # 2. OCNLI (CLUE ocnli validation), labels neutral/entailment/contradiction
    rows = [r for r in pq_rows("clue/clue", "ocnli/validation-00000-of-00001.parquet") if r["label"] in (0, 1, 2)]
    names = ["neutral", "entailment", "contradiction"]
    for r in balanced(rows, lambda r: r["label"], 30, rng):
        out.append(nli_item(f"zh-ocnli-{r['idx']}", "zh", "clue/clue:ocnli", "validation", r["idx"],
                            r["sentence1"], r["sentence2"], names[r["label"]], True))
    # 3. TNEWS 15-way news topic (CLUE tnews validation == official dev)
    tn_codes = list(TNEWS_OFFICIAL)
    tn_zh = {"100": "故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经", "106": "房产", "107": "汽车",
             "108": "教育", "109": "科技", "110": "军事", "112": "旅游", "113": "国际", "114": "股票", "115": "农业",
             "116": "电竞游戏"}
    tn_en = {"100": "story", "101": "culture", "102": "entertainment", "103": "sports", "104": "finance",
             "106": "real estate", "107": "cars", "108": "education", "109": "technology", "110": "military",
             "112": "travel", "113": "world", "114": "stock market", "115": "agriculture", "116": "gaming"}
    rows = pq_rows("clue/clue", "tnews/validation-00000-of-00001.parquet")
    for r in balanced(rows, lambda r: r["label"], 6, rng):
        code = tn_codes[r["label"]]
        nat = q({"标题": r["sentence"]}, "choice", "`标题`属于哪个新闻类别？", {tn_zh[c]: None for c in tn_codes})
        en = q({"headline": r["sentence"]}, "choice", "Which news category does `headline` belong to?",
               {tn_en[c]: None for c in tn_codes})
        out.append(item(f"zh-tnews-{r['idx']}", "zh", "clue/clue:tnews", "validation", r["idx"], "topic", "choice",
                        nat, tn_zh[code], en, tn_en[code], note=f"15 options; official label {TNEWS_OFFICIAL[code]}"))
    # 4. IFLYTEK restricted to 10 distinct app categories (CLUE iflytek validation)
    keep = {95: "lending and loans", 106: "e-commerce shopping", 36: "novels and fiction", 20: "board and card games",
            48: "music", 34: "news", 8: "public transport", 9: "government services", 94: "stock trading",
            91: "sports and fitness"}
    rows = [r for r in pq_rows("clue/clue", "iflytek/validation-00000-of-00001.parquet") if r["label"] in keep]
    for r in balanced(rows, lambda r: r["label"], 6, rng, classes=sorted(keep)):
        nat = q({"应用描述": r["sentence"]}, "choice", "`应用描述`介绍的是哪一类App？", {IFLYTEK_ZH[str(c)]: None for c in keep})
        en = q({"app_description": r["sentence"]}, "choice", "Which category of app does `app_description` describe?",
               {keep[c]: None for c in keep})
        out.append(item(f"zh-iflytek-{r['idx']}", "zh", "clue/clue:iflytek", "validation", r["idx"], "topic", "choice",
                        nat, IFLYTEK_ZH[str(r["label"])], en, keep[r["label"]], note="10 of 119 categories"))
    # 5. ChnSentiCorp binary sentiment (validation)
    rows = pq_rows("lansinuote/ChnSentiCorp", "data/validation-00000-of-00001-405befbaa3bcf1a2.parquet")
    for i, r in enumerate(rows):
        r["_i"] = i
    for r in balanced(rows, lambda r: r["label"], 35, rng):
        gz, ge = ("正面", "positive") if r["label"] == 1 else ("负面", "negative")
        nat = q({"评论": r["text"]}, "choice", "`评论`的情感倾向是什么？", dict(SENT2_ZH))
        en = q({"review": r["text"]}, "choice", "What is the sentiment of `review`?", dict(SENT2_EN))
        out.append(item(f"zh-chnsenti-{r['_i']}", "zh", "lansinuote/ChnSentiCorp", "validation", r["_i"], "sentiment",
                        "choice", nat, gz, en, ge))
    # 6. Amazon reviews zh, 1-5 stars as an ordinal score (validation)
    rows = jsonl_rows("SetFit/amazon_reviews_multi_zh", "validation.jsonl")
    for i, r in enumerate(rows):
        r["_i"] = i
    for r in balanced(rows, lambda r: r["label"], 18, rng):
        nat = q({"评论": r["text"]}, "score", "`评论`表达的评价有多正面？", list(STARS_ZH))
        en = q({"review": r["text"]}, "score", "How positive is the rating expressed in `review`?", list(STARS_EN))
        out.append(item(f"zh-amazon-{r['_i']}", "zh", "SetFit/amazon_reviews_multi_zh", "validation", r["_i"],
                        "rating", "score", nat, int(r["label"]), en, int(r["label"])))
    # 7. LCQMC and AFQMC paraphrase as yes/no (validation)
    lc = pq_rows("C-MTEB/LCQMC", "data/validation-00000-of-00001-ae04bea7d65ea894.parquet")
    for i, r in enumerate(lc):
        r["_i"], r["label"] = i, int(r["score"])
    af = pq_rows("clue/clue", "afqmc/validation-00000-of-00001.parquet")
    for r in af:
        r["_i"] = r["idx"]
    for name, rows in (("lcqmc", lc), ("afqmc", af)):
        src = "C-MTEB/LCQMC" if name == "lcqmc" else "clue/clue:afqmc"
        for r in balanced(rows, lambda r: r["label"], 25, rng):
            nat = q({"句子1": r["sentence1"], "句子2": r["sentence2"]}, "noul", "`句子1`和`句子2`表达的意思相同吗？")
            en = q({"sentence1": r["sentence1"], "sentence2": r["sentence2"]}, "noul",
                   "Do `sentence1` and `sentence2` mean the same thing?")
            out.append(item(f"zh-{name}-{r['_i']}", "zh", src, "validation", r["_i"], "paraphrase", "noul",
                            nat, bool(r["label"]), en, bool(r["label"])))
    return out


def build_en(rng, cache_dir):
    out = []
    # DREAM dialogue MC (the English sister set of C3), dev split from nlpdata/dream on GitHub
    dream = dream_dev(cache_dir)
    flat = [(d[2], j, "\n".join(d[0]), qq) for d in dream for j, qq in enumerate(d[1])]
    flat = [f for f in flat if len(set(f[3]["choice"])) == len(f[3]["choice"]) and f[3]["answer"] in f[3]["choice"]]
    for did, j, ctx, qq in rng.sample(flat, 60):
        crit = {c: None for c in qq["choice"]}
        en = q({"passage": ctx, "question": qq["question"]}, "choice",
               "Based on `passage`, which option correctly answers `question`?", crit)
        out.append(item(f"en-dream-{did}-{j}", "en", "nlpdata/dream", "dev", f"{did}-{j}", "reading_mc", "choice",
                        en, qq["answer"], en, qq["answer"]))
    # MNLI validation_matched
    rows = [r for r in pq_rows("nyu-mll/glue", "mnli/validation_matched-00000-of-00001.parquet") if r["label"] in (0, 1, 2)]
    names = ["entailment", "neutral", "contradiction"]
    for r in balanced(rows, lambda r: r["label"], 20, rng):
        out.append(nli_item(f"en-mnli-{r['idx']}", "en", "nyu-mll/glue:mnli", "validation_matched", r["idx"],
                            r["premise"], r["hypothesis"], names[r["label"]], False))
    # AG News test (no validation split exists)
    ag = {"world": "world news and international politics", "sports": "sports",
          "business": "business and economy", "sci_tech": "science and technology"}
    agk = list(ag)
    rows = pq_rows("fancyzhx/ag_news", "data/test-00000-of-00001.parquet")
    for i, r in enumerate(rows):
        r["_i"] = i
    for r in balanced(rows, lambda r: r["label"], 15, rng):
        en = q({"article": r["text"]}, "choice", "What is the topic of `article`?", dict(ag))
        out.append(item(f"en-agnews-{r['_i']}", "en", "fancyzhx/ag_news", "test", r["_i"], "topic", "choice",
                        en, agk[r["label"]], en, agk[r["label"]]))
    # SST-2 validation (binary sentiment as 2-way choice)
    rows = pq_rows("nyu-mll/glue", "sst2/validation-00000-of-00001.parquet")
    for r in balanced(rows, lambda r: r["label"], 30, rng):
        g = "positive" if r["label"] == 1 else "negative"
        en = q({"review": r["sentence"]}, "choice", "What is the sentiment of `review`?", dict(SENT2_EN))
        out.append(item(f"en-sst2-{r['idx']}", "en", "nyu-mll/glue:sst2", "validation", r["idx"], "sentiment", "choice",
                        en, g, en, g))
    # Amazon reviews en 1-5 stars (validation)
    rows = jsonl_rows("SetFit/amazon_reviews_multi_en", "validation.jsonl")
    for i, r in enumerate(rows):
        r["_i"] = i
    for r in balanced(rows, lambda r: r["label"], 12, rng):
        en = q({"review": r["text"]}, "score", "How positive is the rating expressed in `review`?", list(STARS_EN))
        out.append(item(f"en-amazon-{r['_i']}", "en", "SetFit/amazon_reviews_multi_en", "validation", r["_i"], "rating",
                        "score", en, int(r["label"]), en, int(r["label"])))
    # QQP validation (paraphrase as yes/no)
    rows = pq_rows("nyu-mll/glue", "qqp/validation-00000-of-00001.parquet")
    for r in balanced(rows, lambda r: r["label"], 30, rng):
        en = q({"sentence1": r["question1"], "sentence2": r["question2"]}, "noul",
               "Do `sentence1` and `sentence2` mean the same thing?")
        out.append(item(f"en-qqp-{r['idx']}", "en", "nyu-mll/glue:qqp", "validation", r["idx"], "paraphrase", "noul",
                        en, bool(r["label"]), en, bool(r["label"])))
    return out


def build_xnli(rng):
    en = pq_rows("facebook/xnli", "en/validation-00000-of-00001.parquet")
    zh = pq_rows("facebook/xnli", "zh/validation-00000-of-00001.parquet")
    assert len(en) == len(zh) and all(a["label"] == b["label"] for a, b in zip(en, zh)), "xnli en/zh not aligned"
    names = ["entailment", "neutral", "contradiction"]
    by = collections.defaultdict(list)
    for i in range(len(en)):
        by[en[i]["label"]].append(i)
    pick = [i for c in sorted(by) for i in rng.sample(by[c], 50)]
    rng.shuffle(pick)
    out = []
    for i in pick:
        g = names[en[i]["label"]]
        out.append(nli_item(f"xnli-en-{i}", "en", "facebook/xnli:en", "validation", i, en[i]["premise"],
                            en[i]["hypothesis"], g, False))
        out.append(nli_item(f"xnli-zh-{i}", "zh", "facebook/xnli:zh", "validation", i, zh[i]["premise"],
                            zh[i]["hypothesis"], g, True))
    return out


def build_typed_decisions():
    rows = pq_rows("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet")
    return [{"id": r["id"], "workflow": r["workflow"], "state": json.loads(r["state"]),
             "questions": json.loads(r["questions"]), "gold": json.loads(r["gold"])} for r in rows]


def build_probes(cfg: dict) -> dict:
    out_dir = paths(cfg)["probes"]
    out_dir.mkdir(parents=True, exist_ok=True)
    sums = {}

    def dump(name, rows):
        p = out_dir / name
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sums[name] = sha256_file(p)
        print(f"wrote {name}: {len(rows)} rows", flush=True)

    zh = build_zh(random.Random(SEED))
    assert len({r["id"] for r in zh}) == len(zh)
    dump("zh_probe.jsonl", zh)
    dump("en_control.jsonl", build_en(random.Random(SEED + 1), out_dir / "raw"))
    dump("xnli_parallel.jsonl", build_xnli(random.Random(SEED + 2)))
    dump("typed_decisions_test.jsonl", build_typed_decisions())
    prov = {"revisions": PROVENANCE, "sha256": sums}
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return prov


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args(argv)
    build_probes(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
