"""Converters for PERMISSIVE Chinese sources -> unified typed-decision records.

Each converter yields (record, units) where `units` are the raw text fields of the example, used only by
build.py for de-duplication / contamination checks and never written out.
"""
from __future__ import annotations

import gzip
import json
import sys
import zipfile
from collections import defaultdict

import pandas as pd

from .common import (choice_subset, clean, clean_keep_lines, dl, hash_split, make, rng_for, train_n, truncate)
from .views import view

# ----------------------------------------------------------------------------------------------------
# PAWS-X zh  (google-research-datasets/paws-x, zh)  licence: "may be freely used for any purpose"
# ----------------------------------------------------------------------------------------------------
PAWSX_Q = ["句子A和句子B表达的意思相同吗？", "句子B是否是对句子A的同义改写（意思不变）？", "这两句话的含义是否一致？"]


def pawsx_zh():
    src = "pawsx_zh"
    for off_split, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        df = pd.read_parquet(dl("google-research-datasets/paws-x", f"zh/{off_split}-00000-of-00001.parquet"))
        for _, r in df.iterrows():
            a, b = clean(r["sentence1"]), clean(r["sentence2"])
            if not a or not b or str(r["label"]) not in ("0", "1"):
                continue
            rng = rng_for(src, split, r["id"])
            ctx = f"句子A：{a}\n句子B：{b}"
            yield make(src, split, r["id"], "zh", "noul", rng.choice(PAWSX_Q), str(r["label"]) == "1", context=ctx,
                       view=view("pair", [a, b], "paraphrase")), [a, b]


# ----------------------------------------------------------------------------------------------------
# MASSIVE zh-CN  (AmazonScience/massive via mteb/amazon_massive_intent)  licence: CC BY 4.0
# ----------------------------------------------------------------------------------------------------
MASSIVE_ZH = {
    "alarm_query": "查询闹钟", "alarm_remove": "删除闹钟", "alarm_set": "设置闹钟", "audio_volume_down": "调低音量",
    "audio_volume_mute": "静音", "audio_volume_other": "其他音量设置", "audio_volume_up": "调高音量",
    "calendar_query": "查询日程", "calendar_remove": "删除日程", "calendar_set": "添加日程或提醒",
    "cooking_query": "烹饪相关问题", "cooking_recipe": "查询菜谱", "datetime_convert": "时间或时区换算",
    "datetime_query": "查询日期或时间", "email_addcontact": "添加邮件联系人", "email_query": "查看邮件",
    "email_querycontact": "查询联系人信息", "email_sendemail": "发送邮件", "general_greet": "打招呼",
    "general_joke": "讲笑话", "general_quirky": "闲聊或其他杂项", "iot_cleaning": "启动扫地机器人",
    "iot_coffee": "煮咖啡", "iot_hue_lightchange": "改变灯光颜色", "iot_hue_lightdim": "调暗灯光",
    "iot_hue_lightoff": "关灯", "iot_hue_lighton": "开灯", "iot_hue_lightup": "调亮灯光",
    "iot_wemo_off": "关闭智能插座", "iot_wemo_on": "打开智能插座", "lists_createoradd": "创建清单或添加条目",
    "lists_query": "查询清单", "lists_remove": "删除清单或条目", "music_dislikeness": "表示不喜欢某首音乐",
    "music_likeness": "表示喜欢某首音乐", "music_query": "查询正在播放的音乐", "music_settings": "音乐播放设置",
    "news_query": "查询新闻", "play_audiobook": "播放有声书", "play_game": "玩游戏", "play_music": "播放音乐",
    "play_podcasts": "播放播客", "play_radio": "播放电台", "qa_currency": "汇率或货币问题", "qa_definition": "查询词语定义",
    "qa_factoid": "事实性问答", "qa_maths": "数学计算", "qa_stock": "查询股票行情", "recommendation_events": "推荐活动",
    "recommendation_locations": "推荐地点", "recommendation_movies": "推荐电影", "social_post": "发布社交动态",
    "social_query": "查询社交动态", "takeaway_order": "点外卖", "takeaway_query": "查询外卖信息",
    "transport_query": "查询交通信息", "transport_taxi": "叫出租车", "transport_ticket": "订票",
    "transport_traffic": "查询路况", "weather_query": "查询天气",
}
MASSIVE_EN = {
    "alarm_query": "check alarms", "alarm_remove": "remove an alarm", "alarm_set": "set an alarm", "audio_volume_down": "turn volume down",
    "audio_volume_mute": "mute audio", "audio_volume_other": "other volume setting", "audio_volume_up": "turn volume up",
    "calendar_query": "check calendar", "calendar_remove": "remove calendar event", "calendar_set": "add calendar event or reminder",
    "cooking_query": "cooking question", "cooking_recipe": "find a recipe", "datetime_convert": "convert time or time zone",
    "datetime_query": "ask date or time", "email_addcontact": "add email contact", "email_query": "check email",
    "email_querycontact": "look up a contact", "email_sendemail": "send an email", "general_greet": "greeting",
    "general_joke": "tell a joke", "general_quirky": "chit-chat or other", "iot_cleaning": "start robot vacuum",
    "iot_coffee": "make coffee", "iot_hue_lightchange": "change light colour", "iot_hue_lightdim": "dim the lights",
    "iot_hue_lightoff": "turn lights off", "iot_hue_lighton": "turn lights on", "iot_hue_lightup": "brighten the lights",
    "iot_wemo_off": "turn smart plug off", "iot_wemo_on": "turn smart plug on", "lists_createoradd": "create list or add item",
    "lists_query": "check a list", "lists_remove": "remove list or item", "music_dislikeness": "dislike a song",
    "music_likeness": "like a song", "music_query": "ask what music is playing", "music_settings": "music playback settings",
    "news_query": "get news", "play_audiobook": "play audiobook", "play_game": "play a game", "play_music": "play music",
    "play_podcasts": "play podcast", "play_radio": "play radio", "qa_currency": "currency or exchange rate question",
    "qa_definition": "define a word", "qa_factoid": "factual question", "qa_maths": "maths calculation",
    "qa_stock": "stock price question", "recommendation_events": "recommend events", "recommendation_locations": "recommend places",
    "recommendation_movies": "recommend movies", "social_post": "post on social media", "social_query": "check social media",
    "takeaway_order": "order takeaway", "takeaway_query": "ask about takeaway", "transport_query": "transport information",
    "transport_taxi": "book a taxi", "transport_ticket": "book a ticket", "transport_traffic": "check traffic",
    "weather_query": "check weather",
}
SCEN_ZH = {"alarm": "闹钟", "audio": "音量", "calendar": "日程", "cooking": "烹饪", "datetime": "日期时间", "email": "邮件",
           "general": "闲聊", "iot": "智能家居", "lists": "清单", "music": "音乐", "news": "新闻", "play": "播放娱乐内容",
           "qa": "知识问答", "recommendation": "推荐", "social": "社交媒体", "takeaway": "外卖", "transport": "交通出行",
           "weather": "天气"}
SCEN_EN = {"alarm": "alarms", "audio": "audio volume", "calendar": "calendar", "cooking": "cooking", "datetime": "date and time",
           "email": "email", "general": "small talk", "iot": "smart home", "lists": "lists", "music": "music", "news": "news",
           "play": "playing media", "qa": "general questions", "recommendation": "recommendations", "social": "social media",
           "takeaway": "takeaway food", "transport": "transport", "weather": "weather"}


def _massive(lang):
    src = f"massive_{lang}"
    fname = "zh-CN" if lang == "zh" else "en"
    names = MASSIVE_ZH if lang == "zh" else MASSIVE_EN
    scen_names = SCEN_ZH if lang == "zh" else SCEN_EN
    q_intent = "用户这句话的意图是什么？" if lang == "zh" else "What does the user want the assistant to do?"
    q_scen = "这句话属于哪个场景？" if lang == "zh" else "Which domain does this request belong to?"
    q_noul = "用户是想让助手“{}”吗？" if lang == "zh" else "Is the user asking the assistant to: {}?"
    all_intents = [names[k] for k in sorted(names)]
    all_scen = [scen_names[k] for k in sorted(scen_names)]
    for off_split, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        with gzip.open(dl("mteb/amazon_massive_intent", f"{off_split}/{fname}.json.gz"), "rt") as f:
            rows = [json.loads(line) for line in f]
        for r in rows:
            text, intent = clean(r["text"]), r["label"]
            if not text or intent not in names:
                continue
            rng = rng_for(src, split, r["id"])
            gold = names[intent]
            scen = scen_names[intent.split("_")[0]]
            u = rng.random()
            if split == "test" or u < 0.6:
                if split == "test" or rng.random() < 0.5:
                    opts = list(all_intents)
                    rng.shuffle(opts)
                    lab = opts.index(gold)
                else:
                    opts, lab = choice_subset(rng, all_intents, gold)
                yield make(src, split, r["id"], lang, "choice", q_intent, lab, opts, text, "intent",
                           view("utterance", [text], "massive.intent")), [text]
            elif u < 0.8:
                opts = list(all_scen)
                rng.shuffle(opts)
                yield make(src, split, r["id"], lang, "choice", q_scen, opts.index(scen), opts, text, "scenario",
                           view("utterance", [text], "massive.scenario")), [text]
            else:
                pos = rng.random() < 0.5
                cand = gold if pos else rng.choice([x for x in all_intents if x != gold])
                yield make(src, split, r["id"], lang, "noul", q_noul.format(cand), pos, context=text, variant="intent_yn",
                           view=view("utterance", [text], "massive.intent_yn", cand)), [text]


def massive_zh():
    yield from _massive("zh")


# ----------------------------------------------------------------------------------------------------
# CrossWOZ (ConvLab/crosswoz unified format; thu-coai/CrossWOZ)  licence: Apache-2.0
# ----------------------------------------------------------------------------------------------------
CW_DOMAINS = {"景点": "景点", "餐馆": "餐馆", "酒店": "酒店", "地铁": "地铁", "出租": "出租车"}


def crosswoz_zh():
    src = "crosswoz_zh"
    z = zipfile.ZipFile(dl("ConvLab/crosswoz", "data.zip"))
    dialogues = json.loads(z.read("data/dialogues.json"))
    # request-able slots per domain (for slot-choice distractors)
    slots = defaultdict(set)
    for d in dialogues:
        for t in d["turns"]:
            for k in ("categorical", "non-categorical", "binary"):
                for a in t["dialogue_acts"][k]:
                    if a["intent"] == "Request" and a["domain"] in CW_DOMAINS and "酒店设施" not in a["slot"]:
                        slots[a["domain"]].add(a["slot"])
    slots = {k: sorted(v) for k, v in slots.items()}
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    for d in dialogues:
        split = split_map[d["data_split"]]
        prev_sys = None
        for t in d["turns"]:
            if t["speaker"] == "system":
                prev_sys = clean(t["utterance"])
                continue
            utt = clean(t["utterance"])
            acts = [a for k in ("categorical", "non-categorical", "binary") for a in t["dialogue_acts"][k]]
            if not utt or not acts:
                continue
            doms = {a["domain"] for a in acts if a["domain"] in CW_DOMAINS}
            reqs = [(a["domain"], a["slot"]) for a in acts if a["intent"] == "Request"]
            row = f'{d["dialogue_id"]}-{t["utt_idx"]}'
            rng = rng_for(src, row)
            ctx = (f"系统：{prev_sys}\n用户：{utt}" if prev_sys else f"用户：{utt}")
            kind, texts = ("turns", [prev_sys, utt]) if prev_sys else ("utterance", [utt])
            units = [utt]
            u = rng.random()
            if len(doms) == 1 and u < 0.5:
                opts = list(CW_DOMAINS.values())
                rng.shuffle(opts)
                gold = CW_DOMAINS[next(iter(doms))]
                yield make(src, split, row, "zh", "choice", "用户最后这句话涉及哪个领域？", opts.index(gold), opts, ctx, "domain",
                           view(kind, texts, "crosswoz.domain")), units
            elif len(reqs) == 1 and "酒店设施" not in reqs[0][1] and reqs[0][0] in slots and u < 0.75:
                dom, slot = reqs[0]
                opts, lab = choice_subset(rng, slots[dom], slot, kmin=3, kmax=6)
                yield make(src, split, row, "zh", "choice", "用户最后这句话在询问哪一项信息？", lab, opts, ctx, "slot",
                           view(kind, texts, "crosswoz.slot")), units
            else:
                yield make(src, split, row, "zh", "noul", "用户最后这句话是否在向系统询问某项信息？", bool(reqs), context=ctx,
                           variant="is_request", view=view(kind, texts, "crosswoz.is_request")), units


# ----------------------------------------------------------------------------------------------------
# COLD (thu-coai/cold, github thu-coai/COLDataset)  licence: Apache-2.0
# ----------------------------------------------------------------------------------------------------
COLD_TOPIC = {"race": "种族", "gender": "性别", "region": "地域"}
COLD_Q = ["这条评论是否具有冒犯性（如人身攻击、针对群体的歧视或侮辱）？", "这条言论是否冒犯他人？", "这段话含有攻击性或歧视性内容吗？"]


def cold_zh():
    src = "cold_zh"
    for fname, split in (("train.csv", "train"), ("dev.csv", "dev"), ("test.csv", "test")):
        df = pd.read_csv(dl("thu-coai/cold", fname))
        for i, r in df.iterrows():
            text = clean(r["TEXT"])
            if not text:
                continue
            row = f'{r["Unnamed: 0"]}-{i}'
            rng = rng_for(src, split, row)
            if rng.random() < 0.8 or r["topic"] not in COLD_TOPIC:
                yield make(src, split, row, "zh", "noul", rng.choice(COLD_Q), int(r["label"]) == 1, context=text,
                           view=view("comment", [text], "cold.offensive")), [text]
            else:
                opts = list(COLD_TOPIC.values())
                rng.shuffle(opts)
                yield make(src, split, row, "zh", "choice", "这条评论讨论的是哪类话题？", opts.index(COLD_TOPIC[r["topic"]]), opts,
                           text, "topic", view("comment", [text], "cold.topic")), [text]


# ----------------------------------------------------------------------------------------------------
# CSL — Chinese Scientific Literature (neuclir/csl, upstream ydli-ai/CSL)  licence: Apache-2.0
# Hash-sampled: ~12k train / ~1k dev / ~1.6k test papers out of 395,927.
# ----------------------------------------------------------------------------------------------------
def csl_zh(exclude_abstract_keys=frozenset()):
    from .common import norm_key
    src = "csl_zh"
    import hashlib
    papers = []
    with gzip.open(dl("neuclir/csl", "data/csl.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            x = int.from_bytes(hashlib.sha256(f"csl:{r['doc_id']}".encode()).digest()[:8], "big") / 2**64
            if x < 0.004:
                split = "test"
            elif x < 0.0065:
                split = "dev"
            elif x < 0.0365:
                split = "train"
            else:
                continue
            if norm_key(r["abstract"]) in exclude_abstract_keys:
                continue
            papers.append((split, r))
    cats = sorted({r["category"] for _, r in papers})
    discs = sorted({r["discipline"] for _, r in papers})
    kw_by_cat = defaultdict(list)
    for _, r in papers:
        kw_by_cat[r["category"]].extend(c for c in map(clean, r["keywords"]) if c)
    for split, r in papers:
        title, abstract = clean(r["title"]), clean(r["abstract"])
        if not abstract or not title:
            continue
        rng = rng_for(src, r["doc_id"])
        ctx = f"标题：{title}\n摘要：{truncate(abstract, 1200)}"
        texts = [title, truncate(abstract, 1200)]
        u = rng.random()
        if split == "test" or u < 0.5:
            opts = list(cats)
            rng.shuffle(opts)
            yield make(src, split, r["doc_id"], "zh", "choice", "这篇论文属于哪个学科门类？", opts.index(r["category"]), opts, ctx,
                       "category", view("paper", texts, "csl.category")), [abstract]
        elif u < 0.75:
            opts, lab = choice_subset(rng, discs, r["discipline"], kmin=4, kmax=10)
            yield make(src, split, r["doc_id"], "zh", "choice", "这篇论文最可能属于哪个一级学科？", lab, opts, ctx, "discipline",
                       view("paper", texts, "csl.discipline")), [abstract]
        else:
            kws = [c for c in map(clean, r["keywords"]) if c]
            if not kws:
                continue
            pos = rng.random() < 0.5
            if pos:
                kw = rng.choice(kws)
            else:
                pool = [k for k in kw_by_cat[r["category"]] if k not in kws and k not in title and k not in abstract]
                if not pool:
                    continue
                kw = rng.choice(pool)
            yield make(src, split, r["doc_id"], "zh", "noul", f"“{kw}”是否是这篇论文作者标注的关键词之一？", pos, context=ctx,
                       variant="keyword", view=view("paper", texts, "csl.keyword", kw)), [abstract]


# ----------------------------------------------------------------------------------------------------
# SIB-200 zho_Hans (Davlan/sib200)  licence: CC BY-SA 4.0
# ----------------------------------------------------------------------------------------------------
SIB_ZH = {"science/technology": "科学技术", "travel": "旅行", "politics": "政治", "sports": "体育", "health": "健康",
          "entertainment": "娱乐", "geography": "地理"}


def sib200_zh():
    src = "sib200_zh"
    for fname, split in (("train", "train"), ("dev", "dev"), ("test", "test")):
        df = pd.read_csv(dl("Davlan/sib200", f"data/zho_Hans/{fname}.tsv"), sep="\t")
        for _, r in df.iterrows():
            text = clean(r["text"])
            rng = rng_for(src, split, r["index_id"])
            opts = list(SIB_ZH.values())
            rng.shuffle(opts)
            yield make(src, split, r["index_id"], "zh", "choice", "这段文字的主题是什么？", opts.index(SIB_ZH[r["category"]]), opts,
                       text, view=view("text", [text], "sib.topic") if text else None), [text]


# ----------------------------------------------------------------------------------------------------
# multilingual-NLI-26lang zh (MoritzLaurer/...-2mil7): machine translations (M2M100) of MNLI / WANLI / FEVER-NLI.
# Card states no licence; we take only subsets whose English source is permissive:
#   zh_mnli (MultiNLI: OANC + CC BY(-SA) 3.0 + PD), zh_wanli (WANLI: CC BY 4.0), zh_fever (FEVER-NLI: CC BY-SA 3.0).
#   zh_anli (ANLI: CC BY-NC 4.0) and zh_ling (LingNLI: licence not verified) are EXCLUDED.
# ----------------------------------------------------------------------------------------------------
NLI_OPTS_ZH = ["蕴含：前提成立时，假设一定成立", "中立：假设可能成立，也可能不成立", "矛盾：前提成立时，假设一定不成立"]
NLI_SCORE_ZH = ["一定不成立（与上文矛盾）", "无法确定", "一定成立（可由上文推出）"]  # ordered low -> high; level = 2 - nli_label
NLI26_FILES = {"mnli": "data/zh_mnli-00000-of-00001-7cdc260bd1a5eecb.parquet",
               "wanli": "data/zh_wanli-00000-of-00001-9449492634ac3b4b.parquet",
               "fever": "data/zh_fever-00000-of-00001-b6bb1bca5e3a0240.parquet"}


def _nli26(subset, exclude_en_premises=frozenset()):
    from .common import norm_key
    src = f"nli26_zh_{subset}"
    df = pd.read_parquet(dl("MoritzLaurer/multilingual-NLI-26lang-2mil7", NLI26_FILES[subset]))
    for i, r in df.iterrows():
        p, h, y = clean(r["premise"]), clean(r["hypothesis"]), int(r["label"])
        eo = clean(r["premise_original"])  # English source premise: blocks cross-lingual leakage vs mnli_en
        if not p or not h or y not in (0, 1, 2):
            continue
        if norm_key(r["premise_original"]) in exclude_en_premises:
            continue
        key = f'{subset}|{r["premise_original"]}|{r["hypothesis_original"]}'
        split = hash_split(key, test_frac=0.02, dev_frac=0.01)
        rng = rng_for(src, key)
        u = rng.random()
        if 0.55 <= u < 0.7:
            # ordinal re-framing of the same gold label (contradiction < neutral < entailment) -> zh `score` supervision
            yield make(src, split, i, "zh", "score", f"根据上文，“{h}”成立的可能性有多大？", 2 - y, NLI_SCORE_ZH, p,
                       "likelihood"), [p, h, eo]
        elif split == "test" or u < 0.7:
            if rng.random() < 0.5:
                yield make(src, split, i, "zh", "choice", f"假设：“{h}”。这个假设与上文是什么关系？", y, NLI_OPTS_ZH, p, "rel"), [p, h, eo]
            else:
                yield make(src, split, i, "zh", "choice", "前提与假设之间是什么关系？", y, NLI_OPTS_ZH, f"前提：{p}\n假设：{h}", "rel2"), [p, h, eo]
        else:
            yield make(src, split, i, "zh", "noul", f"如果上文属实，“{h}”是否一定成立？", y == 0, context=p, variant="entail_yn"), [p, h, eo]


def nli26_zh_mnli(exclude_en_premises=frozenset()):
    yield from _nli26("mnli", exclude_en_premises)


def nli26_zh_wanli():
    yield from _nli26("wanli")


def nli26_zh_fever():
    yield from _nli26("fever")


# ----------------------------------------------------------------------------------------------------
# OpenAssistant oasst2 (human quality ratings)  licence: Apache-2.0
# ----------------------------------------------------------------------------------------------------
Q_LEVELS = {
    "zh": {"quality": ("这条助手回复的整体质量如何？", ["很差", "较差", "一般", "较好", "很好"]),
           "helpfulness": ("这条助手回复对用户有多大帮助？", ["毫无帮助", "帮助很小", "有一定帮助", "比较有帮助", "非常有帮助"]),
           "creativity": ("这条助手回复有多大创意？", ["毫无创意", "创意很少", "有一些创意", "比较有创意", "非常有创意"]),
           "prompt_quality": ("这条用户提问写得怎么样？", ["很差", "较差", "一般", "较好", "很好"])},
    "en": {"quality": ("How good is the assistant's reply overall?", ["very poor", "poor", "average", "good", "excellent"]),
           "helpfulness": ("How helpful is the assistant's reply to the user?",
                           ["not helpful at all", "slightly helpful", "somewhat helpful", "quite helpful", "extremely helpful"]),
           "creativity": ("How creative is the assistant's reply?",
                          ["not creative", "slightly creative", "somewhat creative", "quite creative",
                           "very creative"]),
           "prompt_quality": ("How well written is the user's request?",
                              ["very poor", "poor", "average", "good", "excellent"])},
}
OASST_EXTRA = ("quality", "helpfulness", "creativity")  # rated reply attributes (train_sample.oasst2_zh_multi_attr)
OASST_MAX_SHARE = 0.6  # an extra attribute (or prompt quality) whose most common level has a larger share is left out


def _level(lab, attr):
    """Level 0-4 of a rated attribute (oasst2 labels are 0-1 means of >= 2 ratings), None when not rated enough."""
    v = lab.get(attr)
    return None if v is None or v[1] < 2 else min(4, int(float(v[0]) * 5))


def _skewed(levels) -> bool:
    return not levels or max(levels.count(k) for k in set(levels)) / len(levels) > OASST_MAX_SHARE


def _oasst(lang, cap_train=None, cap_test=None):
    src = f"oasst2_{lang}"
    multi, prompts = train_n(f"{src}_multi_attr", False), train_n(f"{src}_prompts", False)
    for fname, split in (("data/train-00000-of-00001-88ba0162028a73fc.parquet", "train"),
                         ("data/validation-00000-of-00001-1deeef95c3248fe0.parquet", "test")):
        df = pd.read_parquet(dl("OpenAssistant/oasst2", fname))
        by_id = dict(zip(df["message_id"], df["text"]))
        extra = split == "train" and (multi or prompts)
        if extra:  # every record of one message tree (its root prompt) is one group: calibration takes all or none
            parent = {m: p for m, p in zip(df["message_id"], df["parent_id"]) if isinstance(p, str)}

            def group(mid):
                for _ in range(len(parent) + 1):  # up to the root (a missing parent counts as the root)
                    if parent.get(mid) not in by_id:
                        break
                    mid = parent[mid]
                return f"{src}/tree/{mid}"
        ok = (df["lang"] == lang) & (~df["deleted"]) & (df["review_result"] == True)  # noqa
        sub = df[ok & (df["role"] == "assistant")]
        out, more = [], []
        for _, r in sub.iterrows():
            labels = r["labels"]
            if labels is None:
                continue
            lab = {n: (v, c) for n, v, c in zip(labels["name"], labels["value"], labels["count"])}
            prompt = by_id.get(r["parent_id"])
            if not prompt:
                continue
            rng = rng_for(src, r["message_id"])
            attr = "helpfulness" if (rng.random() < 0.3 and "helpfulness" in lab and lab["helpfulness"][1] >= 2) else "quality"
            if attr not in lab or lab[attr][1] < 2:
                continue
            level = min(4, int(float(lab[attr][0]) * 5))
            q, levels = Q_LEVELS[lang][attr]
            pt, rt = truncate(clean_keep_lines(prompt), 1200), truncate(clean_keep_lines(r["text"]), 2400)
            ctx = (f"用户：{pt}\n\n助手：{rt}" if lang == "zh" else f"User: {pt}\n\nAssistant: {rt}")
            g = group(r["message_id"]) if extra else None
            out.append((make(src, split, r["message_id"], lang, "score", q, level, levels, ctx, attr,
                             view("chat", [pt, rt], f"oasst.{attr}") if pt and rt else None, g), [rt]))
            if extra and multi:  # the reply's other rated attributes: same texts, the raters' own labels
                more += [(a, _level(lab, a), r["message_id"], ctx, view("chat", [pt, rt], f"oasst.{a}") if pt and rt
                          else None, g, rt) for a in OASST_EXTRA if a != attr and _level(lab, a) is not None]
        if extra and prompts:  # rated user prompts (first and follow-up turns), the prompt on its own
            for _, r in df[ok & (df["role"] == "prompter")].iterrows():
                lab = r["labels"]
                lv = None if lab is None else _level(dict(zip(lab["name"], zip(lab["value"], lab["count"]))), "quality")
                pt = truncate(clean_keep_lines(r["text"]), 1200)  # as in the reply records: the same eval unit
                if lv is not None and pt:
                    v = view("prompt", [pt], "oasst.prompt_quality")
                    more.append(("prompt_quality", lv, r["message_id"], pt, v, group(r["message_id"]), pt))
        if more:
            dist = {a: [x[1] for x in more if x[0] == a] for a in sorted({x[0] for x in more})}
            skip = {a for a, lv in dist.items() if _skewed(lv)}
            print(f"  {src} extra train records per attribute: " + ", ".join(
                f"{a} {len(lv)} (largest level share {max(lv.count(k) for k in set(lv)) / len(lv):.2f}"
                + (", left out" if a in skip else "") + ")" for a, lv in dist.items()), file=sys.stderr, flush=True)
            for a, lv, mid, ctx, v, g, unit in more:
                if a not in skip:
                    q, levels = Q_LEVELS[lang][a]
                    out.append((make(src, split, mid, lang, "score", q, lv, levels, ctx, a, v, g), [unit]))
        cap = cap_train if split == "train" else cap_test
        if cap and len(out) > cap:
            out = rng_for(src, split, "cap").sample(out, cap)
        yield from out


def oasst2_zh():
    yield from _oasst("zh")


# ====================================================================================================
# Eval-only zh sets (test split only; never trained on)
# ====================================================================================================
def belebele_zh():
    src = "belebele_zh"
    with open(dl("facebook/belebele", "data/zho_Hans.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            opts = [clean(r[f"mc_answer{k}"]) for k in range(1, 5)]
            if len(set(opts)) < 4:
                continue
            ctx = clean(r["flores_passage"])
            yield make(src, "test", i, "zh", "choice", clean(r["question"]), int(r["correct_answer_num"]) - 1, opts, ctx), [ctx]


def xcopa_zh():
    src = "xcopa_zh"
    for off, split in (("validation", "dev"), ("test", "test")):
        df = pd.read_parquet(dl("cambridgeltl/xcopa", f"zh/{off}-00000-of-00001.parquet"))
        for _, r in df.iterrows():
            q = "以下哪一项更可能是原因？" if r["question"] == "cause" else "以下哪一项更可能是结果？"
            p = clean(r["premise"])
            yield make(src, split, r["idx"], "zh", "choice", q, int(r["label"]), [clean(r["choice1"]), clean(r["choice2"])], p), [p]


def xwinograd_zh():
    src = "xwinograd_zh"
    df = pd.read_parquet(dl("Muennighoff/xwinograd", "zh/test-00000-of-00001.parquet"))
    for i, r in df.iterrows():
        s = clean(r["sentence"])
        o = [clean(r["option1"]), clean(r["option2"])]
        if o[0] == o[1]:
            continue
        yield make(src, "test", i, "zh", "choice", "句子中的“_”指的是谁（或什么）？", int(r["answer"]) - 1, o, s), [s]


def globalmmlu_zh(n=1000):
    src = "globalmmlu_zh"
    df = pd.read_parquet(dl("CohereLabs/Global-MMLU", "zh/test-00000-of-00001.parquet"))
    idx = sorted(rng_for(src, "sample").sample(range(len(df)), n))
    for i in idx:
        r = df.iloc[i]
        opts = [clean(r[f"option_{k}"]) for k in "abcd"]
        if len(set(opts)) < 4:
            continue
        yield make(src, "test", r["sample_id"].replace("/", "_"), "zh", "choice", clean(r["question"]), "ABCD".index(r["answer"]),
                   opts), [clean(r["question"])]


MINDS_ZH = ["境外用卡", "更改地址", "应用程序故障", "ATM取款限额", "查询余额", "企业贷款", "银行卡问题", "存入现金", "自动扣款",
            "冻结账户或卡片", "大额支付", "联名账户", "最近的交易记录", "缴纳账单"]


def minds14_zh():
    src = "minds14_zh"
    df = pd.read_parquet(dl("PolyAI/minds14", "zh-CN/train-00000-of-00001.parquet"), columns=["path", "transcription", "intent_class"])
    for i, r in df.iterrows():
        t = clean(r["transcription"])
        if not t:
            continue
        rng = rng_for(src, i)
        opts = list(MINDS_ZH)
        rng.shuffle(opts)
        yield make(src, "test", i, "zh", "choice", "这位银行客户来电的意图是什么？", opts.index(MINDS_ZH[int(r["intent_class"])]), opts, t), [t]
