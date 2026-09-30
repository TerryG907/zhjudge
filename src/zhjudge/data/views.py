"""Structured views: the same training example written the way Laya callers write questions.

Converters attach an optional `view` to a record: the texts the example is made of and the task its question asks,
e.g. {"kind": "nli", "text": [premise, hypothesis], "task": "nli.rel"} (plus "arg" when the question carries a value,
such as the candidate intent of a yes/no question). The plain record keeps its own flat `context` / `question`.

At training time (`train.format_aug` > 0) a share of the training records is re-rendered from its view as a
Laya-native question: a JSON state with named fields ({"前提": ..., "假设": ...}) and an instruction that names the
fields in backticks ("已知`前提`，`假设`属于哪种情况？"). This is how the Laya API is used (a dict state and
field-aware instructions, e.g. laya.presets: "What domain does `request` belong to?") and how the baseline probe sets
are written, while the plain records only ever had a flat text state. A share of the rendered records uses the other
language for the instruction, the field names and, where a translation table exists, the options (Chinese text +
English question and the reverse).

Only the rendering changes: the gold option is the same option, so labels come from the datasets as before. Field
names come from 2-4 interchangeable sets per kind and every task has at least two instruction phrasings per language,
so no single template is memorised.

Format variants (`train.format_variants` > 0, v1b; a share of the rendered records): the other question shapes Laya
callers write (laya.presets): noul questions with true/false criteria and custom labels (是/否, yes/no), 5-level scores
merged exactly into 3 levels with their own level texts (levels 0-1 / 2 / 3-4), NLI options as dict criteria (bare
label -> description) and BoolQ as a 2-option choice. Criteria are built only in `zhjudge.records.to_laya` from the
record's current option order (`option_desc` is keyed by option text), so re-ordering and translation cannot misalign
them; the gold is re-derived from the dataset label and the record id never changes.

Relation to the probe sets: no probe item enters training (their texts are removed by decontamination as before), and
no instruction this module can produce is word for word a probe instruction, even with other field names -
`zhjudge validate` fails if one is (`probe_instruction_collisions`), and it fails if a description or level text a
config adds (variants, the v1b data extras) equals a probe criterion text (`probe_option_collisions`). Generic field
names (前提/假设, premise/hypothesis, sentence1/sentence2, ...) and label names (蕴含/中立/矛盾, entailment/neutral/
contradiction; v1b also 正面/负面/中性, positive/negative, 是/否, yes/no) do also occur in the probes.
"""
from __future__ import annotations

import json
import random
import re
import string
from functools import lru_cache
from pathlib import Path

VIEWS_VERSION = 4  # bump when kinds, templates or tables change (part of the tokenization cache key)

# kind -> number of texts and interchangeable field-name sets per language (one set is picked per record)
KINDS = {
    "pair": {"n": 2,
             "zh": [("句子A", "句子B"), ("句子1", "句子2"), ("第一句", "第二句"), ("文本A", "文本B")],
             "en": [("sentence_a", "sentence_b"), ("sentence1", "sentence2"), ("text_a", "text_b"),
                    ("first", "second")]},
    "nli": {"n": 2,
            "zh": [("前提", "假设"), ("上文", "假设"), ("前提", "陈述")],
            "en": [("premise", "hypothesis"), ("text", "hypothesis"), ("premise", "claim")]},
    "utterance": {"n": 1,
                  "zh": [("用户",), ("用户消息",), ("请求",), ("用户输入",)],
                  "en": [("utterance",), ("user_message",), ("request",), ("query",)]},
    "turns": {"n": 2,
              "zh": [("系统", "用户"), ("上一轮系统回复", "用户最新发言"), ("客服", "用户")],
              "en": [("system", "user"), ("previous_system_turn", "user_turn"), ("agent", "user")]},
    "comment": {"n": 1,
                "zh": [("评论",), ("言论",), ("帖子",), ("文本",)],
                "en": [("comment",), ("post",), ("text",), ("message",)]},
    "paper": {"n": 2,
              "zh": [("标题", "摘要"), ("论文标题", "论文摘要")],
              "en": [("title", "abstract"), ("paper_title", "paper_abstract")]},
    "text": {"n": 1,
             "zh": [("文本",), ("段落",), ("内容",)],
             "en": [("text",), ("passage",), ("content",)]},
    "chat": {"n": 2,
             "zh": [("用户", "助手"), ("问题", "回答"), ("提问", "回复")],
             "en": [("prompt", "response"), ("user", "assistant"), ("question", "answer")]},
    "message": {"n": 1,
                "zh": [("客户消息",), ("消息",), ("客户留言",)],
                "en": [("message",), ("customer_message",), ("query",)]},
    "passage_q": {"n": 2,
                  "zh": [("材料", "问题"), ("文章", "问题"), ("段落", "提问")],
                  "en": [("passage", "question"), ("context", "question"), ("document", "query")]},
    "entry": {"n": 2,
              "zh": [("标题", "内容"), ("词条", "正文")],
              "en": [("title", "content"), ("entry", "description")]},
    "question": {"n": 1,
                 "zh": [("问题",), ("题目",)],
                 "en": [("question",), ("prompt",)]},
    "prompt": {"n": 1,
               "zh": [("用户提问",), ("用户请求",), ("提问",)],
               "en": [("prompt",), ("user_request",), ("request",)]},
}

# task -> question type, instruction templates per language ({0}, {1}: field names; {last}: the last field;
# {arg}: the record's arg), the label table its options / arg belong to, and an optional bare-label table
TASKS = {
    "paraphrase": {"type": "noul",
                   "zh": ["{0}和{1}说的是同一个意思吗？", "{1}是否是对{0}的同义改写？", "{0}与{1}的含义是否一致？"],
                   "en": ["Are {0} and {1} saying the same thing?", "Is {1} a paraphrase of {0}?",
                          "Do {0} and {1} express the same meaning?"]},
    "nli.rel": {"type": "choice", "labels": "nli", "bare": "nli_bare",
                "zh": ["{1}相对于{0}属于哪种推理关系？", "判断{1}与{0}之间的逻辑关系。", "已知{0}，{1}属于哪种情况？"],
                "en": ["What kind of relation holds between {0} and {1}?", "How does {1} relate to {0}?",
                       "Given {0}, which describes {1}?"]},
    "nli.entail_yn": {"type": "noul",
                      "zh": ["如果{0}属实，{1}是否一定成立？", "由{0}能否推出{1}？"],
                      "en": ["If {0} is true, must {1} also be true?", "Does {0} imply {1}?"]},
    "massive.intent": {"type": "choice", "labels": "massive_intent",
                       "zh": ["{0}的意图是什么？", "{0}是想让助手做什么？"],
                       "en": ["What is the intent of {0}?", "What does the user want the assistant to do in {0}?"]},
    "massive.scenario": {"type": "choice", "labels": "massive_scenario",
                         "zh": ["{0}属于哪个场景？", "{0}涉及哪个领域？"],
                         "en": ["Which domain does {0} belong to?", "Which area is {0} about?"]},
    "massive.intent_yn": {"type": "noul", "labels": "massive_intent",
                          "zh": ["{0}的意图是“{arg}”吗？", "{0}表达的意图是“{arg}”吗？"],
                          "en": ["Is the intent of {0} \"{arg}\"?", "Does {0} have the intent \"{arg}\"?"]},
    "crosswoz.domain": {"type": "choice", "labels": "crosswoz_domain",
                        "zh": ["{last}涉及哪个领域？", "{last}谈论的是哪个领域？"],
                        "en": ["Which domain does {last} concern?", "What domain is {last} about?"]},
    "crosswoz.slot": {"type": "choice",
                      "zh": ["{last}在询问哪一项信息？", "{last}想了解哪一项信息？"],
                      "en": ["Which piece of information is {last} asking for?",
                             "What information is requested in {last}?"]},
    "crosswoz.is_request": {"type": "noul",
                            "zh": ["{last}是否在询问某项信息？", "{last}中有没有提出信息查询？"],
                            "en": ["Is {last} asking for some information?",
                                   "Is any information requested in {last}?"]},
    "cold.offensive": {"type": "noul",
                       "zh": ["{0}是否具有冒犯性？", "{0}含有攻击性或歧视性内容吗？", "{0}是否冒犯他人？"],
                       "en": ["Is {0} offensive?", "Does {0} contain insulting or discriminatory content?",
                              "Would {0} offend people?"]},
    "cold.topic": {"type": "choice", "labels": "cold_topic",
                   "zh": ["{0}讨论的是哪类话题？", "{0}涉及哪一类敏感话题？"],
                   "en": ["Which topic does {0} discuss?", "Which sensitive topic is {0} about?"]},
    "csl.category": {"type": "choice",
                     "zh": ["根据{0}和{1}，这篇论文属于哪个学科门类？", "{0}和{1}所描述的论文属于哪个门类？"],
                     "en": ["Based on {0} and {1}, which broad field does the paper belong to?",
                            "Which broad field of study is the paper described by {0} and {1} in?"]},
    "csl.discipline": {"type": "choice",
                       "zh": ["根据{0}和{1}，这篇论文最可能属于哪个一级学科？", "{0}和{1}所描述的研究属于哪个一级学科？"],
                       "en": ["Based on {0} and {1}, which discipline does the paper most likely belong to?",
                              "Which discipline is the research described by {0} and {1} in?"]},
    "csl.keyword": {"type": "noul",
                    "zh": ["“{arg}”是否是这篇论文（见{0}和{1}）作者标注的关键词之一？",
                           "{0}和{1}所描述论文的作者关键词里有“{arg}”吗？"],
                    "en": ["Is \"{arg}\" one of the author keywords of the paper described by {0} and {1}?",
                           "Did the authors of the paper in {0} and {1} list \"{arg}\" as a keyword?"]},
    "sib.topic": {"type": "choice", "labels": "sib",
                  "zh": ["{0}的主题是什么？", "{0}主要谈论什么话题？"],
                  "en": ["Which topic does {0} cover?", "What is {0} mainly about?"]},
    "oasst.quality": {"type": "score", "labels": "oasst_quality",
                      "zh": ["{1}的整体质量如何？", "作为对{0}的回答，{1}质量怎么样？"],
                      "en": ["How good is {1} overall?", "As a reply to {0}, how good is {1}?"]},
    "oasst.helpfulness": {"type": "score", "labels": "oasst_helpfulness",
                          "zh": ["{1}对{0}有多大帮助？", "作为对{0}的回复，{1}有多有用？"],
                          "en": ["How helpful is {1} as a reply to {0}?", "How useful is {1} for answering {0}?"]},
    "oasst.creativity": {"type": "score", "labels": "oasst_creativity",
                         "zh": ["{1}有多大创意？", "作为对{0}的回复，{1}有多新颖、有创造性？"],
                         "en": ["How creative is {1}?", "As a reply to {0}, how original is {1}?"]},
    "oasst.prompt_quality": {"type": "score", "labels": "oasst_prompt_quality",
                             "zh": ["作为给助手的请求，{0}的质量如何？", "{0}的整体质量怎么样？"],
                             "en": ["How good is {0} as a request to an assistant?",
                                    "What is the overall quality of {0}?"]},
    "banking.intent": {"type": "choice",
                       "zh": ["{0}最符合哪一种银行业务意图？", "客户在{0}里想办理什么业务？"],
                       "en": ["Which banking intent best describes {0}?", "What does the customer want in {0}?"]},
    "boolq": {"type": "noul",
              "zh": ["根据{0}，{1}的答案是“是”吗？", "依据{0}，对{1}的回答是肯定的吗？"],
              "en": ["Based on {0}, is the answer to {1} yes?", "According to {0}, is the answer to {1} \"yes\"?"]},
    "boolq.choice": {"type": "choice", "labels": "yes_no",  # format variant of a `boolq` record, no converter view
                     "zh": ["根据{0}，{1}的答案是什么？", "依据{0}回答{1}。"],
                     "en": ["Based on {0}, what is the answer to {1}?", "Answer {1} using {0}."]},
    "dbpedia": {"type": "choice", "labels": "dbpedia",
                "zh": ["{0}和{1}描述的是哪一类实体？", "这个百科条目（{0}）属于哪一类？"],
                "en": ["What kind of entity do {0} and {1} describe?", "Which type of entity is the entry {0} about?"]},
    "mc.answer": {"type": "choice",
                  "zh": ["哪个选项正确回答了{0}？", "请为{0}选择正确答案。"],
                  "en": ["Which option correctly answers {0}?", "Choose the best answer to {0}."]},
    "goemo.emotion": {"type": "choice", "labels": "goemo",
                      "zh": ["{0}最明显地表达了哪种情绪？", "{0}中最强烈的情绪是什么？"],
                      "en": ["Which emotion does {0} express most clearly?", "What emotion is strongest in {0}?"]},
    "goemo.emo_yn": {"type": "noul", "labels": "goemo",
                     "zh": ["{0}是否表达了“{arg}”这种情绪？", "{0}流露出“{arg}”的情绪吗？"],
                     "en": ["Does {0} express the emotion \"{arg}\"?", "Is \"{arg}\" one of the emotions in {0}?"]},
    "goemo.polarity": {"type": "choice", "labels": "polarity",
                       "zh": ["{0}整体上带有哪种情绪色彩？", "{0}的总体语气偏向哪一边？"],
                       "en": ["Which overall tone does {0} have?",
                              "Is the overall tone of {0} positive, negative or neutral?"]},
    "goemo.valence": {"type": "score", "labels": "valence",
                      "zh": ["{0}整体上有多积极？", "{0}流露的情绪有多正向？"],
                      "en": ["How positive is the overall tone of {0}?", "How upbeat is {0} overall?"]},
    "civil.tox": {"type": "score", "labels": "civil",
                  "zh": ["多大比例的读者会认为{0}带有恶意或冒犯？", "认为{0}有毒（粗鲁、冒犯或恶意）的读者占多少？"],
                  "en": ["What share of readers would rate {0} as toxic?",
                         "What fraction of readers would find {0} rude, offensive or hostile?"]},
    "civil.tox_yn": {"type": "noul",
                     "zh": ["大多数读者会认为{0}带有恶意或冒犯吗？", "多数读者会觉得{0}有毒吗？"],
                     "en": ["Would most readers consider {0} toxic?", "Would a majority of readers find {0} toxic?"]},
    "hs.helpfulness": {"type": "score", "labels": "hs_helpfulness",
                       "zh": ["{1}对{0}有多大帮助？", "{1}在多大程度上满足了{0}的需要？"],
                       "en": ["How helpful is {1} for {0}?", "How well does {1} meet the needs of {0}?"]},
    "hs.correctness": {"type": "score", "labels": "hs_correctness",
                       "zh": ["{1}在事实上有多准确、完整？", "{1}中的信息有多正确、全面？"],
                       "en": ["How factually correct and complete is {1}?",
                              "How accurate and complete is the information in {1}?"]},
    "hs.coherence": {"type": "score", "labels": "hs_coherence",
                     "zh": ["{1}表达得有多清楚、连贯？", "{1}读起来有多通顺、有条理？"],
                     "en": ["How clear and coherent is {1}?", "How easy to follow is {1}?"]},
    "hs.complexity": {"type": "score", "labels": "hs_complexity",
                      "zh": ["写出{1}需要多高的专业水平？", "{1}的用词和内容有多专业、复杂？"],
                      "en": ["How much expertise is needed to write {1}?",
                             "How sophisticated is the language and content of {1}?"]},
    "hs.verbosity": {"type": "score", "labels": "hs_verbosity",
                     "zh": ["相对于{0}的要求，{1}有多啰嗦？", "就{0}而言，{1}写得有多长、多详细？"],
                     "en": ["How verbose is {1} relative to what {0} asks?",
                            "Given {0}, how long and detailed is {1}?"]},
}

# label tables written only for views (the converters' own label maps are imported lazily in _tables())
CROSSWOZ_DOMAIN_EN = {"景点": "attraction", "餐馆": "restaurant", "酒店": "hotel", "地铁": "metro", "出租车": "taxi"}
COLD_TOPIC_EN = {"种族": "race", "性别": "gender", "地域": "region"}
CIVIL_LEVELS_ZH = ["几乎没有读者（<10%）", "少数读者（10–30%）", "部分读者（30–50%）", "多数读者（50–80%）", "几乎所有读者（≥80%）"]
HS_LEVELS_ZH = {"helpfulness": ["没有帮助", "略有帮助", "一般有帮助", "很有帮助", "极有帮助"],
                "correctness": ["大部分错误", "有若干错误", "部分正确", "基本正确", "完全正确"],
                "coherence": ["混乱不通", "难以理解", "比较清楚", "清楚", "非常清楚"],
                "complexity": ["基础", "简单", "中等", "较高", "专家级"],
                "verbosity": ["非常简短", "简洁", "适中", "详细", "非常冗长"]}
GOEMO_ZH = {"admiration": "钦佩", "amusement": "被逗乐", "anger": "愤怒", "annoyance": "恼火", "approval": "认可",
            "caring": "关心", "confusion": "困惑", "curiosity": "好奇", "desire": "渴望", "disappointment": "失望",
            "disapproval": "反对", "disgust": "厌恶", "embarrassment": "尴尬", "excitement": "兴奋", "fear": "恐惧",
            "gratitude": "感激", "grief": "悲痛", "joy": "喜悦", "love": "爱", "nervousness": "紧张", "optimism": "乐观",
            "pride": "自豪", "realization": "恍然大悟", "relief": "宽慰", "remorse": "懊悔", "sadness": "悲伤",
            "surprise": "惊讶", "neutral": "中性"}
DBPEDIA_ZH = {"company": "公司", "educational institution": "教育机构", "artist": "艺术家", "athlete": "运动员",
              "office holder / politician": "官员或政治人物", "means of transportation": "交通工具", "building": "建筑",
              "natural place": "自然地点", "village": "村庄", "animal": "动物", "plant": "植物", "music album": "音乐专辑",
              "film": "电影", "written work": "书面作品"}
NLI_BARE = [("entailment", "蕴含"), ("neutral", "中立"), ("contradiction", "矛盾")]
POLARITY_ZH = ["正面", "负面", "中性"]  # en.POLARITY_EN
VALENCE_ZH = ["明显负面", "中性", "明显正面"]  # en.VALENCE_EN
NEW_TABLES = {"polarity", "valence", "yes_no", "oasst_creativity", "oasst_prompt_quality"}  # added after v1


@lru_cache(maxsize=1)
def _tables():
    """name -> list of aligned (en, zh) label pairs. Built lazily: the converters import this module."""
    from . import en as E
    from . import zh as Z

    t = {
        "nli": list(zip(E.NLI_OPTS_EN, Z.NLI_OPTS_ZH)),
        "nli_bare": list(NLI_BARE),
        "massive_intent": [(Z.MASSIVE_EN[k], Z.MASSIVE_ZH[k]) for k in sorted(Z.MASSIVE_ZH)],
        "massive_scenario": [(Z.SCEN_EN[k], Z.SCEN_ZH[k]) for k in sorted(Z.SCEN_ZH)],
        "crosswoz_domain": [(e, z) for z, e in CROSSWOZ_DOMAIN_EN.items()],
        "cold_topic": [(e, z) for z, e in COLD_TOPIC_EN.items()],
        "sib": [(k, v) for k, v in Z.SIB_ZH.items()],
        "oasst_quality": list(zip(Z.Q_LEVELS["en"]["quality"][1], Z.Q_LEVELS["zh"]["quality"][1])),
        "oasst_helpfulness": list(zip(Z.Q_LEVELS["en"]["helpfulness"][1], Z.Q_LEVELS["zh"]["helpfulness"][1])),
        "civil": list(zip(E.CIVIL_LEVELS, CIVIL_LEVELS_ZH)),
        "goemo": list(GOEMO_ZH.items()),
        "dbpedia": list(DBPEDIA_ZH.items()),
        "polarity": list(zip(E.POLARITY_EN, POLARITY_ZH)),
        "valence": list(zip(E.VALENCE_EN, VALENCE_ZH)),
        "yes_no": [("yes", "是"), ("no", "否")],
    }
    for attr in ("creativity", "prompt_quality"):
        t["oasst_" + attr] = list(zip(Z.Q_LEVELS["en"][attr][1], Z.Q_LEVELS["zh"][attr][1]))
    for attr, zh_levels in HS_LEVELS_ZH.items():
        t["hs_" + attr] = list(zip(E.HS_Q[attr][1], zh_levels))
    return t


# ---- format variants (train.format_variants): tables written for them, nothing copied from the probe sets ----
# noul task -> language -> (description of true, description of false), each the exact meaning of the dataset label
NOUL_CRIT = {
    "paraphrase": {"zh": ("两段文字意思相同", "两段文字意思不同"), "en": ("both say the same thing", "they differ in meaning")},
    "nli.entail_yn": {"zh": ("由前者可以推出后者", "由前者推不出后者"),
                      "en": ("the second follows from the first", "the second does not follow from the first")},
    "massive.intent_yn": {"zh": ("就是这个意图", "是别的意图"), "en": ("that is the intent", "the intent is something else")},
    "crosswoz.is_request": {"zh": ("在询问某项信息", "没有询问信息"),
                            "en": ("asks for some information", "asks for no information")},
    "cold.offensive": {"zh": ("含有冒犯、攻击或歧视的内容", "不含冒犯内容"), "en": ("offensive or discriminatory", "not offensive")},
    "csl.keyword": {"zh": ("是作者标注的关键词", "不是作者标注的关键词"),
                    "en": ("the authors listed it", "the authors did not list it")},
    "boolq": {"zh": ("答案是“是”", "答案是“否”"), "en": ("the answer is yes", "the answer is no")},
    "goemo.emo_yn": {"zh": ("表达了这种情绪", "没有表达这种情绪"),
                     "en": ("the emotion is expressed", "the emotion is not expressed")},
    "civil.tox_yn": {"zh": ("至少一半的读者认为有毒", "不到一半的读者认为有毒"),
                     "en": ("at least half of readers find it toxic", "fewer than half of readers find it toxic")},
}
NOUL_LABELS = {"zh": ("否", "是"), "en": ("no", "yes")}  # (false, true): Laya's custom noul labels
# NLI dict criteria (entailment, neutral, contradiction): the descriptions of the `nli` table, or these
NLI_DESC = {"zh": ["前提保证假设成立", "前提不能确定假设是否成立", "前提排除了假设"],
            "en": ["the premise guarantees the hypothesis", "the premise leaves the hypothesis open",
                   "the premise rules the hypothesis out"]}
MERGE3 = (0, 0, 1, 2, 2)  # 5 ordered levels -> 3: levels 0-1, 2, 3-4
LEVELS3 = {  # score task -> language -> the 3 merged levels, each covering exactly its original levels
    "civil.tox": {"zh": ["不到30%的读者", "30–50%的读者", "50%及以上的读者"],
                  "en": ["under 30% of readers", "30-50% of readers", "50% of readers or more"]},
    "hs.helpfulness": {"zh": ["没有或略有帮助", "一般有帮助", "很有帮助或极有帮助"],
                       "en": ["not or slightly helpful", "moderately helpful", "very or extremely helpful"]},
    "hs.correctness": {"zh": ["大部分错误或有若干错误", "部分正确", "基本或完全正确"],
                       "en": ["mostly wrong or with several errors", "partly correct", "mostly or fully correct"]},
    "hs.coherence": {"zh": ["混乱或难以理解", "比较清楚", "清楚或非常清楚"],
                     "en": ["incoherent or hard to follow", "somewhat clear", "clear or perfectly clear"]},
    "hs.complexity": {"zh": ["基础或简单", "中等", "较高或专家级"], "en": ["basic or simple", "intermediate", "advanced or expert"]},
    "hs.verbosity": {"zh": ["非常简短或简洁", "适中", "详细或非常冗长"],
                     "en": ["very terse or succinct", "moderate", "detailed or very verbose"]},
    "oasst.quality": {"zh": ["很差或较差", "一般", "较好或很好"], "en": ["very poor or poor", "average", "good or excellent"]},
    "oasst.helpfulness": {"zh": ["毫无帮助或帮助很小", "有一定帮助", "比较有帮助或非常有帮助"],
                          "en": ["not or only slightly helpful", "somewhat helpful", "quite or extremely helpful"]},
    "oasst.creativity": {"zh": ["毫无创意或创意很少", "有一些创意", "比较有创意或非常有创意"],
                         "en": ["not or slightly creative", "somewhat creative", "quite or very creative"]},
    "oasst.prompt_quality": {"zh": ["很差或较差", "一般", "较好或很好"],
                             "en": ["very poor or poor", "average", "good or excellent"]},
}


def translate(table: str | None, s: str, lang: str):
    """`s` in `lang` via label table `table`; `s` itself when it is already in `lang`; None when unknown."""
    if not table:
        return None
    for en, zh in _tables()[table]:
        if s in (en, zh):
            return en if lang == "en" else zh
    return None


def view(kind: str, texts, task: str, arg: str | None = None) -> dict:
    """The `view` a converter attaches to a record."""
    v = {"kind": kind, "text": list(texts), "task": task}
    if arg is not None:
        v["arg"] = arg
    return v


def check(v, qtype: str):
    """Raise AssertionError unless `v` is a well-formed view for a question of type `qtype`."""
    assert isinstance(v, dict) and set(v) <= {"kind", "text", "task", "arg"}, v
    assert v.get("kind") in KINDS and v.get("task") in TASKS, v
    assert TASKS[v["task"]]["type"] == qtype, (v["task"], qtype)
    texts = v.get("text")
    assert isinstance(texts, list) and len(texts) == KINDS[v["kind"]]["n"], v
    assert all(isinstance(x, str) and x.strip() for x in texts), v
    needs_arg = any("{arg}" in s for lang in ("zh", "en") for s in TASKS[v["task"]][lang])
    assert needs_arg == ("arg" in v) and (not needs_arg or (isinstance(v["arg"], str) and v["arg"].strip())), v
    assert _fields_used(v["task"]) <= len(texts), (v["task"], v["kind"])


@lru_cache(maxsize=None)
def _fields_used(task: str) -> int:
    """How many positional fields ({0}, {1}) the task's templates refer to."""
    idx = [int(f) for lang in ("zh", "en") for t in TASKS[task][lang]
           for _, f, _, _ in string.Formatter().parse(t) if f and f.isdigit()]
    return max(idx, default=-1) + 1


_FIELD = re.compile(r"`[^`]*`")


def instruction_patterns(tasks=None):
    """Every instruction render() can produce (for `tasks`, default all), up to the field names, as regexes:
    backticked field references are masked to `*` ({0}, {1}, {last}) and {arg} matches any text. Used to keep probe
    instructions out of training."""
    pats = set()
    for task in (TASKS[k] for k in (TASKS if tasks is None else tasks)):
        for t in task["zh"] + task["en"]:
            s = t.format("`*`", "`*`", last="`*`", arg="\0")
            pats.add(re.escape(s).replace(re.escape("\0"), ".+"))
    return [re.compile(p) for p in sorted(pats)]


def _instructions(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "instructions" and isinstance(v, str):
                yield v
            else:
                yield from _instructions(v)
    elif isinstance(x, list):
        for v in x:
            yield from _instructions(v)


def probe_instruction_collisions(probe_dir, tasks=None) -> list[str]:
    """Probe instructions (data/probes/*.jsonl) that a training rendering (of `tasks`, default all; see tasks_for)
    could reproduce word for word, whatever the field names."""
    probe = set()
    for p in sorted(Path(probe_dir).glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                probe.update(_instructions(json.loads(line)))
    pats = instruction_patterns(tasks)
    return sorted(s for s in probe if any(p.fullmatch(_FIELD.sub("`*`", s)) for p in pats))


# what only a config that asks for it can produce: format variants, and the v1b data extras (data.train_sample)
OPTIONAL_TASKS = {"format_variants": ("boolq.choice",), "goemotions_en_valence": ("goemo.polarity", "goemo.valence"),
                  "oasst2_zh_multi_attr": ("oasst.creativity",), "oasst2_zh_prompts": ("oasst.prompt_quality",)}


def _asked(cfg) -> set:
    on = {k for k, v in (cfg["data"].get("train_sample") or {}).items() if v}
    return on | ({"format_variants"} if float(cfg["train"].get("format_variants") or 0) else set())


def tasks_for(cfg) -> list[str]:
    """The tasks whose instruction templates training under `cfg` can produce (v1's tasks plus what it turns on)."""
    off = {t for k, ts in OPTIONAL_TASKS.items() if k not in _asked(cfg) for t in ts}
    return [t for t in TASKS if t not in off]


def option_texts(cfg) -> tuple[set, set]:
    """(descriptions and level texts, label names) that training renderings under `cfg` add to v1's option texts."""
    tab, on = _tables(), _asked(cfg)
    texts, labels = set(), set()
    if "format_variants" in on:
        texts |= {d for t in NOUL_CRIT.values() for pair in t.values() for d in pair}
        texts |= {e.split(": ", 1)[1] for e, _ in tab["nli"]} | {z.split("：", 1)[1] for _, z in tab["nli"]}
        texts |= {d for ds in NLI_DESC.values() for d in ds}
        texts |= {x for t in LEVELS3.values() for ls in t.values() for x in ls}
        labels |= {x for p in NOUL_LABELS.values() for x in p} | {x for p in tab["yes_no"] + tab["nli_bare"] for x in p}
    for key, name in (("goemotions_en_valence", "valence"), ("oasst2_zh_multi_attr", "oasst_creativity"),
                      ("oasst2_zh_prompts", "oasst_prompt_quality")):
        if key in on:
            texts |= {x for p in tab[name] for x in p}
    if "goemotions_en_valence" in on:
        labels |= {x for p in tab["polarity"] for x in p}
    v1 = {x for name, pairs in tab.items() if name not in NEW_TABLES for p in pairs for x in p}
    return texts - v1 - labels, labels


def _criteria_strings(q, descs, keys):
    """A probe question's criteria strings: descriptions and level texts (dict values, score levels, each rendered
    `key: description` too) and label names (dict keys, list-choice labels, custom noul labels)."""
    crit = q.get("criteria")
    if isinstance(crit, dict):
        keys.update(str(k) for k in crit)
        descs.update(x for k, v in crit.items() if isinstance(v, str) and v for x in (v, f"{k}: {v}"))
    elif isinstance(crit, list):
        (descs if q.get("type") == "score" else keys).update(c for c in crit if isinstance(c, str))
    if isinstance(q.get("labels"), dict):
        keys.update(v for v in q["labels"].values() if isinstance(v, str))


def probe_option_collisions(probe_dir, cfg) -> dict:
    """texts: descriptions / level texts that training under `cfg` adds and that equal a probe criterion text (up to
    case, spacing and punctuation), which `zhjudge validate` rejects; labels: added label names that also name a probe
    option (generic names such as positive / 是; reported only)."""
    from .common import norm_key

    descs, keys = set(), set()
    for p in sorted(Path(probe_dir).glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)  # zh/en/xnli items: q_native / q_en; typed-decisions cases: questions
                for q in [(r.get(k) or {}).get("question") for k in ("q_native", "q_en")] + list(
                        (r.get("questions") or {}).values()):
                    if isinstance(q, dict):
                        _criteria_strings(q, descs, keys)
    texts, labels = option_texts(cfg)
    dn, kn = {norm_key(s) for s in descs}, {norm_key(s) for s in keys}
    return {"texts": sorted(s for s in texts if norm_key(s) in dn | kn),
            "labels": sorted(s for s in labels if norm_key(s) in dn | kn)}


def render(rec: dict, rng: random.Random, cross_lingual: float = 0.3, bare_labels: float = 0.3, with_lang=False):
    """A copy of unified record `rec` written from its view: dict state with named fields, a field-aware instruction,
    options translated when the instruction language differs from the text and a translation exists. The gold
    option (label index) is unchanged. with_lang: (copy, the language of the instruction and field names)."""
    v = rec["view"]
    task, kind = TASKS[v["task"]], KINDS[v["kind"]]
    native = rec["lang"]
    lang = native
    if rng.random() < cross_lingual:
        lang = "en" if native == "zh" else "zh"
    names = rng.choice(kind[lang])
    refs = ["`%s`" % n for n in names]
    arg = v.get("arg")
    if arg is not None:
        arg = translate(task.get("labels"), arg, lang) or arg
    out = dict(rec)
    out["context"] = dict(zip(names, v["text"]))
    out["question"] = rng.choice(task[lang]).format(*refs, last=refs[-1], arg=arg)
    if "options" in rec:
        opts = list(rec["options"])
        table = task.get("labels")
        if task.get("bare") and rng.random() < bare_labels:
            # bare labels ("entailment" instead of "entailment: if the premise is true, ...")
            idx = [next((i for i, pair in enumerate(_tables()[table]) if o in pair), None) for o in opts]
            if None not in idx:
                bare = _tables()[task["bare"]]
                opts = [bare[i][0 if lang == "en" else 1] for i in idx]
                table = task["bare"]
        tr = [translate(table, o, lang) for o in opts]
        if None not in tr and len(set(tr)) == len(tr):
            opts = tr
        out["options"] = opts
    return (out, lang) if with_lang else out


def variant(rec: dict, lang: str, rng: random.Random) -> dict:
    """A copy of rendered record `rec` (instruction language `lang`) in another Laya question shape, when its task
    has one: noul -> true/false criteria and/or custom labels; BoolQ -> also a yes/no choice; a 5-level score ->
    3 merged levels (MERGE3); NLI -> dict criteria (bare label -> description). Descriptions go to `option_desc`,
    keyed by option text (zhjudge.records.to_laya builds the criteria in the final option order). The gold is the same
    answer, re-derived from the dataset label; the id never changes."""
    task, typ = rec["view"]["task"], rec["type"]
    kinds = (["noul"] + (["boolq.choice"] if task == "boolq" else []) if typ == "noul"
             else ["score3"] if typ == "score" and task in LEVELS3 and len(rec["options"]) == len(MERGE3)
             else ["nli_dict"] if typ == "choice" and TASKS[task].get("bare") else [])
    if not kinds:
        return rec
    k = rng.choice(kinds)
    out = dict(rec)
    li = 0 if lang == "en" else 1
    if k == "noul":
        if task in NOUL_CRIT:
            t, f = NOUL_CRIT[task][lang]
            out["option_desc"] = {"false": f, "true": t}
        if task not in NOUL_CRIT or rng.random() < 0.3:
            out["noul_labels"] = dict(zip(("false", "true"), NOUL_LABELS[lang]))
    elif k == "boolq.choice":
        assert isinstance(rec["context"], dict), rec["id"]  # rendered: {passage field: ..., question field: ...}
        refs = ["`%s`" % n for n in rec["context"]]
        yes, no = (p[li] for p in _tables()["yes_no"])
        opts = [yes, no] if rng.random() < 0.5 else [no, yes]
        out.update(type="choice", question=rng.choice(TASKS[k][lang]).format(*refs, last=refs[-1]), options=opts,
                   label=opts.index(yes if rec["label"] else no))
    elif k == "score3":
        out["options"] = list(LEVELS3[task][lang])
        out["label"] = MERGE3[rec["label"]]
    else:
        tab, bare = _tables()["nli"], _tables()["nli_bare"]
        rows = [next((i for i, p in enumerate(tab) if o in p), None) for o in rec["options"]]
        rows = [r if r is not None else next((i for i, p in enumerate(bare) if o in p), None)
                for r, o in zip(rows, rec["options"])]
        if None in rows or len(set(rows)) != len(rows):
            return rec
        descs = NLI_DESC[lang] if rng.random() < 0.5 else [p[li].split(": " if li == 0 else "：", 1)[1] for p in tab]
        out["options"] = [bare[r][li] for r in rows]
        out["option_desc"] = {bare[r][li]: descs[r] for r in rows}
    assert out["id"] == rec["id"]
    return out


def shuffle_options(rec: dict, rng: random.Random) -> dict:
    """A copy of a `choice` record with its options in a random order (the label follows its option)."""
    opts = list(rec["options"])
    gold = opts[rec["label"]]
    rng.shuffle(opts)
    out = dict(rec)
    out["options"] = opts
    out["label"] = opts.index(gold)
    return out
