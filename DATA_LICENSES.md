# 训练数据许可审计 / Training-data licence audit

审计日期 Audited: **2026-09-29**，逐一核对了 Hugging Face 数据卡和上游来源（GitHub LICENSE/README、官方页面、论文页）。
机器可读版本 Machine-readable version: [`configs/licenses.yaml`](configs/licenses.yaml).

> 这不是法律意见。拿不准的一律归为 EXCLUDE。
> This is not legal advice. Anything uncertain was classified EXCLUDE.

---

## 0. 结论 / Bottom line

> 本节的条数来自 phase-0 的构建（v1b 的 `data.train_sample` 之前，v1b 另外多用了 MNLI、GoEmotions 和 oasst2_zh 的样本）；每个发布模型实际用的条数记录在它自己的 `training_data.json` 里。 / Counts in this section are from the phase-0 build, before v1b's `data.train_sample` (v1b uses more MNLI, GoEmotions and oasst2_zh records); each published model's actual counts are in its own `training_data.json`.

**中文**

- 已转换的 27 个来源中，**24 个可以用于训练或评测**（ALLOW 或 CONDITIONAL），**3 个被排除**：`MoritzLaurer/multilingual-NLI-26lang-2mil7` 的 zh_mnli / zh_wanli / zh_fever。原因是它的中文译文没有任何许可（数据卡里没有 licence 字段，也没有许可文字）。它依据的英文上游数据是有许可的，所以我们可以**自己把英文原文重新翻译一遍**来补回这部分（第 6 节）。
- 排除之后，训练数据还剩 **中文 136,978 条**（原 208,461）和 **英文 108,134 条**（英文全部保留）。
- 如果权重要以 **Apache-2.0** 发布，并且完全不想碰"相同方式共享"（share-alike）的风险，就用 `permissive_strict` 方案训练，训练数据剩 **中文 88,137 条 / 英文 64,891 条**。
- 重点检查的数据集里，CLUE 的 TNEWS、IFLYTEK、AFQMC、CMNLI、CLUEWSC 和 CLUE 版 CSL，以及 ChnSentiCorp，都**没有许可**；OCNLI、C3、XNLI、CMMLU、C-Eval 和 Amazon reviews 是**非商用**；LCQMC 和 THUCNews 限**研究用途**或商用需另签协议。这些全部 EXCLUDE。CrossWOZ 可以用（CONDITIONAL）。PAWS-X 是 CONDITIONAL，因为它的句子来自 Wikipedia，带 CC BY-SA。
- 排除 nli26 后，中文 `score` 类型的训练数据从 14,967 条降到 **4,345 条**（只剩 oasst2_zh）。重新翻译是补回来的最快办法。

**English**

- Of the 27 converted sources, **24 may be used for training or evaluation** (ALLOW or CONDITIONAL) and **3 are excluded**: zh_mnli, zh_wanli and zh_fever from `MoritzLaurer/multilingual-NLI-26lang-2mil7`. Its Chinese translations carry no licence at all: the card has no licence field and no licence text. The English upstream data is licensed, so we can **re-translate it ourselves** to get these examples back (section 6).
- After the exclusions, training keeps **136,978 zh** examples (down from 208,461) and **108,134 en** examples (all English kept).
- If the weights must be **Apache-2.0** with no share-alike exposure, train the `permissive_strict` profile, which keeps **88,137 zh / 64,891 en** training examples.
- Among the commonly used Chinese benchmarks checked first: the CLUE tasks TNEWS, IFLYTEK, AFQMC, CMNLI, CLUEWSC and CLUE's CSL packaging, and ChnSentiCorp, have **no licence**. OCNLI, C3, XNLI, CMMLU, C-Eval and Amazon reviews are **non-commercial**. LCQMC and THUCNews are **research-only** or need a separate commercial agreement. All of these are EXCLUDE. CrossWOZ is usable (CONDITIONAL). PAWS-X is CONDITIONAL because its sentences come from Wikipedia under CC BY-SA.
- Dropping nli26 cuts zh `score` training data from 14,967 to **4,345** (only oasst2_zh is left). Re-translating is the quickest way to get it back.

---

## 1. 分级规则 / Classes

| 类别 Class | 含义 | Meaning |
|---|---|---|
| **ALLOW** | 权利人给出了明确的宽松许可（Apache-2.0、MIT、CC0、CC BY、"任何用途"）。义务只有常规的署名和附上许可声明。 | The rights holder gives an explicit permissive licence (Apache-2.0, MIT, CC0, CC BY, "any purpose"). The only duties are standard attribution and a licence notice. |
| **CONDITIONAL** | 可以用，但模型卡必须履行额外义务：相同方式共享（CC BY-SA / GFDL）、继承的上游文本许可，或必要的披露。 | Usable, but the model card must meet extra duties: share-alike (CC BY-SA / GFDL), inherited upstream text licences, or required disclosures. |
| **EXCLUDE** | 非商用、仅限研究、无许可或许可不明且上游也没有授权、条款互相矛盾，或禁止分发衍生品。绝不用于训练。 | Non-commercial, research-only, no licence or unknown licence with no upstream grant, conflicting terms, or terms forbidding redistribution of derivatives. Never trained on. |

`role` 字段 / field: `train` 用于训练 used for training · `eval_only` 仅评测 evaluation only · `probe_only` 许可不允许训练，只在本地算基准分，不分发 licence forbids training, used locally for benchmark numbers only, never redistributed · `decontam_only` 只用来从训练集删掉重叠文本 only used to remove overlapping text from train · `not_converted` 候选，还没有转换器 candidate with no converter yet.

---

## 2. 训练来源（已转换）/ Training sources (converted)

表中"义务"一栏用中文写；每项义务的英文版在 `configs/licenses.yaml` 的 `duties` 字段里。
The Duties column below is in Chinese; the English duties for each source are in the `duties` field of `configs/licenses.yaml`.

| 数据集 Dataset | HF id | 许可 Licence | 类别 Class | 证据 Evidence | 义务 Duties |
|---|---|---|---|---|---|
| PAWS-X zh (`pawsx_zh`) | `google-research-datasets/paws-x` (zh) | Google 许可："may be freely used for any purpose"；句子源自 Wikipedia (CC BY-SA 3.0) | **CONDITIONAL** (SA) | [LICENSE](https://github.com/google-research-datasets/paws/blob/master/LICENSE), [pawsx](https://github.com/google-research-datasets/paws/tree/master/pawsx) | 致谢 Google；署名 Wikipedia；SA 见第 6 节 |
| MASSIVE zh-CN (`massive_zh`) | `mteb/amazon_massive_intent` (上游 `AmazonScience/massive`) | CC BY 4.0 | **ALLOW** | [NOTICE.md](https://github.com/alexa/massive/blob/main/NOTICE.md) | 署名 "MASSIVE © Amazon.com Inc., CC BY 4.0"（基于 SLURP，CC BY 4.0）；注明修改 |
| CrossWOZ (`crosswoz_zh`) | `ConvLab/crosswoz` | Apache-2.0 | **CONDITIONAL** | [LICENSE](https://github.com/thu-coai/CrossWOZ/blob/master/LICENSE), [GEM card](https://huggingface.co/datasets/GEM/CrossWOZ) | Apache 声明 + 引用；在模型卡披露："Annotators agree using the dataset for research purpose."（GEM 卡同时写明 open license - commercial use allowed，其他下游用途：Any） |
| COLD (`cold_zh`) | `thu-coai/cold` | Apache-2.0 | **ALLOW** | [LICENSE](https://github.com/thu-coai/COLDataset/blob/main/LICENSE) | Apache 声明 + 引用 Deng et al. 2022 |
| CSL (`csl_zh`) | `neuclir/csl` | Apache-2.0 | **ALLOW** | [CSL readme §License](https://github.com/ydli-ai/CSL/blob/master/readme.md), [card](https://huggingface.co/datasets/neuclir/csl) | Apache 声明 + 引用 Li et al. 2022（只用学科/关键词标签） |
| SIB-200 zho_Hans (`sib200_zh`) | `Davlan/sib200` | CC BY-SA 4.0 (FLORES-200) | **CONDITIONAL** (SA) | [card](https://huggingface.co/datasets/Davlan/sib200), [FLORES licences](https://github.com/facebookresearch/flores/blob/main/README.md) | 署名 SIB-200 + FLORES-200；SA |
| nli26 zh_mnli | `MoritzLaurer/multilingual-NLI-26lang-2mil7` | **译文无许可** / none on translations | **EXCLUDE** | [card](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7) | 补救：自行重译 MultiNLI 英文原文，重译后为 CONDITIONAL (SA) |
| nli26 zh_wanli | 同上 same | 译文无许可；上游 WANLI 为 CC BY 4.0 | **EXCLUDE** | [card](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7), [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) | 补救：重译 WANLI，重译后为 ALLOW（需披露 WANLI 含 GPT-3 生成文本） |
| nli26 zh_fever | 同上 same | 译文无许可；上游 FEVER 为 CC BY-SA 3.0 | **EXCLUDE** | [card](https://huggingface.co/datasets/MoritzLaurer/multilingual-NLI-26lang-2mil7), [FEVER](https://fever.ai/dataset/fever.html) | 补救：重译 NLI-FEVER，重译后为 CONDITIONAL (SA) |
| OASST2 zh (`oasst2_zh`) | `OpenAssistant/oasst2` | Apache-2.0 | **ALLOW** | [card](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache 声明 |
| BANKING77 (`banking77_en`) · 英文候选 #1 | `legacy-datasets/banking77` | CC BY 4.0 | **ALLOW** | [LICENSE](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE) | 署名 PolyAI；注明修改 |
| BoolQ (`boolq_en`) · 英文候选 #2 | `google/boolq` | CC BY-SA 3.0 | **CONDITIONAL** (SA) | [README §License](https://github.com/google-research-datasets/boolean-questions#license) | 署名 Google；SA |
| MultiNLI (`mnli_en`) · 英文候选 #4 | `nyu-mll/multi_nli` | OANC（大部分）+ CC BY-SA 3.0（*Seven Swords*）+ CC BY 3.0 + 美国公有领域 | **CONDITIONAL** (SA) | [card](https://huggingface.co/datasets/nyu-mll/multi_nli) | 署名 Williams et al. + OANC；两部 CC BY 3.0 小说署名；SA（小说类） |
| DBpedia-14 (`dbpedia14_en`) · 英文候选 #8 | `fancyzhx/dbpedia_14` | CC BY-SA 3.0 + GFDL | **CONDITIONAL** (SA) | [card](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | 署名 Zhang et al. / DBpedia / Wikipedia；选用 CC BY-SA；SA |
| PAWS labeled_final (`paws_en`) | `google-research-datasets/paws` | Google "any purpose" + Wikipedia 句子 (CC BY-SA 3.0) | **CONDITIONAL** (SA) | [LICENSE](https://github.com/google-research-datasets/paws/blob/master/LICENSE) | 致谢 Google；署名 Wikipedia；SA |
| CommonsenseQA (`csqa_en`) | `tau/commonsense_qa` | MIT | **ALLOW** | [第一作者确认 first-author confirmation (issue #5)](https://github.com/jonathanherzig/commonsenseqa/issues/5) | 附 MIT 版权与许可声明 |
| ARC (`arc_en`) | `allenai/ai2_arc` | CC BY-SA 4.0 | **CONDITIONAL** (SA) | [allenai card](https://huggingface.co/datasets/allenai/ai2_arc) | 署名 AI2；SA |
| GoEmotions (`goemotions_en`) | `google-research-datasets/go_emotions` | CC BY 4.0（上游；HF 卡写 apache-2.0） | **ALLOW** | [google-research README](https://github.com/google-research/google-research/blob/master/README.md) | 署名 Google；注明修改 |
| Civil Comments (`civil_en`) | `google/civil_comments` | CC0 1.0 | **ALLOW** | [card](https://huggingface.co/datasets/google/civil_comments) | 无强制义务，礼节性引用 |
| HelpSteer2 (`helpsteer2_en`) | `nvidia/HelpSteer2` | CC BY 4.0 | **ALLOW** | [card](https://huggingface.co/datasets/nvidia/HelpSteer2) | 署名 NVIDIA；注明修改 |
| MASSIVE en-US (`massive_en`) | `mteb/amazon_massive_intent` | CC BY 4.0 | **ALLOW** | [NOTICE.md](https://github.com/alexa/massive/blob/main/NOTICE.md) | 同 massive_zh |
| OASST2 en (`oasst2_en`) | `OpenAssistant/oasst2` | Apache-2.0 | **ALLOW** | [card](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache 声明 |

**说明**

- 如果镜像的标签和上游不一致，一律按上游许可执行。例如 `mteb/amazon_massive_intent` 标着 apache-2.0，但上游是 CC BY 4.0。
- CSL、COLD、GoEmotions、Civil Comments 的原始文本来自期刊摘要或社交媒体。我们依据的是数据集发布方给出的许可。
- HelpSteer2 的回复由 NVIDIA 自研模型生成，数据卡写明："none from proprietary LLM providers such as OpenAI"。

**Notes**

- Where a mirror's tag disagrees with its upstream, the upstream licence governs. For example, `mteb/amazon_massive_intent` is tagged apache-2.0, but the upstream is CC BY 4.0.
- The source text of CSL, COLD, GoEmotions and Civil Comments comes from journal abstracts or social media. We rely on the licence granted by each dataset's publisher.
- HelpSteer2 responses come from NVIDIA's in-house models; the card says "none from proprietary LLM providers such as OpenAI".

---

## 3. 仅评测 / Eval-only

| 数据集 Dataset | HF id | 许可 Licence | 类别 Class | 证据 Evidence | 义务 Duties |
|---|---|---|---|---|---|
| Belebele zho_Hans | `facebook/belebele` | CC BY-SA 4.0 | CONDITIONAL | [README §License](https://github.com/facebookresearch/belebele#license) | 报告分数时引用 |
| XCOPA zh | `cambridgeltl/xcopa` | CC BY 4.0 | ALLOW | [card](https://huggingface.co/datasets/cambridgeltl/xcopa) | 引用 |
| XWinograd zh | `Muennighoff/xwinograd` | CC BY 4.0 | ALLOW | [card](https://huggingface.co/datasets/Muennighoff/xwinograd) | 引用 |
| Global-MMLU zh | `CohereLabs/Global-MMLU` | Apache-2.0 | ALLOW | [card](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | 引用 |
| MInDS-14 zh-CN | `PolyAI/minds14` | CC BY 4.0 | ALLOW | [card](https://huggingface.co/datasets/PolyAI/minds14) | 引用 |
| typed-decisions | `LocalLLaMA/typed-decisions` | Apache-2.0 | ALLOW，但 **禁止训练** / **never train** | [card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 它的 gold 标签来自一个未公开的 LLM 教师接口，我们无法确认其中没有 Jev API 输出，所以只用于评测；它的状态文本也已从我们的训练集里删掉。Its gold labels come from an unnamed LLM teacher endpoint; we cannot confirm they contain no Jev API output, so it stays eval-only, and its states are removed from our train split. |
| feishu_zh (Laya) | — (GitHub `NandhaKishorM/laya`) | Apache-2.0 | ALLOW | [repo](https://github.com/NandhaKishorM/laya) | 引用 Laya。只在 phase-0 spike 里用过，本仓库不重建也不评测 / used in the phase-0 spike only; not rebuilt or evaluated by this repo |

---

## 4. 排除 / Excluded

### 4a. 重点核查 / Commonly used Chinese benchmarks (checked first)

| 数据集 Dataset | HF id | 许可 Licence | 类别 Class | 证据 Evidence（引文 quote） | 可否评测 Eval use |
|---|---|---|---|---|---|
| TNEWS (CLUE) | `clue/clue` tnews | 无 / unknown | **EXCLUDE** | [card](https://huggingface.co/datasets/clue/clue) `license: unknown`；[CLUE](https://github.com/CLUEbenchmark/CLUE) 没有 LICENSE 文件（2026-09-29 复查） | 仅本地 probe |
| IFLYTEK (CLUE) | `clue/clue` iflytek | 无 | **EXCLUDE** | 同上 | 仅本地 probe |
| AFQMC (CLUE) | `clue/clue` afqmc | 无（蚂蚁 ATEC 比赛数据） | **EXCLUDE** | 同上 | 仅本地 probe |
| OCNLI | `clue/clue` ocnli | CC BY-NC 2.0 | **EXCLUDE** | [README](https://github.com/CLUEbenchmark/OCNLI#license): "Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)" | 仅本地 probe |
| CMNLI (CLUE) | `clue/clue` cmnli | 无 + 含 XNLI (CC BY-NC 4.0) | **EXCLUDE** | [XNLI LICENSE](https://github.com/facebookresearch/XNLI/blob/main/LICENSE) | 仅去污染 decontam |
| CSL (CLUE 2020 版) | `clue/clue` csl | 无 | **EXCLUDE** | [card](https://huggingface.co/datasets/clue/clue)。CLUE 自己打包的关键词任务（含生成的假关键词）没有许可；ydli-ai/CSL 的 Apache-2.0 覆盖的是 2022 年的 CSL 语料，我们用的就是那个（`csl_zh`）。CLUE's own keyword-task packaging (with generated fake keywords) has no licence; ydli-ai/CSL's Apache-2.0 covers the 2022 CSL corpus, which is what we use as `csl_zh`. | 仅去污染 decontam |
| C3 | `clue/clue` c3 | 仅限非商用研究 | **EXCLUDE** | [license.txt](https://github.com/nlpdata/c3/blob/master/license.txt): "intended for non-commercial research purpose only" | 仅本地 probe |
| CLUEWSC2020 | `clue/clue` cluewsc2020 | 无 | **EXCLUDE** | [card](https://huggingface.co/datasets/clue/clue) | 仅去污染 decontam |
| ChnSentiCorp | `lansinuote/ChnSentiCorp` | 无 | **EXCLUDE** | 镜像没有数据卡（README 404）；[ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) 没有 LICENSE | 仅本地 probe |
| LCQMC | `C-MTEB/LCQMC` | 仅限研究，需申请 | **EXCLUDE** | [HIT-SZ](http://icrc.hitsz.edu.cn/Article/show/171.html): "only for the specified applicant or study groups for research purposes. Without permission, it may not be used for any commercial purposes." | 仅本地 probe（严格说应先向 HIT-SZ 申请 / strictly, apply to HIT-SZ first） |
| XNLI | `facebook/xnli` | CC BY-NC 4.0 | **EXCLUDE** | [LICENSE](https://github.com/facebookresearch/XNLI/blob/main/LICENSE): "Attribution-NonCommercial 4.0 International" | 仅本地 probe |
| PAWS-X | `google-research-datasets/paws-x` | 见第 2 节 | CONDITIONAL（未排除 not excluded） | 见第 2 节 | 可以 |
| Amazon reviews multi (zh, en) | `SetFit/amazon_reviews_multi_{zh,en}` | Amazon 协议，仅限非商用研究；数据集已下架 | **EXCLUDE** | [defunct card](https://huggingface.co/datasets/defunct-datasets/amazon_reviews_multi): "licensed this dataset under its own agreement for non-commercial research usage only"；"defunct and no longer accessible" | 仅本地 probe（数据已下架，建议移出 probe / dataset withdrawn; consider dropping it from the probe） |
| CrossWOZ | `ConvLab/crosswoz` | Apache-2.0 | CONDITIONAL（未排除 not excluded） | 见第 2 节 | 可以 |
| THUCNews | `Tongjilibo/THUCNews` | 免费开放，商用需另签技术许可 | **EXCLUDE** | [thuctc.thunlp.org](http://thuctc.thunlp.org/): "如有机构或个人拟将THUCTC用于商业目的，请发邮件至thunlp@163.com洽谈技术许可协议。"（正文是 2005–2011 年的新浪新闻） | 不用 |
| CMMLU | `haonan-li/cmmlu` | CC BY-NC-SA 4.0 | **EXCLUDE** | [README 许可证](https://github.com/haonan-li/CMMLU) | 仅本地私有基准 |
| C-Eval | `ceval/ceval-exam` | CC BY-NC-SA 4.0（代码 MIT） | **EXCLUDE** | [README §Licenses](https://github.com/hkust-nlp/ceval#licenses) | 仅本地私有基准 |

### 4b. 英文候选来源 / English candidate sources

英文候选清单的 10 个来源 / the ten English candidate sources: #1 BANKING77 **ALLOW** · #2 BoolQ **CONDITIONAL** · #4 MultiNLI **CONDITIONAL** · #8 DBpedia-14 **CONDITIONAL**（见第 2 节 / see section 2），以及下面 6 个 EXCLUDE / plus the six EXCLUDE sources below:

| 数据集 Dataset | HF id | 许可 Licence | 类别 Class | 证据 Evidence |
|---|---|---|---|---|
| AG News (#3) | `fancyzhx/ag_news` | 学术 / 非商用 | **EXCLUDE** | [card](https://huggingface.co/datasets/fancyzhx/ag_news): "for research purposes ... and any other non-commercial activity" |
| SST-5 (#5)；GLUE SST-2 probe | `SetFit/sst5`, `nyu-mll/glue` sst2 | 无 | **EXCLUDE** | [stanfordnlp/sst](https://huggingface.co/datasets/stanfordnlp/sst): "[Needs More Information]" |
| Yelp review full (#6) | `Yelp/yelp_review_full` | Yelp Dataset Agreement（非商用） | **EXCLUDE** | [card](https://huggingface.co/datasets/Yelp/yelp_review_full) |
| TREC (#7) | `CogComp/trec` | 无 | **EXCLUDE** | [card](https://huggingface.co/datasets/CogComp/trec): "[More Information Needed]" |
| Amazon reviews multi en (#9) | `SetFit/amazon_reviews_multi_en` | 非商用；已下架 | **EXCLUDE** | 见 4a（SetFit 镜像的 apache-2.0 标签与上游矛盾 / SetFit's apache-2.0 tag contradicts upstream） |
| IMDB (#10) | `stanfordnlp/imdb` | 无 | **EXCLUDE** | [card](https://huggingface.co/datasets/stanfordnlp/imdb): "[More Information Needed]" |
| GLUE QQP (probe) | `nyu-mll/glue` qqp | Quora 条款 | **EXCLUDE** | [PAWS README](https://github.com/google-research-datasets/paws): "cannot directly distribute the raw PAWS-QQP data due to the license of QQP" |
| DREAM (probe, EN control 的 reading_mc) | — (GitHub `nlpdata/dream`，sha256 固定) | 仅限非商用研究 / non-commercial research only | **EXCLUDE** | [license.txt](https://github.com/nlpdata/dream/blob/master/license.txt): "DREAM dataset is intended for non-commercial research purpose only." 仅本地 probe / local probe only |

### 4c. 其他排除 / Other exclusions

| 数据集 Dataset | HF id | 许可 Licence | 证据 Evidence |
|---|---|---|---|
| BQ corpus | `C-MTEB/BQ` | 仅限研究 | [HIT-SZ](http://icrc.hitsz.edu.cn/info/1037/1162.htm) |
| SMP2017-ECDT | — | "开源供研究使用" | [repo](https://github.com/HITlilingzhi/SMP2017ECDT-DATA) |
| RiSAWOZ | `GEM/RiSAWOZ` | CC BY-NC 4.0（GEM 标 cc-by-4.0，与上游矛盾） | [README](https://github.com/terryqj0107/RiSAWOZ#data-license) |
| BBT-FinCUGE / FinFE / FinChinaSentiment | `Maciel/FinCUGE-Instruction`, `FinanceMTEB/*` | 无 | [repo](https://github.com/ssymmetry/BBT-FinCUGE-Applications) |
| FinEval sentiment | `FinanceMTEB/FinEvaSentiment` | 矛盾（HF 标 NC-SA，GitHub 为 Apache） | [repo](https://github.com/SUFE-AIFLM-Lab/FinEval) |
| CFBenchmark | `TongjiFinLab/CFBenchmark` | 非商用 + 含 OpenAI 生成内容 | [repo](https://github.com/TongjiFinLab/CFBenchmark) |
| FinanceIQ | `Duxiaoman-DI/FinanceIQ` | CC BY-NC-SA 4.0 | [card](https://huggingface.co/datasets/Duxiaoman-DI/FinanceIQ) |
| DuReader yes/no | `dirtycomputer/dureader_yesno-data` | 不明（Apache 只写在代码仓库上） | [repo](https://github.com/baidu/DuReader) |
| LogiQA 2.0 | `datatune/LogiQA2.0` | CC BY-NC-SA 4.0（镜像标 mit） | [README](https://github.com/csitfun/LogiQA2.0) |
| weibo_senti_100k / waimai / online_shopping | `dirtycomputer/weibo_senti_100k` | 无 | [repo](https://github.com/SophonPlus/ChineseNlpCorpus) |
| C-MTEB JD / OnlineShopping / Waimai | `C-MTEB/JDReview-classification` | 无 | [card](https://huggingface.co/datasets/C-MTEB/JDReview-classification) |
| shibing624/nli_zh, snli-zh | `shibing624/*` | 标签与数据卡矛盾（"用于学术研究"） | [nli_zh](https://huggingface.co/datasets/shibing624/nli_zh) |
| nli26 zh_anli / zh_ling | `MoritzLaurer/...` | 无 + ANLI 非商用 | [anli](https://huggingface.co/datasets/facebook/anli) |
| ANLI | `facebook/anli` | CC BY-NC 4.0 | [card](https://huggingface.co/datasets/facebook/anli) |
| Financial PhraseBank | `takala/financial_phrasebank` | CC BY-NC-SA 3.0 | [card](https://huggingface.co/datasets/takala/financial_phrasebank) |
| dair-ai/emotion | `dair-ai/emotion` | 仅限教育与研究 | [card](https://huggingface.co/datasets/dair-ai/emotion) |
| Amazon counterfactual | `mteb/amazon_counterfactual` | **CC BY-NC 4.0**（mteb 标 cc-by-4.0） | [LICENSE](https://github.com/amazon-research/amazon-multilingual-counterfactual-dataset/blob/main/LICENSE) |

**评测用途**

- 非商用或仅限研究的数据集，可以在运行时下载，在本地计算基准分数。这属于非商业研究评测。
- 这些数据绝不训练，不提交进仓库，也不重新分发。
- 如果项目转向商业用途，需要重新评估。

**Eval use**

- Non-commercial or research-only sets may be downloaded at run time to compute benchmark numbers locally. This counts as non-commercial research evaluation.
- They are never trained on, never committed to the repo and never redistributed.
- Re-assess if the project becomes commercial.

---

## 5. 未转换但许可可用 / Not converted, licence OK

| 数据集 Dataset | HF id | 许可 Licence | 类别 Class | 证据 Evidence |
|---|---|---|---|---|
| CLINC150 | `clinc/clinc_oos` | CC BY 3.0 | ALLOW | [LICENSE](https://github.com/clinc/oos-eval/blob/master/LICENSE) |
| KdConv | `thu-coai/kdconv` | Apache-2.0 | ALLOW | [repo](https://github.com/thu-coai/KdConv) |
| MMMLU zh_CN | `openai/MMMLU` | MIT | ALLOW（建议只做评测 / eval recommended） | [card](https://huggingface.co/datasets/openai/MMMLU) |
| SNLI | `stanfordnlp/snli` | CC BY-SA 4.0 | CONDITIONAL (SA) | [card](https://huggingface.co/datasets/stanfordnlp/snli) |
| XStoryCloze zh | `juletxara/xstory_cloze` | CC BY-SA 4.0 | CONDITIONAL (SA)，仅评测候选 / eval candidate only | [card](https://huggingface.co/datasets/juletxara/xstory_cloze) |

---

## 6. 发布方案、相同方式共享与 nli26 补救 / Release profiles, share-alike and the nli26 remedy

**相同方式共享（SA）**

- 共有 7 个训练来源带 SA：`pawsx_zh`、`sib200_zh`、`boolq_en`、`dbpedia14_en`、`mnli_en`、`paws_en`、`arc_en`。它们都是 CC BY-SA 3.0 或 4.0，大多源自 Wikipedia、FLORES 或考试题。
- 如果模型权重在法律上被认定为这些数据的"演绎作品"（Adapted Material），权重就必须以 CC BY-SA 发布。这一点目前**没有定论**。
- 保守做法是训练两个版本：
  - `default`：ALLOW + CONDITIONAL，用于云端实验。如果要发布，权重用 **CC BY-SA 4.0**。这个许可兼容 BY-SA 3.0 的"后续版本"条款，也允许商用。
  - `permissive_strict`：去掉所有 `share_alike: true` 的来源，权重用 **Apache-2.0** 发布。
- 在本仓库里：`configs/datasets.yaml` 只打开了 ALLOW / CONDITIONAL 的 24 个来源，build 和 validate 会按 `configs/licenses.yaml` 复核。默认是 `default` 方案；要训练 `permissive_strict`，加 `--set data.license_profile=permissive_strict --set export.license=apache-2.0`。

**Share-alike (SA)**

- Seven training sources are SA: `pawsx_zh`, `sib200_zh`, `boolq_en`, `dbpedia14_en`, `mnli_en`, `paws_en` and `arc_en`. All are CC BY-SA 3.0 or 4.0, mostly derived from Wikipedia, FLORES or exam questions.
- If trained weights are legally "Adapted Material" of these datasets, the weights would have to be released under CC BY-SA. This is **unsettled**.
- The conservative approach is to train two profiles:
  - `default`: ALLOW + CONDITIONAL, for cloud experiments. If released, publish the weights under **CC BY-SA 4.0**. That licence satisfies BY-SA 3.0's "later version" clause and still allows commercial use.
  - `permissive_strict`: drop every source with `share_alike: true` and publish the weights under **Apache-2.0**.
- In this repo: `configs/datasets.yaml` enables only the 24 ALLOW / CONDITIONAL sources, and build and validate re-check them against `configs/licenses.yaml`. The `default` profile is used unless you pass `--set data.license_profile=permissive_strict --set export.license=apache-2.0`.

**nli26 补救**

- 数据卡里保留了英文原文（`premise_original` / `hypothesis_original`），而且作者说明了用的翻译模型是 opus-mt。
- 所以可以在云端 GPU 上用 `Helsinki-NLP/opus-mt-en-zh` 直接从上游英文数据重新翻译：MultiNLI、WANLI（CC BY 4.0）、NLI-FEVER（CC BY-SA 3.0）。约 7.5 万对，在 T4 上大约 1 小时。
- 重译后，WANLI 部分为 ALLOW，MNLI 和 FEVER 部分为 CONDITIONAL（SA）。
- 另一条路是请作者补一个许可。他的数据卡上留了联系方式，他用这份数据训练的模型是 MIT 许可。

**nli26 remedy**

- The card keeps the English originals (`premise_original` / `hypothesis_original`) and says the translations were made with opus-mt.
- So we can re-translate straight from the upstream English data with `Helsinki-NLP/opus-mt-en-zh` on a cloud GPU: MultiNLI, WANLI (CC BY 4.0) and NLI-FEVER (CC BY-SA 3.0). That is about 75k pairs, roughly 1 hour on a T4.
- After re-translation, the WANLI part is ALLOW and the MNLI and FEVER parts are CONDITIONAL (SA).
- The alternative is to ask the author to add a licence. His card gives contact details, and the model he trained on this data is MIT-licensed.

---

## 7. 剩余训练量 / Remaining training examples

> 本节的条数来自 phase-0 的构建（v1b 的 `data.train_sample` 之前，v1b 另外多用了 MNLI、GoEmotions 和 oasst2_zh 的样本）；每个发布模型实际用的条数记录在它自己的 `training_data.json` 里。 / Counts in this section are from the phase-0 build, before v1b's `data.train_sample` (v1b uses more MNLI, GoEmotions and oasst2_zh records); each published model's actual counts are in its own `training_data.json`.

数据来自 phase-0 构建产物 `data/unified/manifest.json`（2026-09-24 构建，已去重、已去污染）。
Counts come from the phase-0 build's `data/unified/manifest.json` (built 2026-09-24, de-duplicated and decontaminated).

| | 构建总量 Built | 排除后 After exclusions (`default`) | `permissive_strict`（无 SA / no SA） | nli26 重译后 If nli26 re-translated (`default`) |
|---|---:|---:|---:|---:|
| **zh train** | 208,461 | **136,978** | **88,137** | 208,461 |
| zh dev / test | 16,557 / 21,439 | 15,838 / 19,861 | — | 16,557 / 21,439 |
| **en train** | 108,134 | **108,134** | **64,891** | 108,134 |
| en dev / test | 4,902 / 26,105 | 4,902 / 26,105 | — | — |

按题型 / By decision type (train):

| 方案 Profile | zh choice | zh noul | zh score | en choice | en noul | en score |
|---|---:|---:|---:|---:|---:|---:|
| built | 84,254 | 109,240 | 14,967 | 55,302 | 31,329 | 21,503 |
| `default` | 44,827 | 87,806 | **4,345** | 55,302 | 31,329 | 21,503 |
| `permissive_strict` | 44,126 | 39,666 | **4,345** | 33,648 | 9,740 | 21,503 |

中文 `score` 在排除后只剩 oasst2_zh（4,345 条），是最明显的缺口。可以通过重译 nli26（加回约 1.06 万条）或者找新的有许可的中文评分数据来补。
After the exclusions, zh `score` has only oasst2_zh left (4,345 examples), which is the most visible gap. Fill it by re-translating nli26 (about 10.6k more) or by finding new licensed Chinese rating data.

---

## 8. 模型卡义务（可直接粘贴）/ Model-card NOTICE (paste-ready)

`zhjudge export` 会根据 `configs/licenses.yaml` 自动在模型卡里生成逐个来源的许可与署名表（含 `card_notice` 披露，例如 CrossWOZ 的标注者同意范围），并随附本文件。下面是手工发布时可用的汇总文本（`default` 方案）。
`zhjudge export` writes a per-source licence and attribution table into the model card from `configs/licenses.yaml` (including `card_notice` disclosures such as CrossWOZ's annotator consent) and ships this file next to it. The text below is a summary for manual releases (`default` profile).

```text
This model was trained on the following datasets. Their licences and required attributions:

Apache-2.0: COLD (Deng et al., 2022, THU-COAI); CSL (Li et al., 2022); CrossWOZ (Zhu et al., 2020, THU-COAI;
  annotators consented to research use); OpenAssistant OASST2 (Köpf et al., 2023).
MIT: CommonsenseQA (Talmor et al., 2019).
CC0-1.0: Civil Comments (Borkan et al., 2019; Jigsaw).
CC BY 4.0: MASSIVE (Copyright Amazon.com Inc. or its affiliates; built on SLURP, CC BY 4.0);
  BANKING77 (PolyAI); GoEmotions (Google); HelpSteer2 (NVIDIA). Converted into typed-decision format (modified).
[default profile only — share-alike]
CC BY-SA 3.0 / 4.0: PAWS and PAWS-X (Google LLC; source sentences from Wikipedia); SIB-200 and FLORES-200;
  BoolQ (Google); DBpedia-14 (Zhang et al., 2015; Wikipedia/DBpedia); MultiNLI (Williams et al., 2018; OANC,
  "Seven Swords" CC BY-SA 3.0, "Living History" and "Password Incorrect" CC BY 3.0); ARC (AI2).
Evaluation only (not trained on): Belebele, XCOPA, XWinograd, Global-MMLU, MInDS-14, LocalLLaMA/typed-decisions;
  benchmark sets under non-commercial or unclear terms (CLUE, LCQMC, ChnSentiCorp, Amazon reviews, XNLI, AG News,
  GLUE, DREAM) were used locally for scoring and decontamination only.
No Jev / TypeSafe API outputs were used for training.
```

---

## 9. 镜像改标签警告 / Mirror re-tag warnings

下列 HF 镜像的许可标签和上游矛盾，一律以上游为准。The following HF mirrors carry licence tags that contradict their upstream; the upstream always governs:

- `Tongjilibo/THUCNews` 和 `Tongjilibo/LCQMC` 标 apache-2.0，上游为研究用途或商用需签协议。`Tongjilibo/THUCNews` and `Tongjilibo/LCQMC` are tagged apache-2.0; upstream is research-only or needs a commercial agreement.
- `SetFit/amazon_reviews_multi_en` 标 apache-2.0，上游非商用且已下架。`SetFit/amazon_reviews_multi_en` is tagged apache-2.0; upstream is non-commercial and withdrawn.
- `GEM/RiSAWOZ` 标 cc-by-4.0，上游是 CC BY-NC 4.0。`GEM/RiSAWOZ` is tagged cc-by-4.0; upstream is CC BY-NC 4.0.
- `datatune/LogiQA2.0` 标 mit，上游是 CC BY-NC-SA 4.0。`datatune/LogiQA2.0` is tagged mit; upstream is CC BY-NC-SA 4.0.
- `mteb/amazon_counterfactual` 标 cc-by-4.0，上游是 CC BY-NC 4.0。`mteb/amazon_counterfactual` is tagged cc-by-4.0; upstream is CC BY-NC 4.0.
- `Maciel/FinCUGE-Instruction` 标 apache-2.0，上游没有许可。`Maciel/FinCUGE-Instruction` is tagged apache-2.0; upstream has no licence.
- `shibing624/nli_zh` 和 `shibing624/snli-zh` 的标签与数据卡正文（"用于学术研究"）矛盾。The tags of `shibing624/nli_zh` and `shibing624/snli-zh` contradict their card text ("用于学术研究", for academic research).
- 以下两处镜像标签更宽松，但上游许可也允许使用，所以不影响结论：`mteb/amazon_massive_intent`（标 apache-2.0，上游 CC BY 4.0）和 `google-research-datasets/go_emotions`（标 apache-2.0，上游 CC BY 4.0）。Two mirrors are tagged more loosely than upstream, but upstream still permits use, so the verdict is unchanged: `mteb/amazon_massive_intent` (apache-2.0 tag, upstream CC BY 4.0) and `google-research-datasets/go_emotions` (apache-2.0 tag, upstream CC BY 4.0).

---

## 10. 与 phase-0 普查的差异 / Changes vs the phase-0 census

phase-0 是本项目公开前的原型实验，它的普查和 manifest 没有公开，这里只引用结论和条数。 / phase-0 is this project's unpublished prototype; its census and manifest are not public, only their conclusions and counts are quoted here.

| 数据集 Dataset | phase-0 | 本次 This audit | 原因 Why |
|---|---|---|---|
| nli26 zh_mnli / zh_wanli / zh_fever | OK-INHERIT | **EXCLUDE** | 译文没有许可；已给出补救办法。No licence on the translations; remedy given. |
| Amazon counterfactual | OK | **EXCLUDE** | 上游 LICENSE 是 CC BY-NC 4.0。Upstream LICENSE is CC BY-NC 4.0. |
| CSL (CLUE 版) | OK | **EXCLUDE** | CLUE 2020 打包版没有许可；可用的是 `neuclir/csl`。The CLUE 2020 packaging has no licence; `neuclir/csl` is the one to use. |
| PAWS / PAWS-X | OK | **CONDITIONAL** | 句子源自 Wikipedia (CC BY-SA)。Sentences come from Wikipedia (CC BY-SA). |
| CrossWOZ | OK | **CONDITIONAL** | 需要披露标注者同意的范围是研究用途。Annotator consent was for research use and must be disclosed. |
| GoEmotions | OK | ALLOW（按 CC BY 4.0，不按 HF 卡的 apache-2.0） | 上游 google-research 仓库声明所有数据集为 CC BY 4.0。The upstream google-research repo states all datasets are CC BY 4.0. |

---

## 11. 方法 / Method

1. 对每个 HF id 调用 `https://huggingface.co/api/datasets/<id>`，读取 `cardData.license` 和 `license:` 标签，再读原始 README，逐行找许可、商用、研究等条款。For each HF id, fetch `https://huggingface.co/api/datasets/<id>` for `cardData.license` and the `license:` tags, then read the raw README line by line for licence, commercial and research terms.
2. 读取上游 GitHub 的 LICENSE 文件和 README 的许可章节（通过 raw.githubusercontent.com），以及官方页面（HIT-SZ ICRC、thuctc.thunlp.org、fever.ai、NYU MultiNLI、allenai.org）。Read upstream GitHub LICENSE files and README licence sections (via raw.githubusercontent.com), plus the official pages (HIT-SZ ICRC, thuctc.thunlp.org, fever.ai, NYU MultiNLI, allenai.org).
3. 镜像与上游冲突时以上游为准；证据缺失或互相矛盾时归为 EXCLUDE。Where a mirror conflicts with its upstream, the upstream wins; where evidence is missing or conflicting, the dataset is EXCLUDE.
4. 条数由 `configs/licenses.yaml` 与 phase-0 的 `manifest.json` 联合计算。Counts are computed by joining `configs/licenses.yaml` with the phase-0 `manifest.json`.
