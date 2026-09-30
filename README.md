# zhjudge：中文优先的开源「类型化决策」模型

[English below](#english)

> **独立项目声明**：zhjudge 是一个独立的开源项目，与 TypeSafe（Jev 的开发方）和 Convai（Laya 的开发方）没有任何隶属、合作、赞助或背书关系。"Jev"、"Laya" 是其各自所有者的名称，本仓库提到它们只是为了说明模型类别（Jev 式的类型化决策，即 "System One"）和所用的 Laya 检查点格式。本项目从未运行或测量过 Jev，不声称与 Jev 的接口兼容，不调用 Jev 的接口，也从不用它的输出训练（唯一标签来源不明的是评测集 typed-decisions，只用于评测，见 DATA_LICENSES.md）。

## 这是什么

一个**不生成文字、只做判断**的小模型（和 Jev / Laya 同类的 "System One"）。你给它一段内容和一个带选项的问题，它一次前向计算就返回每个选项的**校准过的概率**：

```text
内容：客户说：三月的账单被重复扣款了，今天不退款我们就取消订阅。
问题：应由哪个部门处理？   选项：账单 / 技术 / 销售
输出：账单 0.86   技术 0.05   销售 0.09      （示意数字）
```

三种问题类型：`choice`（多选一）、`score`（有序等级，如 1–5 星）、`noul`（是/否，输出为"是"的概率）。模型底座是 [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)（MIT，多语言），决策头直接使用 [Laya](https://github.com/NandhaKishorM/laya)（Apache-2.0）上游 `laya` 包里的模块，检查点按 Laya 的格式保存。本仓库的 `zhjudge compat` 在冒烟测试导出的模型上实测通过了 `laya` 参考运行时（与本仓库批量评估路径的概率最大差 5e-5，2026-09-29 实测）和 Laya 的 `/v1/systemone` 服务；`laya-mlx` 只能在 Apple 芯片上检查（phase-0 spike 曾在 Apple 芯片上实测通过，本仓库未重跑）；`laya-coreml` 的检查脚本是 `scripts/coreml_compat.py`，本仓库尚未实测。

**状态：开发中。** 训练流程已在 CPU 冒烟测试和 Kaggle T4 上端到端跑通。第一个完整模型 v0（2026-09-29）在多数评测集上校准误差明显低于基线，但在要对标的探针集上没有全面超过基线（NLI 明显退步），因此没有发布。v1（2026-09-30）改了训练数据的写法：XNLI 中文从 33.3% 回到 69.3%，与基线持平；但 XNLI 英文、英文对照仍低于基线，同样没有发布。v1b（2026-09-30，单独的配置，在 v1 上再加一组数据和训练配方的改动）在中文探针上显著超过基线（55.5% 对 47.3%，逐题配对区间整个在 0 以上），XNLI 英文追平（84.7%），XNLI 中文高 6 个点（区间含 0）；但 typed-decisions 仍低于基线（32.6% 对 35.1%，区间含 0）。v1b 已于 2026-09-30 作为**预览版**发布在 Hugging Face：[TerryGao9/zh-decision-mmbert-v1b](https://huggingface.co/TerryGao9/zh-decision-mmbert-v1b)（CC BY-SA 4.0，用法见下面的「[下载与使用](#下载与使用预览版)」）。详见[「训练记录」](#训练记录)。下面第一张表是要超过的基线，不是本项目的成绩。

## 要超过的基线（实测，来自 phase-0 spike）

laya-multilingual（`convaiinnovations/laya@5e7b2b1`，`multilingual` 子目录）在 spike 的探针集上的**实测**结果（phase-0 spike 是本项目公开前的原型实验，它的文件没有公开，这里只引用它的测量结果；探针集由 `zhjudge probes` 逐字节重建。来源：spike 的 `baselines/results/summary.md`，2026-09-24；官方 `laya` 0.3.7 PyTorch 运行时，Apple M5 Pro MPS，fp32）：

| 评测集 | 条数 | 准确率 | Brier ↓ | ECE-15 ↓ |
|---|---|---|---|---|
| 中文探针，中文提示 | 600 | **47.3%** | 0.813 | 0.341 |
| 中文探针，英文提示 | 600 | 50.2% | 0.792 | 0.334 |
| 英文对照（同类任务） | 360 | **68.1%** | 0.490 | 0.190 |
| XNLI 平行语料，中文 | 150 | **70.0%** | 0.464 | 0.166 |
| XNLI 平行语料，英文 | 150 | **84.7%** | 0.254 | 0.116 |
| typed-decisions 测试集 | 2,000 | 35.2% | 0.890 | 0.314 |

同一批题目，中文比英文低 15–21 个百分点，这就是本项目要补的差距。`zhjudge eval` 会把我们的模型和这些数字并排输出，并（默认）在相同题目上重新测一遍 laya-multilingual；本仓库在 CPU 上重测（2026-09-29 实测，`laya` 0.3.9，fp32）的准确率、Brier、ECE-15 与上表逐项一致（小数点后 4 位）。在 Kaggle T4 上重测时个别题目的结果与上表不同（XNLI 中文三次运行都是 69.3%；typed-decisions 在 v1、v1b 的运行里是 35.1%，v0 的运行里与上表相同；各差 1–2 道题），可能来自 CUDA 上的数值差异（上表在 Apple 芯片的 GPU 上测得，CPU 重测与它一致）；下面每个版本的 Δ 都用同一次运行里的重测值。

## 训练记录

### v0（2026-09-29，Kaggle T4，未发布）

`configs/train_base.yaml`（mmBERT-base，2 个 epoch，21.8 万条训练样本），单张 T4、fp16 + GradScaler。整次运行 3.1 小时，其中训练 2.8 小时。准确率 / Brier / ECE-15（后两者越低越好），基线是 laya-multilingual 在同一批题目上的重测：

| 评测集 | 条数 | v0 | laya-multilingual |
|---|---|---|---|
| 中文探针，中文提示 | 600 | 48.5% / 0.614 / 0.091 | 47.3% / 0.813 / 0.341 |
| 中文探针，英文提示 | 600 | 49.3% / 0.600 / 0.098 | 50.2% / 0.792 / 0.334 |
| 英文对照 | 360 | 56.7% / 0.554 / 0.082 | 68.1% / 0.490 / 0.190 |
| XNLI 平行语料，中文 | 150 | 33.3% / 0.703 / 0.204 | 69.3% / 0.464 / 0.171 |
| XNLI 平行语料，英文 | 150 | 43.3% / 0.613 / 0.099 | 84.7% / 0.255 / 0.113 |
| typed-decisions 测试集 | 2,000 | 35.0% / 0.724 / 0.125 | 35.2% / 0.890 / 0.314 |
| 本项目留出测试集（全部） | 34,596 | 72.1% / 0.372 / 0.034 | 47.9% / 0.722 / 0.236 |
| 本项目留出测试集（中文） | 13,363 | 76.1% / 0.327 / 0.046 | 51.5% / 0.693 / 0.230 |

- 从没训练过的中文评测集，准确率高于基线的有：Belebele 39.9% 对 34.9%（注意：v0 的训练集里有 SIB-200 句子，而 Belebele 的篇章由同一批 FLORES-200 句子组成，所以部分篇章的文字 v0 见过，只是没见过题目；v1 起训练时会去掉这些句子），XCOPA 64.6% 对 57.4%，XWinograd 65.8% 对 55.9%，MInDS-14 75.5% 对 56.0%。Global-MMLU 28.7% 对 25.7%（998 题，两者都接近四选一的随机水平 25%，95% Wilson 区间 25.9–31.5% 与 23.0–28.5% 重叠）不能算作超过基线。
- 概率校准：ECE 除 XNLI 中文（0.204 对 0.171）和 XCOPA（0.139 对 0.114）外都低于基线，多数只有基线的 1/2 到 1/7；但 Brier 在英文对照和 XNLI 上比基线差（因为准确率低）。XNLI 每种语言只有 150 题，这个样本量下即使完美校准的模型 ECE-15 也有约 0.05–0.12，那里的 ECE 差别都在噪声范围内。
- 问题：中文探针准确率只比基线高 1.2 个点（600 题的误差范围内）；中文 NLI 退到随机水平（中文探针里 NLI 60.0% → 33.3%，XNLI 中文 69.3% → 33.3%），XNLI 英文 84.7% → 43.3%，英文对照低 11 个点。
- 判断的原因（待 v1 验证）：探针（以及 Laya 接口的正常用法）把内容写成带字段名的 JSON 状态，例如 `{"前提": …, "假设": …}`，题目用反引号点名字段；v0 的训练数据全部是纯文本状态，NLI 的假设还写在题目里。模型没见过这种写法。中文 NLI 训练数据为 0（`nli26_zh_*` 没有许可）。

### v1：训练时的 Laya 式写法（`configs/train_base.yaml`，run 名 `mmbert-base-zh-v1`）

不加新数据源，许可结论不变，只改训练样本的写法（`src/zhjudge/data/views.py`）。探针集和留出测试集的抽取规则都不变（抽样种子固定为 `eval.seed: 13`，即 v0 的 `run.seed`），v0 的那份评测表（`eval.json` 的 `model` / `baseline_measured`，`eval.md` 的前半部分）按原来的方法计算，结果可以和 v0 直接比较；评测报告只是在后面加了几部分（见「[评测报告与 v0 比较](#评测报告与-v0-比较)」）。例外：去污染现在也检查 BoolQ 的问题和 DBpedia 的标题，如果其中有与探针文本相同的，对应的测试样本会被去掉。`zhjudge validate` 在训练前自动检查这一点（`eval_set_vs_v0`）：各来源的 dev/test 条数都与 v0 相同（`configs/eval_reference_v0.json`，取自 v0 的构建日志），而且没有来源因与探针重叠去掉 dev/test 样本（manifest 的 `dropped_eval_overlapping_probes`，v0 全为 0），test:* 就与 v0 是同一批题目；否则打印警告（不中止运行）。评测后 `eval.json` 里每个切片的指纹和 `zhjudge compare` 再逐条核对。

- 每个训练来源的转换器额外记下样本由哪几段文字组成（如前提 + 假设、用户 + 助手、标题 + 摘要）以及问的是什么。
- 训练时一半样本（`train.format_aug: 0.5`）按 Laya 的标准写法重写：JSON 状态加带名字的字段，题目用反引号点名字段。字段名每类有 2–4 种说法，每个任务的题目在每种语言里至少有两种说法，不固定成一个模板。其中 30% 用另一种语言写题目和字段名，有对照表时选项也一起翻译（中文内容配英文题目，英文内容配中文题目，例如英文 MNLI 配中文的「蕴含 / 中立 / 矛盾」）。
- 一半的选择题（`train.option_shuffle: 0.5`）打乱选项顺序，避免模型记住位置。
- 标签始终来自原数据集，只改写法。
- 去污染补了一个漏洞：训练文本（规范化后 ≥ 12 字）如果出现在只用于评估的数据集的某段文字里面，也会被去掉。v0 只比对整条文本，SIB-200 的训练句子出现在 Belebele 的篇章里时查不出来，因为两者都取自 FLORES-200。build 日志和 manifest 的 `dropped_train_inside_eval_only` 会给出去掉的条数。
- 校准温度改为按真实标签拟合。v0 误按标签平滑后的训练目标拟合，温度被推高到 1.09 / 1.20 / 1.20，校准后留出集上的 ECE 反而从 0.0350 升到 0.0354。
- 校准集在改写之后才从训练来源中划出（仍不参与训练），所以约一半也是 Laya 式写法，温度按训练时同样的混合分布拟合。v0 → v1 的 Brier / ECE 变化因此不只来自训练。
- 运行方式（不改训练内容，续训哈希不变，v1 的检查点照常续训）：GPU 上的 AdamW 改用 fused 实现（v0 是 foreach；续训沿用检查点里的实现），训练批次补齐到 16 的倍数（v0 为 32；分批相同，logits 相同，只有决策头 dropout 的随机数随补齐长度不同），训练前在单独进程里测一次不开梯度检查点时的显存（`zhjudge memprobe`），训练中多记几项监控（见「[流程与输出](#流程与输出)」）。
- 与探针的关系：探针题目和答案不进训练，去污染规则不变。训练里能生成的题目写法，即使不看字段名，也没有一条与探针的题目相同，`zhjudge validate` 会检查。但探针里的字段名（如 `前提`/`假设`、`sentence1`/`sentence2`）和 NLI 标签名（蕴含 / 中立 / 矛盾）都是通用写法，训练里也有。另外，v1 的写法是看了 v0 的探针结果之后才设计的，所以探针上的提升不能全算作对「没见过的写法」的泛化。

### v1 结果（2026-09-30，Kaggle 单张 T4，未发布）

代码为公开前的开发版本（开发提交 `2652941`，不在本仓库的历史里），`configs/train_base.yaml`。整次运行 2.98 小时，其中训练 2.6 小时：51 条/秒（v0 为 43），显存峰值 4.9 GiB（v0 为 6.0）。准确率单位为 %。Δ 是 v1 减去 laya-multilingual（同一批题目重测），方括号里是逐题配对的 95% 区间，`*` 表示区间不含 0。v0 → v1 的变化还没有做逐题配对：要用 v0 的 `eval_preds.jsonl` 跑 `zhjudge compare`。

| 评测集 | 条数 | v0 | v1 | laya-multilingual | Δ（v1 − 基线） |
|---|---|---|---|---|---|
| 中文探针，中文提示 | 600 | 48.5 | 50.5 | 47.3 | +3.2 [−2.1, +8.4] |
| 中文探针，英文提示 | 600 | 49.3 | 52.8 | 50.2 | +2.7 [−2.1, +7.4] |
| 英文对照 | 360 | 56.7 | 63.1 | 68.1 | −5.0 [−10.9, +0.9] |
| XNLI 中文 | 150 | 33.3 | 69.3 | 69.3 | 0.0 [−9.1, +9.1] |
| XNLI 英文 | 150 | 43.3 | 66.7 | 84.7 | −18.0 [−25.9, −10.1]* |
| typed-decisions 测试集 | 2,000 | 35.0 | 34.8 | 35.1 | −0.3 [−3.2, +2.6] |
| 本项目留出测试集（全部） | 34,596 | 72.1 | 71.5 | 47.9 | +23.6 [+23.0, +24.2]* |
| 本项目留出测试集（中文） | 13,363 | 76.1 | 75.0 | 51.5 | +23.4 [+22.5, +24.4]* |
| 从没训练过的评测来源 | 3,395 | 49.3 | 46.3 | 41.7 | +4.6 [+2.6, +6.6]* |
| 留出测试集改写成 Laya 式（test:laya） | 8,929 | – | 71.3 | 49.6 | +21.8 [+20.6, +23.0]* |

按登记的判定标准，四个主要集合与基线相比都「分不出差别」：中文探针、XNLI 中文、英文对照、typed-decisions 的区间都含 0。

- **写法问题解决了**：同一批测试题改写成 Laya 式后，v1 的准确率与纯文本相同（差 −0.1 [−0.6, +0.5]）。XNLI 中文从 33.3% 回到 69.3%。
- **NLI 的差距现在是能力，不是写法**：在改写成 Laya 式的 MNLI 测试题上，v1 71.6%，基线 82.4%（test:laya 的 mnli_en）。v1 只有 1.2 万条 MNLI 训练对。v1b 增到 3 万条；中文 NLI 训练数据仍为 0。
- **退步的地方**（探索性结果，每族 60–150 题）：
  - 中文探针的情感族，v0 77.1% → v1 57.1%，基线 71.4%。训练里没有情感数据，v1b 加入了 GoEmotions 的情绪倾向。
  - 英文对照的主题族（AG News），65.0% 对基线 91.7%。
  - typed-decisions 没有变化。
- **Belebele 39.9% → 35.6%**：去污染修复在构建时去掉了 SIB-200 中文训练集 701 条里的 633 条，它们都出现在 Belebele 的篇章里。v0 在 Belebele 上的领先大部分来自见过篇章文字。
- **选项被截断**：banking77 的 2,000 道测试题全部被截断（77 个选项每个只剩 3 个 token，有的截完完全相同，题目也只剩 8 个 token）。60 个选项的 MASSIVE 中文题的选项也全部被截断。v1b 把 `head_max_len` 从 256 调到 512。
- **校准**：v1 拟合的温度都小于 1（0.91 / 0.99 / 0.93），在留出的校准集上改善了 ECE（0.0295 → 0.0272），但在测试集上反而变差：留出测试集 ECE 0.052，温度为 1 时是 0.042。校准集取自训练集的划分，官方测试集的分布与它不同。怎么改留到 v2，不根据测试集调。
- **训练**：RL 项对 logits 的梯度是 CE 项的约 5 倍（`train_log.jsonl` 的 `glogit_*`，前者约 0.5–2，后者约 0.1）。v1b 把 RL 组从 4 调到 64，降低这部分噪声。
- **显存**：`memprobe` 估计不开梯度检查点时峰值 9.5 GiB，T4 可用 12.4 GiB，放得下，关掉后能再快约 20%。v1b 仍开着检查点，求稳。

### v1b：数据与训练配方（`configs/train_v1b.yaml`，run 名 `mmbert-base-zh-v1b`，2026-09-30 已训练，结果见下一节）

v1b = v1 再加一组数据和训练配方的改动。它们放在单独的 `configs/train_v1b.yaml`（`train_base.yaml` 的完整副本，只多下面这些键），不放进 v1：v1 对 v0 只衡量写法的改变，v1b 对 v1 衡量这一组改动（分不出组内哪一项起了作用）。这些键默认全部关闭，`train_base.yaml`（v1）训练的内容和续训哈希都不变。

- 数据（`data.train_sample`，只改训练集；dev/test 文件逐字节不变，评测集与 v1 相同）：
  - `mnli_en: 30000`：MNLI 训练对从 1.2 万增加到 3 万（唯一启用的 NLI 来源，v0 最差的任务族就是 NLI）。3 万条的样本是同一个随机排列的前 3 万条，包含 v1 的 1.2 万条。
  - `goemotions_en_valence: 8000`：再取 8,000 条 v1 没用到的 GoEmotions 训练评论，按数据集作者自己的情绪分组（sentiment_mapping：正面 12 种、负面 11 种、模糊 4 种）问整体情绪倾向（choice：positive / negative / neutral）或正负程度（3 级 score）。只标了 neutral 的评论算中性；含模糊情绪、正负混合或 neutral 与其他情绪并存的评论不用。估计约 6,800 条（实测：GoEmotions 训练条数从 v1 的 9,959 增到 16,556）。
  - `oasst2_zh_multi_attr` / `oasst2_zh_prompts`：oasst2_zh 的每条回复再问它的其他人工评分项（quality、helpfulness、creativity，至少 2 人评分），并加上有质量评分的用户提问；某一项最常见的等级占比超过 60% 时整项不用（build 日志打印每项的条数和最大占比）。估计约 1–1.5 万条中文打分题（v1 只有 4,345 条；实测 oasst2_zh 训练条数增到 15,457）。同一棵对话树的所有样本带同一个 `group`，校准集整组取或整组不取。
  - 语料的 manifest 记下 `train_sample`（`split_sources` 里按来源注明，两者都写进导出的 `training_data.json`）；`zhjudge train` 拒绝用别的 `train_sample` 构建的语料（提示重新构建），Kaggle 笔记本也按它决定是否重新构建。
- 模型：`model.head_max_len: 512`（v1 为 256）。v1 的实测显示，选项多的题目（banking77 的 77 个意图、MASSIVE 的 60 个意图）的选项和题目会被截断，见上面的「v1 结果」。只有这类题目的输入会变长；运行时从检查点的 `rl_agent_config.json` 读取这个值。这项改变了 banking77 和 MASSIVE 测试题的输入，所以 v1b 对 v1 在这几个来源上的差别包含这一项的作用。
- 训练：
  - `train.rl_group: 64`（v1 为 4）：RL 项每个样本 64 组噪声 logit 样本，梯度噪声更小；每个样本的基线和标准化都随组大小变化，RL 梯度相对 CE 约大 15%（√((1−1/64)/(1−1/4))）。
  - `train.format_aug_per_epoch: true`：第 2 个 epoch 每条训练样本重新抽一次写法和选项顺序（第 1 个 epoch 与 v1 相同；校准集样本在每个 epoch 都不参与训练；每个 epoch 一份分词缓存，多约 2.5 分钟分词和几百 MB 缓存）。
  - `train.format_variants: 0.25`：Laya 式写法的样本中四分之一再换一种 Laya 调用方常用的题目形式（`views.variant`）：noul 题加 true / false 的说明（其中 30% 还用自定义标签 是 / 否、yes / no）；5 级打分按 0–1 / 2 / 3–4 精确合并成 3 级，配自己的等级说明；NLI 选项写成字典（标签 → 说明）；BoolQ 写成「是 / 否」二选一。说明只在 `to_laya` 里按当前的选项顺序生成，答案仍由数据集的原标签确定，样本 id 不变。`VIEWS_VERSION` 升到 4：已有任务的模板和改写不变（test:laya 的每条改写与 v1 相同，`zhjudge compare` 照常比较），只有 `meta.laya_test` 和 test:laya 切片指纹因版本号不同而不同。
- 与探针的关系：这组改动是看了 v0 在各探针任务族上的结果之后设计的（NLI → MNLI 3 万条；情感、评分 → GoEmotions 情绪倾向、oasst2 评分；Laya 题目形式 → 格式变体），所以 v1b 在探针上的提升比 v1 更不能算作泛化。探针题目仍不进训练；用 v1b 的配置时 `zhjudge validate` 还检查新增的题目模板（即使不看字段名）和新增的说明、等级文字都不与探针的相同，标签名（正面 / 负面 / positive / negative / 是 / 否等通用名）与探针共用的只列出、不报错。新增的中文标签、等级和说明文字与 v1 的模板和对照表一样是本仓库编写的，不是数据集的官方译文；标签始终来自数据集本身。没有新的数据来源，许可结论不变。
- 在 Kaggle 上运行：设置单元格里 `CONFIG = "configs/train_v1b.yaml"`，另存为这个笔记本的一个新版本（Save & Run All），不要挂 v1 的输出（run 名不同，里面没有 v1b 能续训的东西）；以后续训 v1b 时挂 v1b 那个版本的输出（公开前的开发版本训练出的检查点，在本仓库的代码下不能续训：改名后配置哈希不同，笔记本会停下来说明）。同一个会话里跑完 v1 再跑 v1b，笔记本发现已有语料的 `train_sample` 不同，会重新构建。耗时（2026-09-30 在 Kaggle 单张 T4 上实测）：训练样本 25.3 万条（v1 为 21.7 万），训练 3.2 小时（v1 为 2.6 小时），整次运行 3.49 小时。

### v1b 结果（2026-09-30，Kaggle 单张 T4，已发布预览版）

已发布：[TerryGao9/zh-decision-mmbert-v1b](https://huggingface.co/TerryGao9/zh-decision-mmbert-v1b)（Hugging Face，预览版；模型卡开头的状态表与下表出自同一份 `eval.json`）。

代码为公开前的开发版本（开发提交 `96c5375`，不在本仓库的历史里），`configs/train_v1b.yaml`。整次运行 3.49 小时，其中训练 3.2 小时：25.3 万条训练样本（v1 为 21.7 万），49 条/秒，显存峰值 4.9 GiB。准确率单位为 %。Δ 是 v1b 减去 laya-multilingual（同一批题目重测），方括号里是逐题配对的 95% 区间，`*` 表示区间不含 0。v0 / v1 / v1b 之间的比较是分开测的，还没有做逐题配对（要用各自的 `eval_preds.jsonl` 跑 `zhjudge compare`）。

| 评测集 | 条数 | v0 | v1 | v1b | laya-multilingual | Δ（v1b − 基线） |
|---|---|---|---|---|---|---|
| 中文探针，中文提示 | 600 | 48.5 | 50.5 | **55.5** | 47.3 | +8.2 [+3.4, +13.0]* |
| 中文探针，英文提示 | 600 | 49.3 | 52.8 | **57.3** | 50.2 | +7.2 [+2.6, +11.7]* |
| 英文对照 | 360 | 56.7 | 63.1 | 66.9 | 68.1 | −1.1 [−6.4, +4.2] |
| XNLI 中文 | 150 | 33.3 | 69.3 | **75.3** | 69.3 | +6.0 [−2.1, +14.1] |
| XNLI 英文 | 150 | 43.3 | 66.7 | **84.7** | 84.7 | 0.0 [−7.2, +7.2] |
| typed-decisions 测试集 | 2,000 | 35.0 | 34.8 | 32.6 | 35.1 | −2.5 [−5.6, +0.6] |
| 本项目留出测试集（全部） | 34,596 | 72.1 | 71.5 | 73.9 | 47.9 | +26.1 [+25.5, +26.7]* |
| 本项目留出测试集（中文） | 13,363 | 76.1 | 75.0 | 76.4 | 51.5 | +24.9 [+23.9, +25.8]* |
| 从没训练过的评测来源 | 3,395 | 49.3 | 46.3 | 49.1 | 41.7 | +7.4 [+5.5, +9.4]* |
| 留出测试集改写成 Laya 式（test:laya） | 8,929 | – | 71.3 | 73.7 | 49.6 | +24.2 [+23.0, +25.3]* |

按登记的判定标准：
- 中文探针（600 题）：比基线高，区间整个在 0 以上。中文提示和英文提示都是这样，按任务族的宏平均也是（+7.6 [+2.5, +12.8]*）。
- XNLI 中文、英文对照、typed-decisions：与基线分不出差别，区间都含 0。
- Brier 在所有主要集合上都低于基线，其中中文探针、XNLI 中文、typed-decisions 的区间不含 0。

变化最大的地方：
- **NLI**：MNLI 训练对从 1.2 万增到 3 万，XNLI 英文 66.7% → 84.7%（追平基线），MNLI 测试集 74.8% → 80.6%，XNLI 中文 69.3% → 75.3%。中文 NLI 训练数据仍为 0。
- **情感和评分**（探索性结果，每族 70–90 题）：中文探针的情感族 57.1% → 77.1%（加入了 GoEmotions 的情绪倾向），评分族 26.7% → 35.6%（oasst2 中文加了 1.1 万条打分题）。
- **选项截断**：`head_max_len` 调到 512 后，训练、dev、校准样本的选项和题目全部完整，测试集里只剩 Global-MMLU 的 10 道题有截断；banking77 从 75.4% 升到 88.4%。
- **校准**：拟合的温度接近 1（0.98 / 1.08 / 1.01），校准后留出测试集的 ECE 为 0.048（温度为 1 时 0.052）。

仍然不行的地方：
- **typed-decisions** 32.6%（基线 35.1%）：security_incidents 23.4% 对 39.4%、customer_service 33.0% 对 42.0%（区间都在 0 以下）；invoice_processing 44.0% 对 30.8%（区间在 0 以上）。这类题的状态是结构化的工作流记录，一个状态问好几个问题，启用的数据源里没有同类数据。
- **英文对照的主题族**（AG News）68.3%，基线 91.7%：启用的来源里没有英文新闻分类数据。

需要注意的地方：
- v1b 的改动是看了 v0 在各探针任务族上的结果之后设计的，所以探针上的提升不能全算作泛化。最不针对探针的尺子是「从没训练过的评测来源」：v1b 49.1%，基线 41.7%，v1 46.3%。
- RL 组从 4 调到 64 之后，RL 项对 logits 的梯度仍是 CE 项的约 8–25 倍（`glogit_*`：约 0.7–1.5 对 0.06–0.08）。组变大只减小了噪声，没有改变两项的比例。

### 判定标准（看到 v1 和 v1b 的评测结果之前登记）

- 主要标准：与 v0、以及与在同一批题目上重测的 laya-multilingual 逐题配对的准确率差值，95% 区间不含 0，看四个集合：中文探针（600 题）、XNLI 中文（150）、英文对照（360）、typed-decisions（2,000）。每个集合分别判定：区间整个在 0 以上算提高，整个在 0 以下算下降，含 0 算分不出差别。最不针对探针的尺子是从没训练过的评测来源（`test:group:eval-only`）。
- v1b 对 v1 用同样的方法（`zhjudge compare` 两边的 `eval_preds.jsonl`）。任务族层面（每族 70–150 题）的变化和 ECE 的变化只作探索性结果。
- 所有区间只反映抽到哪些题目的随机性：每个版本只训练一次，看不出训练随机种子的影响，v1 与 v1b 之间的小差别重训一次未必重现。在 v1 和 v1b 之间选择时用非探针指标 test:* 和 test:laya（两次运行评的是同一批题目，`zhjudge compare` 会核对），两者的探针结果都报告，探针不用来挑模型。校准集监控不用来比较：两次运行留出的校准样本几乎不同，v1b 的还含格式变体（如 3 级合并的打分题），它只用来看单次训练的过程。

## 下载与使用（预览版）

v1b 预览版：[TerryGao9/zh-decision-mmbert-v1b](https://huggingface.co/TerryGao9/zh-decision-mmbert-v1b)（CC BY-SA 4.0，Laya 检查点格式）。它还不是正式版：中文探针上显著好于 laya-multilingual，typed-decisions 上略低（分不出差别），逐项结论见模型卡开头的状态表。概率在训练数据的分布上校准过，用在你自己的数据上之前请先抽查一下校准。

```python
# pip install laya==0.3.9（本仓库 `zhjudge compat` 实测过的版本）
import laya

agent = laya.load("TerryGao9/zh-decision-mmbert-v1b")  # 从 Hugging Face 只下载权重、配置和分词器
print(agent.predict("客户说：三月的账单被重复扣款了，今天不退款我们就取消订阅。", {
    "churn": {"type": "noul", "instructions": "客户是否威胁要取消？"},
    "dept": {"type": "choice", "instructions": "应由哪个部门处理？",
             "criteria": {"billing": "账单、付款、退款", "technical": "故障", "sales": "新合同"}},
}))
```

## 在云端训练

仓库不包含任何数据或模型权重，全部在运行时从 Hugging Face 下载。

**训练数据白名单**：`configs/datasets.yaml` 已按许可审计（`configs/licenses.yaml`，人读版 [DATA_LICENSES.md](DATA_LICENSES.md)）打开了 24 个来源：class 为 ALLOW 或 CONDITIONAL 且有转换器的（其中 5 个只用于评估）。`nli26_zh_*` 三个中文 NLI 译本没有许可，保持关闭。`build` 和 `validate` 会按审计复核，审计不允许的来源即使被打开也会被拒绝。默认用 `default` 方案（含 7 个相同方式共享来源，权重按 CC BY-SA 4.0 发布）；要 Apache-2.0 权重，加 `--set data.license_profile=permissive_strict --set export.license=apache-2.0`。

### Colab / Kaggle（免费 T4 16 GB）

打开 [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)（Colab：<https://colab.research.google.com/github/TerryG907/zhjudge/blob/main/notebooks/colab_train.ipynb>），选择 T4 GPU，从上到下运行即可。笔记本会克隆仓库、安装 uv、跑完整流程、打印评估结果，可选把导出的模型存到 Google Drive 或推送到 Hugging Face（Token 运行时粘贴，只在内存里，不会写进文件）。

- Colab 免费会话可能中途断开：打开 `USE_DRIVE`，把 `runs/` 放在 Google Drive 上，重连后从上到下重新运行全部单元格，就会从最近的检查点继续（mmBERT-base 的可续训检查点约 4 GB（估算：fp32 权重 + AdamW 状态），会占用 Drive 空间，训练完成后自动删除）。
- 仓库是公开的，不需要 GitHub 令牌。只有用私有仓库（例如你用 GitHub 的 Import repository 建的私有副本）时：Colab 从 GitHub 打开笔记本时勾选 "Include private repos"；笔记本里把 `REPO_URL` 改成你的仓库地址、`PRIVATE_REPO` 设为 `True`，运行时粘贴一个只读的 GitHub fine-grained token（只在内存里，不写进 `.git/config` 或任何文件）。
- Kaggle：请用专门的 [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb)，见下面的[「在 Kaggle 上训练」](#在-kaggle-上训练)（自动续训、可选双卡）。这个 Colab 笔记本在 Kaggle 上也能跑，但只用一块 GPU，续训要手动设 `RESUME_FROM`。

**需要多久？**（2026-09-29 在 Kaggle 单张 T4 上实测，v0）
- 训练量：19 个训练来源、每个来源最多 3 万条，共 221,887 条，减去 4,000 条校准集后 217,887 条；平均 128 token；2 个 epoch 共 14,932 次参数更新。
- 训练：每秒约 43 条，**每个 epoch 1.4 小时**，2 个 epoch 2.8 小时；fp16 + GradScaler，梯度检查点打开，显存峰值 6.0 GiB（T4 共 15 GiB），全程没有非有限损失。
- 整次运行 3.1 小时：构建数据 2.2 分钟（下载 877 MB），分词约 2.5 分钟，训练 2.8 小时，校准 0.8 分钟，评测 9.1 分钟（含重测 laya-multilingual），兼容性检查 2.1 分钟。
- 训练日志会实时显示 `ETA`；结束后 `runs/<name>/train_stats.json` 里的 `hours_per_epoch_estimate` 是按实测吞吐算出的数字。
- v1 和 v1b（2026-09-30 在 Kaggle 单张 T4 上实测）：v1 每秒 51 条，训练 2.6 小时，整次运行 2.98 小时，显存峰值 4.9 GiB；v1b 训练样本 25.3 万条，每秒 49 条，训练 3.2 小时，整次运行 3.49 小时。
- 参考（spike 实测，不是 T4）：Apple M5 Pro 上用 MLX bf16 训练 mmBERT-base，机器空闲时 59 条/秒（5,000 条小样本，平均 149 token），与其他任务并行时 15 条/秒。

### 在 Kaggle 上训练

用 [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb)。它克隆本仓库（公开仓库不需要令牌；私有仓库从 Kaggle Secrets 读取 `GITHUB_TOKEN`），装好依赖，依次跑 构建数据 → 校验 → 训练 → 校准 → 评测 → 导出 → 兼容性检查，最后把导出的模型、评测报告和日志打包到 `/kaggle/working/outputs`。T4 上用 fp16 自动混合精度 + GradScaler，默认单卡；训练中定期把可续训检查点存到 `/kaggle/working`。用 Save & Run All 运行时，快到 Kaggle 的 12 小时上限会先保存检查点并暂停，挂上这个版本的输出再运行一次就能接着训。

#### 1. 准备 GitHub 令牌（可选：只有私有仓库需要）

本仓库是公开的，可以直接跳到第 2 步。只有训练你自己的私有副本（例如用 GitHub 的 Import repository 建的私有仓库；公开仓库的 fork 不能设为私有）时才需要令牌，并在设置里把 `REPO` 改成它：

1. GitHub 右上角头像 → **Settings** → 左侧最下方 **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**。
2. **Token name** 随意（如 `kaggle-zhjudge`），**Expiration** 选一个到期日；**Repository access** 选 **Only select repositories**，只选你那个私有仓库；**Permissions** 里把 **Contents** 设为 **Read-only**（Metadata 只读会自动带上），其他都不要。
3. 点 **Generate token**，复制令牌（只显示这一次）。只把它粘贴到下面第 2 步的 Kaggle Secret 里，不要贴进笔记本、仓库或任何文件。

#### 2. 在 Kaggle 上建笔记本（只做一次）

1. 登录 [kaggle.com](https://www.kaggle.com)。账号需要先完成手机验证（头像 → **Settings** → **Phone verification**），否则不能用 GPU 和 Internet。
2. 在 GitHub 打开本仓库的 `notebooks/kaggle_train.ipynb`，点右上角的 **Download raw file** 下载到本地。
3. Kaggle 左侧 **Create**（＋）→ **New Notebook**；在笔记本编辑器的菜单 **File** → **Import Notebook**，选择刚下载的 `kaggle_train.ipynb` 导入。
4. 右侧边栏 **Session options**：**Accelerator** 选 **GPU T4 x2**；**Internet** 打开（On）。
5. （只有私有仓库需要）菜单 **Add-ons** → **Secrets** → **Add Secret**：**Label** 填 `GITHUB_TOKEN`，**Value** 粘贴第 1 步的令牌，保存；确认列表里 `GITHUB_TOKEN` 那一行已勾选（附加到本笔记本）。
6. 需要改设置时只改第一个代码单元格（见下面的「设置项」），默认值就是正式训练。

#### 3. 开始训练

- 正式训练：右上角 **Save Version** → 选 **Save & Run All (Commit)** → **Save**。它在后台运行，可以关掉浏览器；进度在该版本页面的 **Logs** 里看。
- 想先确认整条流程：把 `CONFIG` 改成 `configs/train_smoke.yaml` 跑一次（mmBERT-small、20 步，在 CPU 上几分钟），确认没问题再改回来。
- 也可以在编辑器里点 **Run All** 边看边跑，但浏览器断开太久时交互式会话可能被回收，`/kaggle/working` 里的检查点也会跟着丢失（除非事先在 Session options 里打开了保留文件的 Persistence）；长时间训练请用 Save & Run All。
- 耗时：v0 在单张 T4 上整次运行实测 3.1 小时（训练 2.8 小时），一次会话就能跑完，见上面的「需要多久？」。

#### 4. 断线、超时后续训

- 训练每 `SAVE_EVERY` 次参数更新（默认用配置里的 500）把可续训检查点写到 `/kaggle/working/zhjudge/runs/<run>/ckpt/last.pt`（mmBERT-base 约 4 GB，覆盖写入），旁边的 `last.json` 记录进度。
- Kaggle 单次会话最多 12 小时。笔记本在会话开始后 `SESSION_HOURS − POST_TRAIN_HOURS`（默认 11.5 − 1 = 10.5 小时）时保存检查点并暂停训练，然后照常打包输出，状态显示 `paused`。时间从本会话第一次运行设置单元格开始算，在同一会话里重新运行不会重新计时；交互式会话打开后闲置的时间不算在内，所以打开后请尽快运行。
- 续训：打开这个笔记本的编辑器 → 右侧 **Input** → **Add Input** → **Your Work** 里找到这个笔记本本身，选它最新版本的输出添加；然后再 **Save Version → Save & Run All**。笔记本会在 `/kaggle/input` 里自动找到用同样配置训练、进度最靠前的检查点（连同当时构建的数据）复制回来继续；如果找到的都是别的配置训练的，笔记本会停下来说明，不会悄悄从头训练。之后每次续训，确认挂的是最新那个版本的输出。
- 同一个交互式会话里重新 **Run All**：直接从 `/kaggle/working` 里的检查点继续。交互式会话被回收后，就只能从上一个已保存版本的输出继续。
- 单卡和双卡的检查点可以互相续训。想丢掉旧进度从头来（或者改了 `LICENSE_PROFILE` 等影响训练内容的设置），只在那一次运行把 `RESUME` 设为 `False`，之后一定改回 `True`，否则每次运行都会从头训练。

#### 5. 结果在哪里

版本页面 → **Output** → `outputs/`：

| 路径 | 内容 |
|---|---|
| `outputs/model/<导出名>/` | 导出的 Laya 格式检查点（含模型卡、DATA_LICENSES.md）；只有本次运行成功导出时才有 |
| `outputs/reports/` | 评测报告 `eval.md` / `eval.json` / `eval_preds.jsonl`（重测了基线时另有 `eval_preds_baseline.jsonl`），以及 `calibration.json`、`compat.json`、`train_stats.json`、数据清单与校验结果；每个文件只在对应步骤本次运行成功时才有（暂停的运行里是 `checkpoint_progress.json`）。`STATUS.md` 里还写明评测集是否与 v0 相同 |
| `outputs/logs/` | 每一步的日志、`train_log.jsonl`、`dev_log.jsonl` |
| `outputs/STATUS.md`、`status.json` | 本次运行的状态（`done` / `paused` / `failed:<步骤>`）和下一步该做什么 |
| `outputs/zhjudge-<run>-outputs.tar.gz` | 以上全部打成一个包，方便一次下载 |

`/kaggle/working/zhjudge/` 里是构建好的数据、训练检查点和日志，下次续训要用。

#### 6. 设置项（第一个代码单元格）

| 设置 | 默认 | 说明 |
|---|---|---|
| `REPO` | `TerryG907/zhjudge` | 要克隆的 GitHub 仓库；用自己的私有副本时改成 `用户名/仓库名`，并按第 1 步设置 Secret `GITHUB_TOKEN` |
| `BRANCH` / `CONFIG` | `main` / `configs/train_base.yaml` | 要训练的分支和配置（`configs/train_v1b.yaml` = v1b，作为新版本运行，见上面的 v1b 一节） |
| `RUN_NAME` | 空 | 空 = 配置里的 `run.name`；续训靠它找检查点，前后几次要一致 |
| `PRECISION` | `fp16` | T4 不支持 bf16，用 fp16 + GradScaler；连续出现 NaN 时改成 `fp32` |
| `USE_TWO_GPUS` | `False` | 用 torchrun 在两张 T4 上做数据并行，见下面的说明 |
| `LICENSE_PROFILE` | `default` | `permissive_strict` = 去掉相同方式共享的来源，权重可用 Apache-2.0（要从头训练：那一次把 `RESUME` 设为 `False`） |
| `SAVE_EVERY` | `0` | 每多少次参数更新存一次检查点，0 = 用配置里的值 |
| `SESSION_HOURS` / `POST_TRAIN_HOURS` | `11.5` / `1.0` | 本次 Kaggle 会话的总时长上限（从第一次运行设置单元格算），以及为校准、评测、导出、打包预留的时间 |
| `RESUME` | `True` | 自动从最近的检查点续训；`False` 只用于某一次从头训练，之后改回 `True` |
| `EVAL_BASELINE` | `True` | 评测时在同一批题目上重测 laya-multilingual（多下载约 650 MB） |
| `COMPARE_PREDS` / `COMPARE_LABEL` | 空 / `v0` | 上一版的 `eval_preds.jsonl`（例如把 v0 的这个文件做成 Kaggle Dataset 挂上，填 `/kaggle/input/<数据集>/eval_preds.jsonl`）：评测报告里加上与它逐题配对的比较；只传给评测这一步，不影响续训 |
| `MAKE_TARBALL` | `True` | 另外生成 `.tar.gz` |
| `EXTRA_SETS` | `[]` | 额外的 `--set`，如显存不足时 `["train.max_tokens=4096"]`（改训练内容的键后旧检查点不能续训） |

**双卡（可选）**：`USE_TWO_GPUS = True` 时，笔记本先让两张卡用 NCCL 做一次 all-reduce 自检，通过才用 `torchrun --nproc_per_node=2` 训练，否则自动退回单卡。每次参数更新看两批数据（有效批量翻倍），更新次数减半，学习率不变，所以结果和单卡不会完全一样。两张 T4 之间没有 NVLink，笔记本默认设 `NCCL_P2P_DISABLE=1`（走主机内存，更稳）。双卡在 Kaggle T4 上的实际加速**尚未实测**；本仓库只在 CPU（gloo 后端）上验证过双进程训练、暂停和跨卡数续训。

**令牌安全**（用私有仓库时）：令牌只从 Kaggle Secrets 读取，只在内存里，通过环境变量交给那一条 `git clone` 命令：不出现在命令行、不写进 `.git/config` 或任何文件，也不打印。打包前笔记本还会扫描 `/kaggle/working` 里的文本类文件（跳过权重、检查点、分词缓存 `.pt/.safetensors/.pkl/.tmp` 和超过 256 MB 的文件），万一有文件含令牌就删除该文件并提示你到 GitHub 作废令牌。

### 发布到 Hugging Face

不用重新训练：[`notebooks/kaggle_publish.ipynb`](notebooks/kaggle_publish.ipynb) 直接拿某次 Kaggle 训练版本输出里的 `outputs/model/<名字>/`，用当前代码重新生成模型卡，检查结构和许可后上传（`zhjudge publish`）。模型卡开头有「状态」一节：写明这是预览版，并按逐题配对的 95% 区间逐项给出与 laya-multilingual 相比「更好 / 分不出差别 / 更差」。仓库默认建成**私有**的，检查无误后再改成公开。

1. **Hugging Face 令牌**（只做一次）：登录 huggingface.co → 头像 → **Settings → Access Tokens → Create new token** → Token type 选 **Write** → 名字随意（如 `kaggle-publish`）→ **Create token** → 复制。不要发给任何人，也不要粘贴到笔记本、仓库、Issue 或任何聊天工具里，只存进下一步的 Kaggle Secret。
2. **存进 Kaggle**：在 Kaggle 新建一个笔记本（**File → Import Notebook** 上传 `notebooks/kaggle_publish.ipynb`），**Add-ons → Secrets → Add Secret**：Label 填 `HF_TOKEN`，Value 粘贴令牌，勾选本笔记本（公开仓库不需要 `GITHUB_TOKEN`；私有仓库再把训练时用的 `GITHUB_TOKEN` 也勾选上）。
3. **挂上模型**：右侧 **Input → Add Input → Your Work → Notebooks**，选训练笔记本、要发布的那个版本（如跑 v1b 的版本）。**Session options**：Accelerator 选 **None**，Internet 打开。
4. **运行**：第 1 个单元格填 `HF_REPO_ID = "你的用户名/模型名"`（如 `TerryGao9/zh-decision-mmbert-v1b`；名字里最好不带 "jev"，免得被当成官方模型；`CONFIG` 保持训练时用的配置），然后 **Run All**，几分钟就好。输出里的 `No module named 'wrapt'` 来自 Kaggle 自带的启动脚本，不影响结果。
5. **检查并公开**：打开最后打印的 `https://huggingface.co/你的用户名/模型名` 看模型卡和文件，没问题就到仓库的 **Settings → Change repo visibility** 改成 Public。

模型卡里有许可（CC BY-SA 4.0）、每个训练数据集的署名义务和独立项目声明，随附 `DATA_LICENSES.md`；冒烟测试的模型、以及权重许可与相同方式共享的训练来源不符的模型，`zhjudge publish` 都会拒绝上传。

### AutoDL / RunPod 等租用 GPU

```bash
git clone https://github.com/TerryG907/zhjudge && cd zhjudge
pip install uv                      # 或 curl -LsSf https://astral.sh/uv/install.sh | sh
CONFIG=configs/train_base.yaml bash scripts/run_all.sh
```

仓库是公开的，上面的 https 地址直接可用。私有副本请用 SSH 地址（在租用机器上配一个只读 deploy key），不要把 token 写进 URL。断线后重跑同一条命令即可续训（已完成的训练会跳过）。国内机器访问 Hugging Face / GitHub 可能需要平台的网络加速，或把 `HF_ENDPOINT` 设为你信任的镜像。

`scripts/run_all.sh` 会自动选择合适的 PyTorch 2.14.0 版本：默认（PyPI，CUDA 13.0）需要 NVIDIA 驱动 ≥ 580、算力 ≥ 7.5（T4/A10/L4/A100/3090/4090/H100）；驱动更旧或 V100 会自动改用 CUDA 12.6 版（`TORCH_VARIANT=cu126`）；没有 GPU 则用 CPU 版。精度也自动选择：Ampere 及更新的卡用 bf16，T4/V100 用 fp16 + GradScaler，CPU 用 fp32；显存 < 20 GiB 时自动开启梯度检查点。

没有 GPU 的机器（包括多数云端开发环境）只适合跑冒烟测试。

## 本地冒烟测试（CPU，约 1 分钟）

```bash
nice -n 15 bash scripts/smoke.sh
```

用 `jhu-clsp/mmBERT-small`、约 200 条样本、20 步，在 CPU 上把 构建数据 → 校验 → 训练 → 校准 → 评估 → 导出 → 兼容性检查 全部跑一遍（即使有 GPU / MPS 也强制 CPU）。在 Apple M5 Pro 上从全新克隆运行（依赖和模型已缓存，全新 `.venv`；2026-09-29）实测约 60 秒：build 14.3 s、validate 1.5 s、train 33.2 s、calibrate 2.0 s、eval 3.8 s、export 0.1 s、compat 4.1 s。第一次运行需要下载约 0.64 GB（mmBERT-small 564 MB，3 个数据集 73 MB；空缓存实测）。冒烟测试用的 3 个数据集（cold_zh、massive_zh、oasst2_zh）在许可审计中都是 ALLOW。冒烟测试的指标没有意义，导出的模型卡会标注"SMOKE TEST — NOT FOR RELEASE"。

## 流程与输出

`scripts/run_all.sh` = `uv sync` 之后执行 `zhjudge all`，依次为下表各步（单独运行某一步：在仓库目录里 `uv run zhjudge <命令> --config configs/train_base.yaml`）：

| 命令 | 作用 | 输出 |
|---|---|---|
| `zhjudge probes` | 重建基线探针集（与 spike 逐字节一致） | `data/probes/` |
| `zhjudge build` | 按白名单下载、转换、去重、去污染，确定性划分 | `data/unified/`（含 sha256 清单） |
| `zhjudge validate` | 独立校验：格式、泄漏、探针重叠、白名单、评测要打分的题目 id 不重复；评测集与 v0 的比较（只警告） | `data/unified/validation.json` |
| `zhjudge train` | 训练（可续训、定期验证） | `runs/<name>/model/`、日志 |
| `zhjudge calibrate` | 在留出的校准集上拟合温度 | 写入 `model/rl_agent_config.json`，`calibration.json` |
| `zhjudge eval` | 准确率、Brier、ECE-15，分语言/任务/题型，对比 laya-multilingual；区间、配对差值等见下 | `runs/<name>/eval.md`、`eval.json`、`eval_preds.jsonl` |
| `zhjudge export` | Laya 格式检查点 + 模型卡 | `runs/<name>/export/<名字>/` |
| `zhjudge compat` | 用 Laya 官方运行时加载导出结果并比对概率 | `runs/<name>/compat.json` |
| `zhjudge compare A B` | 两次运行的 `eval_preds.jsonl` 逐题对比（不在 `all` 里，不需要模型） | 屏幕，`--out` 写 md + json |
| `zhjudge memprobe` | 一步训练在不开梯度检查点时的显存峰值（只报告，不在 `all` 里；`train.memprobe: true` 时 `train` 在 CUDA 上先在单独进程里自动跑一次，出错或超时 15 分钟只记日志；没有 CUDA 时直接跳过） | `runs/<name>/memprobe.json`，也写进 `train_stats.json` |
| `zhjudge publish` | 把一个已导出的模型（如 Kaggle 输出里的 `outputs/model/<名字>/`）重新生成模型卡后上传到 Hugging Face（令牌只从 `HF_TOKEN` 环境变量读取；默认私有仓库；不在 `all` 里） | 上传后的 Hugging Face 仓库 |

每个命令都接受 `--config` 和 `--set key=value`。常用环境变量：`CONFIG`、`ZHJUDGE_DEVICE`、`ZHJUDGE_PRECISION`、`RUN_NAME`、`ZHJUDGE_RUNS_DIR`（例如指向 Google Drive）、`TORCH_VARIANT`。

**训练时的监控（只记录，不改变训练）**：`dev_log.jsonl` 每次验证除了总体、题型和语言，还有每个来源（`source:<名字>`）和各来源的平均（`macro_src`）。dev 只覆盖 10 个来源，没有打分题和 NLI，所以 `train.monitor_calib_per_source`（`train_base.yaml` 设 120，默认 0）另从留出的校准集里每个训练来源最多取 120 条（三种题型都有，约一半是 Laya 式写法），在第 0 次更新、每次验证和训练结束时各评一次，按来源、题型和写法（`fmt:laya` / `fmt:plain`）记在同一行的 `calib_slice` 里（最终结果也在 `train_stats.json` 的 `calib_slice_final`）。每条训练日志还记下本批的题型数（`qtypes`），每 `log_every` 步记下 RL、CE、act 三项损失对决策头输出的梯度范数（`glogit_rl` / `glogit_ce` / `glogit_act`，也打印在日志行里）。监控只在 0 号进程上运行、不做进程间通信；任何一项出错只打印一次并关掉该项（`monitors_off`），训练照常。

### 评测报告与 v0 比较

`zhjudge eval` 先写出与 v0 相同的报告（`eval.json` 的 `model` / `baseline_measured` / `meta`，`eval.md` 的前半部分，`eval_preds.jsonl`；v1 每行多一个 `slices` 字段），再在后面追加以下各部分（`meta` 里另加 `eval_set`、`option_fit`、`laya_test`）。追加部分任何一项出错只记在 `eval.json` 的 `report_errors` 里并打印警告，不影响主报告，也不改变退出码：

- 每个切片准确率的 95% Wilson 区间（把每道题当作相互独立；typed-decisions 一个案例有几道题，这个区间偏窄，与基线的配对差值才按案例重抽样），以及 ECE 下限：一个完美校准的模型在同样的置信度和样本量下也会有的 ECE-15，接近它的 ECE 只是噪声。
- 与在同一批题目上重测的 laya-multilingual 逐题配对的差值（准确率、Brier、NLL）和 95% 区间（typed-decisions 同一案例的几道题一起重抽样），以及精确 McNemar 检验。六个基线探针集都分任务族 / 题型列表，另有宏平均（每个来源或任务族权重相同，准确率区间用正态近似）。
- `test:group:trained` / `test:group:eval-only`：训练过的来源与只用于评测、从没训练过的来源（Belebele、XCOPA、XWinograd、Global-MMLU、MInDS-14）分开统计。
- 温度为 1 时的 Brier / ECE-15 / NLL（`uncalibrated`）：v0 的温度按平滑后的训练目标拟合（偏高），v1 按真实标签拟合，两者的校准要比 T=1 的数字（`zhjudge compare --t1` 用两边的 `eval_preds.jsonl` 和 `eval.json` 算出 v1 与 v0 在 T=1 下的 ECE-15 和逐题的 Brier / NLL 差）。`calibration.json` 另有 `calib_crossfit`：校准集一半拟合、另一半打分的样本外效果。
- `test:laya:*`（`train_base.yaml` 设 `eval.laya_test: 500`，默认 0）：每个来源最多 500 条 test:* 样本按训练时的模板改写成 Laya 式（JSON 状态、字段名、点名字段的题目；30% 用另一种语言，30% 的 NLI 用裸标签，选项顺序不变），每条的写法由 id 固定；我们的模型和基线都评，id 带 `#laya` 后缀，不改动 test:* 本身。同一批题目的纯文本版本在 test:* 里，报告给出同一模型「Laya 式 − 纯文本」的配对差。它衡量没见过的内容、训练里见过的写法；`VIEWS_VERSION` 记在 `meta.laya_test`，每行的 `render` 是改写结果的哈希。
- `meta.eval_set`：语料和探针文件的 sha256、抽样种子与上限、每个切片的指纹（排序后 (id, 答案) 的 sha256）。两次运行指纹相同，评的就是同一批题目。
- `meta.option_fit`：题目的选项放进 `head_max_len` 时被截断、或截断后完全相同的情况（见下）。
- 设了 `eval.compare_preds`（例如 v0 的 `eval_preds.jsonl`）时，加上与它逐题配对的比较。

**所有区间只反映抽到哪些题目的随机性，不反映训练随机种子的影响**：每个版本只训练了一次，看不出重训一次数字会变多少。

两次运行逐题比较（不需要模型和 GPU）：

```bash
uv run zhjudge compare runs/mmbert-base-zh-v1/eval_preds.jsonl v0/eval_preds.jsonl --label-a v1 --label-b v0 --out compare.md
```

按 id 配对，每个切片给出双方准确率、配对差值和 95% 区间、McNemar p 和两边的指纹。某个切片两边的题目不同（只有一方有的 id，或同一 id 的答案或改写不同）时不比较，命令以退出码 1 结束；加 `--allow-diff` 才在共同的题目上比较。v0 的 `eval_preds.jsonl` 没有存切片，按 id 推出（typed-decisions 的工作流不在 id 里，按另一方存的切片归类）。`--t1` 先用各自旁边 `eval.json` 里的温度把概率还原到 T=1（或用 `--temps-a` / `--temps-b` 指定）。

**选项截断（只报告，不改输入）**：`laya.common.build_sequence` 每个选项最多保留 48 个 token；选项总长超过 `head_max_len`（256）时，每个选项只保留 [MASK] + max(4, (head_max_len − 16) // K) − 1 个 token，题目文字截到 max(8, 剩余) 个 token。banking77 有 77 个选项，按每个词至少一个 token 的下限算：25 个意图被截断，7 个截断后完全相同（如 lost or stolen card / lost or stolen phone），题目只剩 8 个 token（真实分词器下题目的 token 更多，v1 的 Laya 式题目末尾点名字段的部分很可能被截掉）。`zhjudge train`（日志和 `train_stats.json` 的 `option_fit`，含训练、dev 和校准样本）和 `zhjudge eval`（`eval.json` 的 `meta.option_fit`）按来源记录实际分词器下的数字；`uv run python scripts/option_truncation.py <分词器目录>` 可以用任意导出的 `tokenizer/` 单独算（不需要 GPU；`--corpus` 统计评测题目，`--lower-bound` 不用分词器）。v1 不改 `head_max_len`（改了 banking77 / MASSIVE 的输入就与 v0 不同）；v1 的实测里 banking77 的 2,000 道测试题全部被截断，所以 v1b 把 `head_max_len` 调到 512，之后训练、dev、校准样本全部完整，测试集里只剩 Global-MMLU 的 10 道题有截断。

**常见问题**：显存不足 → 调小 `train.max_tokens`；T4 上出现连续 NaN → `ZHJUDGE_PRECISION=fp32`；`torch.cuda.is_available()` 为 False → `TORCH_VARIANT=cu126`。

## 项目结构

```text
configs/datasets.yaml      训练数据白名单（已按许可审计打开 24 个来源）
configs/licenses.yaml      许可审计（机器可读；build / validate 据此复核）
configs/train_base.yaml    正式训练：mmBERT-base（v1）
configs/train_v1b.yaml     v1b：v1 + 数据与训练配方的改动（train_base.yaml 的完整副本）
configs/train_smoke.yaml   冒烟测试：mmBERT-small，20 步
configs/eval_reference_v0.json  v0 各来源 dev/test 条数（validate 据此检查评测集是否与 v0 相同）
scripts/run_all.sh         一条命令跑完整流程（CONFIG=... 可换配置）
scripts/smoke.sh           CPU 冒烟测试
scripts/option_truncation.py  选项截断报告：用任意分词器统计哪些选项被截断或截成相同（不需要 GPU）
scripts/coreml_compat.py   可选，仅 Apple：laya-coreml 兼容性检查（独立环境）
notebooks/colab_train.ipynb  Colab 训练笔记本
notebooks/kaggle_train.ipynb Kaggle 训练笔记本（自动续训、可选双卡；私有仓库时从 Secrets 取令牌）
notebooks/kaggle_publish.ipynb Kaggle 发布笔记本：把训练好的模型上传到 Hugging Face
src/zhjudge/               Python 包：data/（转换器、训练时的 Laya 式写法 views、构建、校验、探针）、model、train、memprobe、calibrate、eval（stats：区间与配对差值）、compare、export、compat、publish
```

## 许可与规则

- 代码：Apache-2.0（见 [LICENSE](LICENSE)）。
- 数据：每个数据集的许可和是否启用见 [DATA_LICENSES.md](DATA_LICENSES.md)、`configs/licenses.yaml` 和 `configs/datasets.yaml`。探针集（CLUE、LCQMC、ChnSentiCorp、Amazon reviews、XNLI、AG News、GLUE、DREAM、typed-decisions）在运行时下载，只在本地用于评估和去污染，从不用于训练，也不随仓库或模型分发；其中多个是非商用或没有许可的（见 DATA_LICENSES.md 第 4 节），项目如转向商用需重新评估。
- 模型权重的许可取决于底座（mmBERT，MIT）和启用的训练数据。按许可审计的 `default` 方案，导出的模型卡默认标注 CC BY-SA 4.0，逐个列出训练和评测数据的许可与署名义务（取自 `configs/licenses.yaml`），并随附 DATA_LICENSES.md；只用非相同方式共享来源（`permissive_strict`）时才可改为 Apache-2.0（`export.license`）；许可与训练来源不符时 `zhjudge export` 会警告，并拒绝 `--push-to`。
- **绝不使用 Jev（或任何闭源决策 API）的输出做训练数据或标签。** 所有标签都来自公开数据集（人工标注、考试答案或作者提供的元数据），不来自任何模型的输出。
- 仓库里没有任何密钥；Hugging Face Token 只在运行时从环境变量读取。
- 说明：spike 用的 `laya` 0.3.7 已从 PyPI 下架，本项目固定使用最接近的 0.3.9（提示词格式与模型结构未变，兼容性检查通过）。spike 中的 MLX 训练脚本未移植，云端训练只用 PyTorch。

---

<a id="english"></a>

# zhjudge: a Chinese-first open typed-decision model

> **Independent project.** zhjudge is an independent open-source project. It is not affiliated with, endorsed by or sponsored by TypeSafe (maker of Jev) or Convai (maker of Laya). "Jev" and "Laya" belong to their respective owners and are named here only to identify the model category (Jev-style typed decisions, a "System One") and the Laya checkpoint format this project uses. This project has never run or measured Jev, makes no claim of compatibility with Jev's API, does not call it, and never trains on its outputs (the one evaluation set whose labels come from an unnamed model, typed-decisions, is used for evaluation only; see DATA_LICENSES.md).

## What it is

A small model that **judges instead of generating** (a "System One" in the Jev / Laya sense). Give it a state and a typed question with options; one forward pass returns **calibrated probabilities** per option:

```text
state:    客户说：三月的账单被重复扣款了，今天不退款我们就取消订阅。
question: Which department should handle this?   options: billing / technical / sales
output:   billing 0.86   technical 0.05   sales 0.09      (illustrative numbers)
```

Question types: `choice`, `score` (ordered levels) and `noul` (yes/no, returns P(yes)). Base encoder: [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) (MIT, multilingual). The decision head is the module from upstream [Laya](https://github.com/NandhaKishorM/laya)'s `laya` package (Apache-2.0) and checkpoints are saved in Laya's format. In this repo `zhjudge compat` has been run on the smoke-test export and passes the `laya` reference runtime (max probability difference 5e-5 vs our batched eval path, measured 2026-09-29) and Laya's `/v1/systemone` server. `laya-mlx` can only be checked on Apple silicon (the phase-0 spike passed it there; not re-run in this repo). `laya-coreml` has a check script, `scripts/coreml_compat.py`, that has not been run for this repo.

**Status: work in progress.** The pipeline runs end to end on CPU (smoke test) and on a Kaggle T4. The first full model, v0 (2026-09-29), has a much lower calibration error than the baseline on most sets and beats it on our held-out set and on four of the five never-trained Chinese sets (its Belebele lead came mostly from SIB-200 training sentences contained in Belebele passages, see 训练记录; Global-MMLU 28.7% vs 25.7% is within noise: Wilson 95% intervals 25.9-31.5% and 23.0-28.5% overlap), but not across the probe sets it targets (NLI fell to chance in Chinese and to 43.3% on XNLI en; our diagnosis: the probes write states as JSON with named fields, v0 only ever saw plain-text states), so it was not released. v1 (2026-09-30) brought XNLI zh from 33.3% back to 69.3% (level with the baseline) but is still below the baseline on XNLI en (66.7% vs 84.7%) and the EN control (63.1% vs 68.1%), so it was not released either; v1b (2026-09-30, a separate config adding MNLI 30k, GoEmotions polarity, more oasst2_zh ratings, RL group 64, per-epoch renderings, question-shape variants and head_max_len 512) is significantly above the baseline on the zh probe (55.5% vs 47.3%, paired interval above 0), ties it on XNLI en (84.7%), is 6 points higher on XNLI zh (interval includes 0), but is still below it on typed-decisions (32.6% vs 35.1%, interval includes 0). v1b was published on 2026-09-30 as a **preview** on Hugging Face: [TerryGao9/zh-decision-mmbert-v1b](https://huggingface.co/TerryGao9/zh-decision-mmbert-v1b) (CC BY-SA 4.0). See 训练记录 for the full table. v1 trains half of the records in Laya's structured question format (`src/zhjudge/data/views.py`, `train.format_aug`, `train.option_shuffle`; same sources and licences, same probe sets and held-out sample (drawn with `eval.seed: 13`, v0's `run.seed`), v0's report computed the same way, with the additions described under [Pipeline](#pipeline) after it; no training instruction equals a probe instruction, even up to the field names; `zhjudge validate` checks it). The numbers below are the baseline to beat, not results of this project.

**v1b** (`configs/train_v1b.yaml`, run `mmbert-base-zh-v1b`, trained 2026-09-30, published as a preview; results in 训练记录) is v1 plus a data and recipe bundle, kept in its own config (a full copy of `train_base.yaml` with the keys below; each is off by default, so v1's training and resume hash do not change): v1 vs v0 measures the format change, v1b vs v1 this bundle as a whole. Data (`data.train_sample`, train records only; dev/test files byte-identical): MNLI 30k instead of 12k (the 30k sample contains the 12k), 8,000 more GoEmotions train comments asked for polarity / 3-level valence from the dataset authors' own emotion grouping (ambiguous and mixed comments left out; estimated 6.8k records, measured: GoEmotions train records went from 9,959 to 16,556), and the other rated attributes (quality, helpfulness, creativity) of each oasst2_zh reply plus rated oasst2_zh prompts (an attribute whose most common level exceeds 60% is left out; estimated 10-15k Chinese score questions, measured: oasst2_zh train records went from 4,345 to 15,457; all records of one message tree share a `group`, which the calibration slice takes or leaves as a whole). The manifest records `train_sample` (per source in `split_sources`; both go into the export's `training_data.json`), `zhjudge train` refuses a corpus built with other values and the Kaggle notebook rebuilds. Recipe: `train.rl_group` 64 (v1: 4; the RL gradient grows by about 15% relative to CE), `train.format_aug_per_epoch` (a fresh rendering and option order of every training record in epoch 2; epoch 1 as in v1, calibration records excluded in every epoch) and `train.format_variants` 0.25 (a quarter of the Laya-style records in another Laya question shape: noul true/false criteria (30% also with custom yes/no labels), 5 score levels merged exactly into 3 with their own level texts, NLI as dict criteria, BoolQ as a yes/no choice; criteria are built from the record's final option order, the gold comes from the dataset label, ids never change; `VIEWS_VERSION` 4, which leaves v1's templates and test:laya renderings unchanged). These items were designed after seeing v0's probe families (NLI, sentiment / rating, Laya question shapes), so a probe gain says even less about generalisation than v1's; with the v1b config `zhjudge validate` also checks the new templates and the new description / level texts against the probes (shared generic label names are listed only). The new labels and level texts are written in this repo like v1's templates; labels always come from the datasets; no new source, no licence change. On Kaggle set `CONFIG = "configs/train_v1b.yaml"` and run it as a new notebook version without v1's output attached (attach v1b's own output to resume it). Measured on a Kaggle T4 (2026-09-30): 253k training records (v1: 217k), 3.2 h of training (v1: 2.6 h), 3.49 h for the whole run.

**Success criteria (registered before v1's and v1b's eval results were seen).** Primary: the item-paired accuracy difference vs v0 and vs laya-multilingual re-measured on the same items, with a 95% interval excluding 0, on the zh probe (600), XNLI zh (150), the English control (360) and typed-decisions (2,000), each set judged on its own (better when the interval lies entirely above 0, worse when entirely below, otherwise no detectable difference); the never-trained eval-only sources (`test:group:eval-only`) are the least targeted yardstick. v1b vs v1 the same way (`zhjudge compare` on both `eval_preds.jsonl`). Family-level changes (70-150 items per family) and ECE changes are exploratory. The intervals cover item sampling only: each version is trained once, so a small v1 vs v1b difference may not replicate. Choose between v1 and v1b on non-probe measures, test:* and test:laya (the same items in both runs; `zhjudge compare` checks this), and report the probe results of both; the probes are not used to pick a model. The calibration-slice monitor is not a comparison: the two runs hold out almost entirely different calibration records, v1b's with format variants (e.g. 3-level merged score questions), so it is a within-run training diagnostic only.

## Baseline to beat (measured in the phase-0 spike)

laya-multilingual (`convaiinnovations/laya@5e7b2b1`, `multilingual`), measured by the spike (the phase-0 spike is this project's unpublished prototype; its files are not public, only its measurements are quoted here, and `zhjudge probes` rebuilds its probe sets byte for byte; source: the spike's `baselines/results/summary.md`, 2026-09-24) with the official `laya` 0.3.7 PyTorch runtime on Apple M5 Pro MPS, fp32:

| set | n | accuracy | Brier ↓ | ECE-15 ↓ |
|---|---|---|---|---|
| Chinese probe, Chinese prompt | 600 | **47.3%** | 0.813 | 0.341 |
| Chinese probe, English prompt | 600 | 50.2% | 0.792 | 0.334 |
| English control (same task families) | 360 | **68.1%** | 0.490 | 0.190 |
| XNLI parallel, zh | 150 | **70.0%** | 0.464 | 0.166 |
| XNLI parallel, en | 150 | **84.7%** | 0.254 | 0.116 |
| typed-decisions test | 2,000 | 35.2% | 0.890 | 0.314 |

`zhjudge eval` prints our model next to these numbers and (by default) re-measures laya-multilingual on the same records; this repo's CPU re-measurement (measured 2026-09-29, `laya` 0.3.9, fp32) matches accuracy, Brier and ECE-15 in the table to 4 decimals. Re-measured on a Kaggle T4 a few items come out differently (XNLI zh 69.3% in all three runs; typed-decisions 35.1% in the v1 and v1b runs and 35.2%, as in the table, in v0's; 1-2 items each), probably CUDA numerics (the table was measured on an Apple GPU and the CPU re-measurement matches it); every Δ in 训练记录 uses the re-measurement from the same run.

## Train in the cloud

No data or weights are committed; everything is downloaded from Hugging Face at runtime.

**Training-data allowlist**: `configs/datasets.yaml` enables the 24 sources the licence audit (`configs/licenses.yaml`, human-readable [DATA_LICENSES.md](DATA_LICENSES.md)) rates ALLOW or CONDITIONAL and that have a converter (5 of them eval-only). The three `nli26_zh_*` Chinese NLI translations carry no licence and stay disabled. `build` and `validate` re-check every source against the audit and refuse one it does not allow, even if enabled. The default profile is `default` (includes 7 share-alike sources; weights CC BY-SA 4.0); for Apache-2.0 weights pass `--set data.license_profile=permissive_strict --set export.license=apache-2.0`.

- **Colab / Kaggle (free T4 16 GB)**: open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) (Colab: <https://colab.research.google.com/github/TerryG907/zhjudge/blob/main/notebooks/colab_train.ipynb>), pick a T4 GPU, run top to bottom. Keep `runs/` on Google Drive (`USE_DRIVE`) so a dropped Colab session resumes after you rerun all cells (the resumable checkpoint for mmBERT-base is ~4 GB of Drive space by estimate — fp32 weights plus AdamW state — deleted after a successful run). The repo is public, so no GitHub token is needed; only for a private copy (e.g. made with GitHub's Import repository; a fork of a public repo cannot be private), set `REPO_URL` to it, tick "Include private repos" when opening the notebook from GitHub in Colab, set `PRIVATE_REPO = True` and paste a read-only fine-grained GitHub token when asked (kept in memory, never written to `.git/config` or any file).
- **Kaggle (T4 x2)**: use [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb); click-by-click steps are in the Chinese section [在 Kaggle 上训练](#在-kaggle-上训练). Set Accelerator = GPU T4 x2 and Internet on (no GitHub token is needed for this public repo; a private copy needs `REPO` set to it and a read-only fine-grained token as the Secret `GITHUB_TOKEN`), then Save Version → Save & Run All. It runs the whole pipeline with fp16 + GradScaler on one T4 (optionally both, via torchrun, after an NCCL self-check), writes resumable checkpoints to `/kaggle/working`, pauses cleanly before Kaggle's 12-hour limit, resumes automatically from the most advanced checkpoint trained with the same config when the previous version's output is attached as an input, and packages the export, reports and logs into `/kaggle/working/outputs`. A GitHub token, when used, is only read from Kaggle Secrets and kept in memory. For v1b set `CONFIG = "configs/train_v1b.yaml"` in the settings cell and save it as a new version without v1's output attached.
- **AutoDL / RunPod / any GPU box**:
  ```bash
  git clone https://github.com/TerryG907/zhjudge && cd zhjudge
  pip install uv
  CONFIG=configs/train_base.yaml bash scripts/run_all.sh
  ```
  The https clone works as is (public repo); for a private copy, clone over SSH with a read-only deploy key rather than putting a token in the URL. Rerun the same command to resume. `run_all.sh` picks the torch build (PyPI CUDA 13.0 needs driver >= 580 and compute capability >= 7.5; otherwise CUDA 12.6 via `TORCH_VARIANT=cu126`; CPU build without a GPU) and precision (bf16 on Ampere+, fp16 + GradScaler on T4/V100, fp32 on CPU; gradient checkpointing on GPUs < 20 GiB).
- **Time on a T4 (measured on Kaggle, 2026-09-29, v0).** 217,887 training records (19 sources, 30k cap, 4k calibration slice held out), 128 tokens on average; about 43 examples/s, **1.4 h per epoch**, 2.8 h of training for the default 2 epochs (fp16 + GradScaler, gradient checkpointing, peak 6.0 GiB, no non-finite losses); 3.1 h for the whole run including data build, calibration, evaluation and the compat check. The log shows a live `ETA`; `hours_per_epoch_estimate` in `runs/<name>/train_stats.json` is computed from the measured throughput. v1 and v1b (measured on a Kaggle T4, 2026-09-30): v1 51 examples/s, 2.6 h of training, 2.98 h for the run, peak 4.9 GiB; v1b 253k training records, 49 examples/s, 3.2 h of training, 3.49 h for the run. For reference only (spike measurement, not a T4): MLX bf16 training of mmBERT-base on an Apple M5 Pro ran at 59 examples/s on an idle machine (5,000-record sample, 149 tokens on average) and 15 examples/s while other jobs were running.
- Machines without a GPU (including most cloud dev environments): smoke test only.

Publishing: `notebooks/kaggle_publish.ipynb` uploads a finished Kaggle run's `outputs/model/<name>/` to your Hugging Face account with `zhjudge publish` (model card regenerated with a status table of paired verdicts vs laya-multilingual; private repo first, make it public after checking). Store a Hugging Face *write* token as the Kaggle Secret `HF_TOKEN`; never paste it anywhere else. The v1b preview is at [TerryGao9/zh-decision-mmbert-v1b](https://huggingface.co/TerryGao9/zh-decision-mmbert-v1b); load it with `laya.load("TerryGao9/zh-decision-mmbert-v1b")` (`laya==0.3.9`, the version `zhjudge compat` checks; example in the Chinese section 下载与使用（预览版）). It is a preview, not a final release: significantly better than laya-multilingual on the zh probe, slightly lower on typed-decisions (no clear difference); the per-set verdicts are in the status table at the top of the model card. Probabilities are calibrated on the training distribution, so spot-check calibration on your own data before relying on confidence.

## Local smoke test (CPU, about a minute)

```bash
nice -n 15 bash scripts/smoke.sh
```

`jhu-clsp/mmBERT-small`, ~200 examples, 20 steps, the whole pipeline on CPU (forced even when a GPU/MPS exists). Measured on 2026-09-29 from a fresh clone on an Apple M5 Pro with dependencies and models cached and a fresh `.venv`: about 60 s wall (build 14.3 s, validate 1.5 s, train 33.2 s, calibrate 2.0 s, eval 3.8 s, export 0.1 s, compat 4.1 s). The first run downloads ~0.64 GB (measured with an empty cache: mmBERT-small 564 MB, three datasets 73 MB, all ALLOW-class in the licence audit). Smoke metrics are meaningless and the export's model card says so.

## Pipeline

`scripts/run_all.sh` = `uv sync`, then `zhjudge all`: `probes` (rebuild the baseline probe sets, byte-identical to the spike) → `build` (allowlisted download, convert, dedupe, decontaminate, deterministic splits, sha256 manifest) → `validate` (schema, leakage, probe overlap, allowlist) → `train` (resumable, periodic dev eval) → `calibrate` (temperatures on a held-out slice) → `eval` (accuracy, Brier, ECE-15 per language / task / type vs laya-multilingual) → `export` (Laya checkpoint + model card) → `compat` (loads the export in Laya's reference runtime and checks probability parity). Run a single step with `uv run zhjudge <step> --config configs/train_base.yaml` from the repo directory. Every command takes `--config` and `--set key=value`. Not in `all`: `zhjudge compare` (below) and `zhjudge memprobe` (the GPU peak of a training step without gradient checkpointing, the AdamW state on top and whether it would fit, in `runs/<name>/memprobe.json`; report only; with `train.memprobe: true` (as in `train_base.yaml`) `train` runs it by itself on CUDA in a separate process before building its model, so an OOM there never reaches training, and a failure or its 15-minute timeout is only logged; the result is also in `train_stats.json`; without CUDA it prints `skipped (no CUDA)`). Troubleshooting: CUDA OOM → lower `train.max_tokens`; repeated NaN on a T4 → `ZHJUDGE_PRECISION=fp32`; no CUDA → `TORCH_VARIANT=cu126`.

**Training monitors (recorded only; they never change what is trained).** Every dev evaluation in `dev_log.jsonl` also has each source (`source:<name>`) and the mean over sources (`macro_src`). Dev covers 10 of the 19 sources and has no score or NLI questions, so `train.monitor_calib_per_source` (120 in `train_base.yaml`, default 0) also scores up to that many held-out calibration records per training source (all question types, about half written Laya-style) at update 0, with every dev evaluation and at the end, per source, question type and rendering (`fmt:laya` / `fmt:plain`), as `calib_slice` in the same row (the last one also as `calib_slice_final` in `train_stats.json`). Every `train_log.jsonl` row has the batch's question types (`qtypes`), and every `log_every` steps the gradient norm of the RL, CE and act terms at the head's outputs (`glogit_rl` / `glogit_ce` / `glogit_act`, also on the log line). Monitors run on rank 0 without collectives; one that raises is logged once and switched off (`monitors_off`), and training goes on. Operational settings that leave the recipe and the resume hash unchanged: `train.optimizer_impl` (auto = fused AdamW on CUDA; v0 used foreach; a resumed run keeps its checkpoint's kernel) and `train.pad_multiple` (16 in `train_base.yaml`, v0 32: training batches only, same batch plan and logits; only the decision head's dropout draws follow the padded shape).

**Evaluation report and comparing runs.** `validate` also fails when an id repeats among the decisions `eval` will score, and warns (`eval_set_vs_v0` in `validation.json`, also in the Kaggle `STATUS.md`) when a source's dev/test counts differ from v0's (`configs/eval_reference_v0.json`, from v0's build log) or a source dropped dev/test records for overlapping the probes: then its test:* items are not v0's. `eval` first writes v0's report unchanged (`eval.json` `model` / `baseline_measured` / `meta`, the first part of `eval.md`, `eval_preds.jsonl`, whose rows now also carry their `slices`), then appends (and adds `eval_set`, `option_fit`, `laya_test` to `meta`), each part on its own (a failure is logged in `report_errors` and never changes the report above or the exit code): Wilson 95% intervals (decisions treated as independent, so too narrow for typed-decisions, whose paired differences resample by case) and the ECE-15 floor of every slice; paired differences vs laya-multilingual re-measured on the same items (accuracy, Brier, NLL with 95% intervals, typed-decisions resampled by case; exact McNemar p); family / question-type tables for all six probe sets; macro averages (normal-approximation interval); `test:group:{trained,eval-only}`; Brier / ECE-15 / NLL at temperature 1 (`uncalibrated`: v0's temperatures were fitted to the smoothed targets, v1's to the gold label, so compare calibration at T=1); `test:laya:*` (`eval.laya_test: 500` in `train_base.yaml`: up to 500 test records per source written Laya-style with a fixed rendering per item, ids `<id>#laya`, scored for both models, plus the same model's Laya-minus-plain paired difference); `meta.eval_set` (corpus and probe sha256, sample seed, a fingerprint of every slice); `meta.option_fit`; and with `eval.compare_preds` (e.g. v0's `eval_preds.jsonl`; the Kaggle notebook's `COMPARE_PREDS`) an item-by-item comparison with that run. `calibration.json` adds `calib_crossfit` (fit on half the slice, score the other half). The intervals cover item sampling only, not training-seed variance: each version was trained once. `uv run zhjudge compare A/eval_preds.jsonl B/eval_preds.jsonl --label-a v1 --label-b v0 [--out f.md] [--allow-diff] [--t1]` compares two runs without a model: per slice both accuracies, the paired difference with its interval, McNemar p, ECE-15 and both fingerprints; a slice whose items differ is not compared (exit 1) unless `--allow-diff`; v0's rows get their slices from their ids; `--t1` undoes each run's temperatures (from the `eval.json` next to it) first.

**Option truncation (report only).** `laya.common.build_sequence` keeps at most 48 tokens per option and, once the options overflow `head_max_len` (256), [MASK] + max(4, (head_max_len − 16) // K) − 1 tokens of each, and cuts the instruction to max(8, what is left) tokens. At a lower bound of one token per word, 25 of banking77's 77 intents are cut, 7 end up identical (e.g. lost or stolen card / phone) and the instruction keeps 8 tokens (real subword tokens make the instruction longer, so the field reference at the end of a v1 Laya-style instruction is likely cut). `train` (log and `train_stats.json` `option_fit`, for train, dev and calibration items) and `eval` (`meta.option_fit`) record the real counts per source; `uv run python scripts/option_truncation.py <tokenizer dir>` computes them for any exported `tokenizer/` without a GPU (`--corpus` for the eval records, `--lower-bound` without a tokenizer). v1 keeps `head_max_len` (raising it changes banking77 / MASSIVE inputs relative to v0); v1's measurement showed all 2,000 banking77 test items cut, so v1b raises `head_max_len` to 512, after which every train, dev and calibration item fits and only 10 Global-MMLU test items are still cut.

## Layout

See the tree in the Chinese section: `configs/` (allowlist, licence audit, base (v1), v1b and smoke configs, v0's eval-set counts), `scripts/` (run_all, smoke, option truncation report, optional Apple-only Core ML check), `notebooks/colab_train.ipynb` (Colab), `notebooks/kaggle_train.ipynb` (Kaggle), `notebooks/kaggle_publish.ipynb` (upload to Hugging Face), `src/zhjudge/` (data converters/build/validate/probes, model, train, memprobe, calibrate, eval with stats (intervals, paired differences), compare, export, compat, publish).

## Licences and rules

- Code: Apache-2.0 ([LICENSE](LICENSE)). Data: per-dataset licences and decisions in [DATA_LICENSES.md](DATA_LICENSES.md), `configs/licenses.yaml` and `configs/datasets.yaml`. Probe sets (CLUE, LCQMC, ChnSentiCorp, Amazon reviews, XNLI, AG News, GLUE, DREAM, typed-decisions) are downloaded at run time and are evaluation-only: used locally, never trained on or redistributed; several are non-commercial or unlicensed (DATA_LICENSES.md section 4), so re-assess if the project becomes commercial. Weight licensing depends on the base (mmBERT, MIT) and the enabled data: following the audit's `default` profile the exported model card says CC BY-SA 4.0, lists every trained and evaluated source's licence and attribution duties (from `configs/licenses.yaml`) and ships DATA_LICENSES.md; switch `export.license` to apache-2.0 only for the `permissive_strict` profile (no share-alike sources); `zhjudge export` warns, and refuses `--push-to`, when the licence does not match the trained sources.
- **Never train on outputs of Jev or any other proprietary decision API.** All labels come from public datasets (human annotations, exam answer keys or author-supplied metadata), not from a model.
- No secrets in the repo; a Hugging Face token is only read from the environment at runtime.
- The spike used `laya` 0.3.7, which has since been removed from PyPI; this repo pins the nearest release, 0.3.9 (same prompt format and architecture; compat check passes). The spike's MLX trainer is not ported; cloud training is PyTorch only.
