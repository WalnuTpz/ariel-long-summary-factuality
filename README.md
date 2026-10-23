# ARIEL: Adaptive Risk-Informed Evidence Localization for Long-Document Summary Factuality

[方法与评估](docs/method.md) | [实验指南](docs/reproduction.md) | [数据与结果格式](data/README.md)

ARIEL 面向英文长文档摘要的事实一致性评测与局部纠错。给定源文档和生成摘要，它为每个摘要句定位证据，用自然语言推断（NLI）判断证据是否支持该句，再依据证据改写选中的风险句。

项目以 GovReport 长报告为实验对象，研究**证据选择如何影响摘要事实性判断与纠错**。仓库提供摘要生成、证据检索、逐句验证、局部纠错和评估的完整流程，以及检索基线、方法消融和受控摘要长度实验。

## 方法概览

长报告中的事实依据常分散在不同段落。与摘要主题相关的片段，可能仍缺少验证某个数字、实体或关系所需的信息。ARIEL 在同一源文档内结合词面匹配、语义检索和事实要素覆盖情况选择证据：

- **风险感知融合**：分别用 BM25 和 Dense 检索召回候选，将两路分数归一化；根据摘要句中的数字、日期、实体和关系线索，调整词面与语义分数的融合权重。
- **覆盖率重排序**：计算候选证据对查询句关键词、数字日期和实体的覆盖率，为包含关键事实要素的候选加分。
- **相邻上下文扩展**：覆盖不足时，加入候选块前后的文档块，重新计算覆盖得分，补充相邻段落中的主体和限定条件。

![ARIEL：从源文档检索证据，逐句验证并局部纠错](docs/architecture.png)

默认保留 Top-3 证据，由 NLI 将摘要句分为 Supported、Unsupported 或 Not Enough Information（NEI）。纠错模型对 Unsupported 句及 neutral 分数达到 0.7 的 NEI 句生成局部改写，随后重新检索和验证。前后句子保留相同标识，以便比较证据、判断和指标的变化。

## 安装

运行环境为 Linux 或 WSL、Python 3.10+ 和 Bash。以下示例使用 CUDA；CPU 运行时，将命令中的 `SUMMARY_DEVICE`、`EMBEDDING_DEVICE` 和 `NLI_DEVICE` 改为 `cpu`，纠错设备默认跟随摘要设备。所有命令从仓库根目录执行。

```bash
git clone https://github.com/WalnuTpz/ariel-long-summary-factuality.git
cd ariel-long-summary-factuality
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

实验直接使用以下预训练模型进行推理，首次运行时从 Hugging Face 下载至本地缓存。

| 用途 | 默认模型 |
|---|---|
| 摘要生成与局部纠错 | `Qwen/Qwen2.5-1.5B-Instruct` |
| Dense 证据检索 | `sentence-transformers/all-MiniLM-L6-v2` |
| 句子级 NLI | `facebook/bart-large-mnli` |

## 快速开始

下面用五篇报告完成摘要生成、事实性验证和局部纠错，并查看纠错前后的评估结果。

### 1. 准备报告

从 GovReport validation split 按数据集顺序取前五篇有效报告。下载器保留正文和参考摘要均非空、正文至少 2,000 字符的记录；规范化后，每行包含一篇报告的 `doc_id`、`document` 和 `reference_summary`。

```bash
python scripts/download_govreport_sample.py \
  --split validation --limit 5 --min-document-chars 2000 \
  --output data/raw/govreport_quickstart_5.jsonl

python -m src.load_data \
  --input data/raw/govreport_quickstart_5.jsonl \
  --output data/processed/documents_quickstart_5.jsonl \
  --min-document-chars 2000
```

### 2. 运行完整流程

```bash
CONFIG= PYTHON=.venv/bin/python \
  DOCUMENTS=data/processed/documents_quickstart_5.jsonl \
  RUN_DIR=data/results/quickstart_5 \
  SUMMARY_FIELD=generated_summary RETRIEVAL_METHOD=ariel \
  RUN_SUMMARIZE=1 RUN_CORRECTION=1 RUN_AFTER_VERIFY=1 RUN_CASE_EXPORT=1 \
  SUMMARY_DEVICE=cuda EMBEDDING_DEVICE=cuda NLI_DEVICE=cuda \
  bash scripts/run_pipeline.sh
```

`CONFIG=` 让脚本直接使用命令中的环境变量。生成阶段按已有输出中的 `doc_id` 续跑；更换模型或摘要长度时，使用新的 `RUN_DIR` 保存一组新摘要。

### 3. 查看输出

```bash
python -m json.tool data/results/quickstart_5/evaluation.ariel.summary.json
```

结果保存在 `data/results/quickstart_5/`。其中，`generated_summaries.jsonl` 是初始摘要，`corrections.ariel.jsonl` 包含纠错摘要与逐句编辑，`cases.ariel.md` 展示选出的风险句、检索证据及纠错结果。

汇总文件 `evaluation.ariel.summary.json` 的 `by_version` 分别记录纠错前后的指标，`paired_error_reduction_rate` 记录配对摘要的风险句数下降幅度。指标定义见下文“实验与评估”。

## 实验与评估

### 检索对比与消融

完成快速开始后，可复用刚生成的五篇摘要比较检索方法：

```bash
CONFIG= PYTHON=.venv/bin/python \
  DOCUMENTS=data/processed/documents_quickstart_5.jsonl \
  SUMMARIES=data/results/quickstart_5/generated_summaries.jsonl \
  SUMMARY_FIELD=generated_summary \
  RUN_DIR=data/results/quickstart_5_compare \
  EMBEDDING_DEVICE=cuda NLI_DEVICE=cuda \
  bash scripts/run_compare_retrieval.sh
```

默认运行 BM25、Dense、Fixed Hybrid、ARIEL，以及三组 ARIEL 消融。Fixed Hybrid 对两路归一化分数按 0.5 / 0.5 融合；三组消融分别取消覆盖率排序加分、相邻上下文扩展，或将风险相关的融合权重统一为 0.5。覆盖率消融仍保留扩展判断，风险权重消融仍保留覆盖率排序与上下文扩展。

各组读取相同的文档和生成摘要，共用 Top-3 证据数量、BART-MNLI 验证器及 NLI 决策规则。结果写入 `data/results/quickstart_5_compare/`：`comparison.md` 汇总各组指标，逐句证据与 NLI 判断分别保存在 `retrieval.<method>.jsonl` 和 `verification.<method>.jsonl`。

### 主实验设置

主实验固定 GovReport validation split 中按顺序选出的 600 篇有效报告，以及每篇报告的一条生成摘要。摘要提示的目标长度为 240 词；检索对比、消融和局部纠错均复用这组摘要。受控长度实验取其中前 100 篇报告，为每篇分别生成三种目标长度的摘要。

| 实验 | 比较内容 | 配置 |
|---|---|---|
| 检索与消融 | 四种检索方法及三组 ARIEL 消融 | `configs/main_600_compare_gpu.env` |
| 局部纠错 | ARIEL 纠错前后的句子判断、风险句数与摘要长度 | `configs/main_600_pipeline_gpu.env` |
| 受控摘要长度 | 80、240、520 词三种生成目标下的实际长度与事实性指标 | `configs/controlled_length_100_gpu.env` |

主实验的检索与纠错结果由 `scripts/make_tables.sh` 配合 `configs/main_600_tables_gpu.env` 汇总为 `outputs/tables/main_600_gpu/paper_tables.md`。受控长度脚本自动生成 `outputs/tables/controlled_length_100_wide_gpu/controlled_length.md`，报告各组实际平均词数和验证指标。

### 指标含义

- **ROUGE-1/2/L**：生成摘要与参考摘要的 unigram、bigram 和最长公共子序列重叠 F1。
- **Supported Ratio**：NLI 判为 Supported 的句子比例。
- **Risk Ratio**：NLI 判为 Unsupported 或 NEI 的句子比例。
- **Error Reduction Rate**：对纠错前存在风险句的配对摘要，计算 `(纠错前风险句数 − 纠错后风险句数) / 纠错前风险句数`，再取平均。

Supported Ratio 与 Risk Ratio 先在每篇摘要内计算，再对摘要取平均。这些事实性指标描述 NLI 模型基于检索证据作出的判断；案例文件保留原句、证据、改写与前后标签，可据此分析具体变化。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

回归检查使用 Python 标准库、小型合成 JSONL 和模型调用替身，验证数据关联、检索融合、纠错、指标及脚本编排。运行过程使用本地代码与合成输入。

## 致谢

项目源于南京大学自然语言处理课程的长文摘要事实一致性任务，保留了[原始课程要求](docs/长文摘要事实一致性评测与纠错.md)。课程项目基础来自 [NJU_NLP_Project_2026](https://github.com/zifengcheng/NJU_NLP_Project_2026/)。实验使用 [GovReport](https://huggingface.co/datasets/ccdv/govreport-summarization) 数据，以及 Qwen2.5、Sentence Transformers 和 BART-MNLI 的公开模型。
