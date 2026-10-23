# 实验指南

本指南从环境和数据准备开始，运行 GovReport 检索对比、ARIEL 消融、局部纠错及受控长度实验，并生成用于分析的表格。主实验固定 600 篇文档与每篇一条生成摘要；各检索方法使用相同输入和 NLI 设置。受控长度实验另取其中前 100 篇，对同一文档生成三种目标长度的摘要。

## 安装与模型

运行环境为 Linux 或 WSL、Python 3.10+ 和 Bash。以下主实验命令采用 CUDA；仓库也提供 Dense 检索与 NLI 在 CPU 上运行的配置。所有命令从仓库根目录执行。

```bash
git clone https://github.com/WalnuTpz/ariel-long-summary-factuality.git
cd ariel-long-summary-factuality
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 安装 `torch`、`transformers`、`sentence-transformers` 和 `datasets`。模型分工如下：

| 用途 | 默认模型 | 运行设备设置 |
|---|---|---|
| 摘要生成 | `Qwen/Qwen2.5-1.5B-Instruct` | `SUMMARY_DEVICE` |
| Dense 证据检索 | `sentence-transformers/all-MiniLM-L6-v2` | `EMBEDDING_DEVICE` |
| 句子级 NLI | `facebook/bart-large-mnli` | `NLI_DEVICE` |
| 局部纠错 | `Qwen/Qwen2.5-1.5B-Instruct` | `CORRECTION_DEVICE` |

`*_gpu.env` 配置将模型设备设为 `cuda`，Python 入口设为 `.venv/bin/python`。可查看当前 PyTorch 的版本与 CUDA 设备可用状态：

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
```

模型默认按 Hugging Face 名称加载，首次运行时下载到本地缓存。已有本地模型目录也可以作为模型参数。离线使用模型缓存时，将对应配置中的 `HF_LOCAL_FILES_ONLY`、`EMBEDDING_LOCAL_FILES_ONLY`、`NLI_LOCAL_FILES_ONLY`、`CORRECTION_LOCAL_FILES_ONLY` 设为 `1`；GovReport 数据在后面的准备步骤单独获取。

## 准备并固定主实验输入

### 获取 600 篇报告

```bash
python scripts/download_govreport_sample.py \
  --split validation --limit 600 --min-document-chars 2000 \
  --output data/raw/govreport_main_600.jsonl

python -m src.load_data \
  --input data/raw/govreport_main_600.jsonl \
  --output data/processed/documents_main_600.jsonl \
  --min-document-chars 2000
```

下载器流式读取 `ccdv/govreport-summarization`，按数据集顺序保留正文和参考摘要均非空、正文达到 2,000 字符的记录。`--limit` 指保留的文档数。规范化输出保留 `doc_id`、正文、参考摘要及原始 `meta.source_index`；过滤后原始序号可以出现间隔。

### 生成固定摘要

```bash
python -m src.summarize \
  --input data/processed/documents_main_600.jsonl \
  --output data/processed/generated_summaries.qwen2p5_1p5b.main_600.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype auto \
  --target-words 240 \
  --chunk-input-tokens 6000 --chunk-overlap-tokens 256 \
  --chunk-max-new-tokens 256 --final-max-new-tokens 384 \
  --resume --continue-on-error
```

模型先摘要长文档的各个 token 窗口，再合并中间摘要。单窗口文档直接生成最终摘要。默认采用贪心解码；240 词是提示中的目标，实际长度记录在 `summary_word_count` 中。`dtype=auto` 在 CUDA 上使用 float16。

`--resume` 按输出文件中已有的文档 ID 跳过已完成记录。`--continue-on-error` 记录失败文档并继续，失败记录计入 `failed_documents`；再次执行同一命令时，会处理仍未写入输出的文档。结束后核对输入和生成记录数：

```bash
wc -l data/processed/documents_main_600.jsonl \
  data/processed/generated_summaries.qwen2p5_1p5b.main_600.jsonl
```

这两份文件共同定义主实验样本。后续检索对比、消融与局部纠错均读取它们；主流水线配置中 `RUN_SUMMARIZE=0` 保持这组生成摘要固定。

## 比较检索方法与消融

```bash
CONFIG=configs/main_600_compare_gpu.env bash scripts/run_compare_retrieval.sh
```

脚本切分同一批文档与摘要，按 `METHODS` 指定的方法运行共享召回、证据融合、NLI 和评估。默认运行下列七组：

| 方法名 | 本组证据定位方式 |
|---|---|
| `bm25` | BM25 词面检索 |
| `embedding` | Dense 语义检索 |
| `hybrid` | 两路 min-max 归一化后按 0.5 / 0.5 融合 |
| `ariel` | 风险权重融合、覆盖率重排序及相邻扩展 |
| `ariel_no_coverage` | 排序覆盖率加分为 0，扩展仍使用覆盖率判断 |
| `ariel_no_expansion` | 使用原证据块，保留风险权重和覆盖率排序 |
| `ariel_no_risk_weighting` | 两路融合权重统一为 0.5，保留其余机制 |

默认文档按段落切块，长段再分窗，`CHUNK_MAX_TOKENS=256`；两路召回和最终保留证据均取 Top-3。NLI 把证据作为前提、摘要句作为待判断命题，逐块预测蕴含、矛盾或中立关系，最大输入长度为 1,024 个模型 tokens。最高分标签为蕴含且分数至少 0.5 的证据使句子成为 Supported；没有这类证据时，再用相同阈值寻找矛盾证据并标为 Unsupported；其余情况归为 Not Enough Information。

结果写入 `data/results/main_600_compare_gpu/`。直接阅读 `comparison.md` 可比较各方法的摘要数量、ROUGE、Supported Ratio、Risk Ratio 和平均风险句数；`comparison.tsv` 便于后续分析。每组同时保留 `retrieval.<method>.jsonl`、`verification.<method>.jsonl`、`evaluation.<method>.jsonl` 和 `evaluation.<method>.summary.json`，分别保存证据、逐句判断、逐摘要指标和汇总指标。

## 局部纠错与前后比较

```bash
CONFIG=configs/main_600_pipeline_gpu.env bash scripts/run_pipeline.sh
```

配置沿用固定生成摘要，使用 ARIEL 检索证据。所有 Unsupported 句与 neutral 分数达到 0.7 的 NEI 句进入改写。纠错模型从最多三个证据块中选择最多五条证据句，生成一个新句子；提示最多输入 4,096 tokens，新输出最多 128 tokens。程序按原句位置重建摘要，并重新检索和验证。

结果写入 `data/results/main_600_ariel_gpu/`：

| 阅读任务 | 输出 |
|---|---|
| 查看原句和新句使用的证据 | `retrieval.before.ariel.jsonl`、`retrieval.after.ariel.jsonl` |
| 比较同一句子的 NLI 判断 | `verification.before.ariel.jsonl`、`verification.after.ariel.jsonl` |
| 阅读完整纠错摘要和编辑原因 | `corrections.ariel.jsonl` |
| 分析每篇摘要的指标变化 | `evaluation.ariel.jsonl` |
| 查看总体和长度分桶指标 | `evaluation.ariel.summary.json` |
| 阅读选出的具体案例 | `cases.ariel.md`、`cases.ariel.jsonl` |

前后句子通过同一个 `query_id` 关联。成功纠错案例同时满足：原句属于风险句、编辑动作是 `replace`、新标签为 Supported。案例导出按类型和优先级选取，默认最多十条。

## 汇总结果表

完成检索对比和纠错两组运行后执行：

```bash
CONFIG=configs/main_600_tables_gpu.env bash scripts/make_tables.sh
```

脚本从 `data/results/main_600_compare_gpu/` 读取检索与消融结果，从 `data/results/main_600_ariel_gpu/` 读取纠错结果，输出到 `outputs/tables/main_600_gpu/`：

| 表格内容 | 文件 |
|---|---|
| 主方法总体指标与文档统计 | `table_1_main_metrics.md` |
| 自然摘要长度分桶 | `table_2_length_buckets.md` |
| 四种检索方法比较 | `table_3_retrieval_comparison.md` |
| ARIEL 三组消融 | `table_4_ariel_ablation.md` |
| 局部纠错前后比较 | `table_5_correction.md` |
| 五张表的合集 | `paper_tables.md` |

Supported Ratio 与 Risk Ratio 先在每篇摘要内计算，再对摘要取平均。汇总中的 `risk_count`、`total_sentences` 是平均风险句数和平均验证句数。ROUGE 使用词项重叠及最长公共子序列的 F1。Length Ratio 先按摘要计算纠错后词数与原词数之比，再求平均。

`paired_error_reduction_rate` 对纠错前风险句数大于 0、且有前后验证计数的摘要，分别计算 `(风险句数_before − 风险句数_after) / 风险句数_before`，再取平均。缺失的指标值以 `null` 保存，汇总使用有数值的记录。

默认长度分桶按原摘要词数划分：short ≤150，medium 为 151–225，long >225；前后摘要沿用同一组别。表格直接使用评估文件中的分桶结果。

## 受控摘要长度实验

准备好前面的 `documents_main_600.jsonl` 后运行：

```bash
CONFIG=configs/controlled_length_100_gpu.env \
  bash scripts/run_controlled_length_experiment.sh
```

脚本取前 100 篇报告，保存为 `data/processed/documents_controlled_length_100.jsonl`，为每篇报告分别生成三组摘要，再用相同的 ARIEL 和 NLI 设置评估。各组运行到纠错前评估阶段。

| 设置 | 目标词数 | 分段最大新增 tokens | 最终最大新增 tokens |
|---|---:|---:|---:|
| short | 80 | 128 | 160 |
| medium | 240 | 256 | 384 |
| long | 520 | 640 | 900 |

`LENGTH_SPECS` 用 `标签:目标词数:分段输出上限:最终输出上限` 定义每组。提示中的目标词数容差为 15%，汇总表报告实际平均词数。各组改变目标长度及生成预算，复用同一份文档子集。

中间结果位于 `data/results/controlled_length_100_wide_gpu/short/`、`medium/`、`long/`，每组含生成摘要、证据、NLI 和评估文件。脚本自动将各标签的结果写入 `outputs/tables/controlled_length_100_wide_gpu/controlled_length.md` 与 `controlled_length.json`。

## 调整一次运行

### 配置加载顺序

脚本首先加载 `CONFIG` 指定的 `.env`，再为尚未设置的变量填入默认值。因此 `.env` 中的赋值会覆盖同名外部环境变量。单次调整时，在子 Shell 中导出配置，再设置覆盖值并清空 `CONFIG`。例如只比较 BM25 和 ARIEL：

```bash
(
  set -a
  source configs/main_600_compare_gpu.env
  set +a
  CONFIG= METHODS="bm25 ariel" \
    RUN_DIR=data/results/compare_bm25_ariel \
    bash scripts/run_compare_retrieval.sh
)
```

只做纠错前验证时使用已有摘要，并关闭纠错及后验证：

```bash
(
  set -a
  source configs/main_600_pipeline_gpu.env
  set +a
  CONFIG= RUN_DIR=data/results/ariel_verify_only \
    RUN_CORRECTION=0 RUN_AFTER_VERIFY=0 \
    bash scripts/run_pipeline.sh
)
```

`RUN_CASE_EXPORT` 控制主流水线的案例导出；检索对比脚本中的同类开关是 `EXPORT_CASES`。`NLI_LIMIT` 设置一次运行中验证的句子数。完整文档规模由数据准备时的 `--limit` 确定。

### 在 CPU 上运行

检索对比和表格生成可使用 `configs/main_600_compare.env`、`configs/main_600_tables.env`，输出目录分别为 `data/results/main_600_compare/` 和 `outputs/tables/main_600/`。普通流水线配置中的摘要与纠错设备为 `auto`；下面的命令显式让纠错在 CPU 上运行，并读取已生成的主实验摘要：

```bash
CONFIG=configs/main_600_compare.env bash scripts/run_compare_retrieval.sh

(
  set -a
  source configs/main_600_pipeline.env
  set +a
  CONFIG= SUMMARY_DEVICE=cpu CORRECTION_DEVICE=cpu \
    bash scripts/run_pipeline.sh
)

CONFIG=configs/main_600_tables.env bash scripts/make_tables.sh
```

首次生成固定摘要时，将前面 `src.summarize` 命令的 `--device` 改为 `cpu`。`dtype=auto` 在 CPU 上使用 float32。

### 复用输入与继续运行

生成摘要的阶段按 `doc_id` 续跑。改变生成模型、目标长度或输入文档时，为摘要选择新的输出路径，并将检索与纠错配置中的 `SUMMARIES` 同步指向该文件。对同一组固定摘要比较新检索参数时，保留 `SUMMARIES`，为结果选择新的 `RUN_DIR`。

`FORCE` 控制文档块的重建，默认 `1`；设为 `0` 时可复用已有的非空 `CHUNKS` 文件。其余检索、验证、纠错和评估阶段会重新写出各自结果。受控长度脚本默认复用已存在的文档子集；更换来源或样本数时，设置 `FORCE_DOCUMENT_SUBSET=1` 并使用新的 `RUN_ROOT`。

使用早期版本的生成摘要时，可直接复用摘要 JSONL，在新的结果目录重新运行分句及后续阶段。当前句子 ID 包含文档、摘要和原句位置；覆盖率消融保留上下文扩展。新目录产生的一整组证据和验证结果使用相同的关联方式与消融定义。

### 保存实验设置

为每组实验保存输入文档、生成摘要、所用配置、结果文件和运行环境。以下命令记录 GPU 主实验的环境与代码版本：

```bash
mkdir -p outputs/reproduction
python --version > outputs/reproduction/python.txt
python -m pip freeze > outputs/reproduction/requirements.txt
nvidia-smi > outputs/reproduction/gpu.txt
git rev-parse HEAD > outputs/reproduction/commit.txt
cp configs/main_600_compare_gpu.env configs/main_600_pipeline_gpu.env \
  configs/main_600_tables_gpu.env outputs/reproduction/
```

依赖安装使用 `requirements.txt` 中的包名；上面的记录保存该次实际安装版本。模型使用本地路径时，连同 tokenizer 和模型配置保存相应目录；按 Hub 名称运行时，记录所用缓存快照版本。

## 运行代码检查

```bash
python3 -m unittest discover -s tests -v
```

测试使用标准库和合成输入，覆盖查询 ID、前后句子对应关系、证据融合、消融、局部改写、评估与 Shell 参数传递。模型调用使用替身，整组检查在本地执行。

仓库另有 `scripts/run_debug.sh`，用于下载五条 GovReport 样本、把参考摘要切成查询句，并在 CPU 上执行 BM25、Dense、Hybrid 和 NLI 输入构造。该脚本的输出位于 `/tmp/long_summary_factuality_debug_*`。`configs/debug.env` 则供主流水线读取预先准备的 `documents_debug_5.jsonl`、`summaries_debug_5.jsonl` 及本地模型缓存；两者承担不同的检查步骤。
