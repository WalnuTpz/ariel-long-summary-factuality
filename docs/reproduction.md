# 实验复现

本文档说明如何使用仓库现有脚本运行 GovReport 主实验、ARIEL 消融、局部纠错和受控摘要长度实验。方法概览与小样本运行流程见 [README](../README.md)。

所有命令均在仓库根目录执行，并使用同一个 `.venv` 环境。

## 1. 环境与模型

先完成 [README 中的安装步骤](../README.md#1-安装)。主实验的 `*_gpu.env` 配置使用 CUDA，Python 入口为 `.venv/bin/python`。

| 用途 | 模型 |
|---|---|
| 摘要生成与局部纠错 | `Qwen/Qwen2.5-1.5B-Instruct` |
| Dense 检索 | `sentence-transformers/all-MiniLM-L6-v2` |
| NLI 验证 | `facebook/bart-large-mnli` |

配置中的 `HF_LOCAL_FILES_ONLY=0` 允许首次运行时下载模型。数据由 `datasets` 从 `ccdv/govreport-summarization` 加载。模型缓存、数据文件与实验输出存放在本地。

论文附录记录的材料整理环境为 Python 3.12.3、PyTorch 2.11.0、Transformers 5.8.0、Sentence Transformers 5.4.1 和 Datasets 4.8.5。仓库的 `requirements.txt` 当前未固定版本；重新运行时可将实际环境与结果一并保存：

```bash
mkdir -p outputs/reproduction
python --version > outputs/reproduction/python.txt
python -m pip freeze > outputs/reproduction/requirements.txt
nvidia-smi > outputs/reproduction/gpu.txt
git rev-parse HEAD > outputs/reproduction/commit.txt
```

小样本检查命令为 `bash scripts/run_debug.sh`，在 CPU 上运行到检索结果与 NLI 输入构造阶段。

## 2. 准备主实验文档

下载 GovReport validation split 中前 600 条有效记录。下载脚本要求正文和参考摘要均非空，并过滤正文少于 2,000 字符的记录。

```bash
python scripts/download_govreport_sample.py \
  --split validation \
  --limit 600 \
  --min-document-chars 2000 \
  --output data/raw/govreport_main_600.jsonl

python -m src.load_data \
  --input data/raw/govreport_main_600.jsonl \
  --output data/processed/documents_main_600.jsonl \
  --min-document-chars 2000
```

采样按数据集流式读取顺序进行。记录中的 `doc_id` 和 `meta.source_index` 保留对应关系；后续实验共享这份文档文件。JSONL 字段说明见 [data/README.md](../data/README.md)。

## 3. 生成并固定主实验摘要

使用 Qwen2.5-1.5B-Instruct 生成摘要，并直接保存到两份主实验配置共同读取的 `SUMMARIES` 路径：

```bash
python -m src.summarize \
  --input data/processed/documents_main_600.jsonl \
  --output data/processed/generated_summaries.qwen2p5_1p5b.main_600.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --dtype auto \
  --target-words 240 \
  --chunk-input-tokens 6000 \
  --chunk-overlap-tokens 256 \
  --chunk-max-new-tokens 256 \
  --final-max-new-tokens 384 \
  --resume \
  --continue-on-error
```

长文档先按模型 token 窗口生成分段摘要，再合并为最终摘要。命令使用贪心解码；`--resume` 会跳过输出中已有的文档，便于中断后继续运行。

生成结束后查看 `written_summaries`、`skipped_existing_doc_id` 和 `failed_documents`，并核对文档与摘要记录数：

```bash
wc -l data/processed/documents_main_600.jsonl \
  data/processed/generated_summaries.qwen2p5_1p5b.main_600.jsonl
```

完整主实验使用 600 篇文档及各自的一条生成摘要。检索对比、消融和纠错均复用这份摘要文件，因此保留主流水线配置中的 `RUN_SUMMARIZE=0`。修改生成设置时使用新的输出文件，并同步更新配置中的 `SUMMARIES`。

论文报告的这批摘要平均长 193.5 词，共 5,338 个句子级验证单元。

## 4. 检索方法对比与消融

```bash
CONFIG=configs/main_600_compare_gpu.env bash scripts/run_compare_retrieval.sh
```

脚本对相同文档块和摘要句子运行四种检索方法，以及三组 ARIEL 消融：

| 实验 | 配置中的方法名 |
|---|---|
| BM25 | `bm25` |
| Dense embedding | `embedding` |
| Fixed Hybrid | `hybrid` |
| ARIEL | `ariel` |
| 移除覆盖率重排序 | `ariel_no_coverage` |
| 移除自适应上下文扩展 | `ariel_no_expansion` |
| 移除风险自适应权重 | `ariel_no_risk_weighting` |

主要参数如下；完整配置见 [main_600_compare_gpu.env](../configs/main_600_compare_gpu.env)。

| 参数 | 设置 |
|---|---|
| 文档切块 | `CHUNK_TYPE=paragraph`，`CHUNK_MAX_TOKENS=256` |
| 检索证据数 | Top-3 |
| Fixed Hybrid 的 BM25 权重 | 0.5 |
| NLI 最大输入长度 | 1,024 个模型 token |
| NLI 聚合 | `per_chunk`，分别验证每条证据 |
| 蕴含／矛盾阈值 | 均为 0.5 |

覆盖率消融仅移除排序加分，仍按覆盖率判断是否扩展上下文；上下文扩展消融仅关闭扩展。
Fixed Hybrid 与 ARIEL 使用相同的 min-max 归一化规则，同分证据按 chunk ID 排序。
早期版本的覆盖率消融同时关闭了扩展，相关旧结果需要重新运行才能用于单因素比较。

结果位于 `data/results/main_600_compare_gpu/`：

```text
chunks.jsonl                         共享文档块
summary_sentences.jsonl              共享摘要句子
retrieval.<method>.jsonl              各方法的证据与分数
verification.<method>.jsonl           各方法的句子级 NLI 结果
evaluation.<method>.jsonl             各摘要的评估记录
evaluation.<method>.summary.json      汇总指标与长度分桶
comparison.md / comparison.tsv       方法对比表
```

## 5. 局部纠错与再次验证

```bash
CONFIG=configs/main_600_pipeline_gpu.env bash scripts/run_pipeline.sh
```

该配置使用 ARIEL 证据定位，依次完成纠错前验证、Qwen2.5 局部改写、纠错后分句与重新检索、再次 NLI 验证和成对评估。

纠错对象为 Unsupported 句子，以及 neutral 分数不低于 0.7 的 Not Enough Information 句子。每次改写最多使用 5 条筛选后的证据句，提示输入上限为 4,096 tokens，输出上限为 128 tokens。

结果位于 `data/results/main_600_ariel_gpu/`，主要文件为：

```text
verification.before.ariel.jsonl       纠错前 NLI 结果
corrections.ariel.jsonl               纠错后摘要与逐句编辑记录
verification.after.ariel.jsonl        纠错后 NLI 结果
evaluation.ariel.jsonl                纠错前后逐摘要指标
evaluation.ariel.summary.json         汇总指标
cases.ariel.md / cases.ariel.jsonl    导出的代表性案例
```

完整配置见 [main_600_pipeline_gpu.env](../configs/main_600_pipeline_gpu.env)。

## 6. 汇总主实验表格

完成检索对比和纠错后运行：

```bash
CONFIG=configs/main_600_tables_gpu.env bash scripts/make_tables.sh
```

脚本分别读取 `data/results/main_600_compare_gpu/` 和 `data/results/main_600_ariel_gpu/`，将表格写入 `outputs/tables/main_600_gpu/`。

| 论文中的实验内容 | 输出文件 |
|---|---|
| 主实验总体指标与数据统计 | `table_1_main_metrics.md` |
| 自然生成摘要的长度分桶 | `table_2_length_buckets.md` |
| BM25 / Embedding / Fixed Hybrid / ARIEL 对比 | `table_3_retrieval_comparison.md` |
| ARIEL 三组消融 | `table_4_ariel_ablation.md` |
| 局部纠错前后对比 | `table_5_correction.md` |
| 上述表格合集 | `paper_tables.md` |

表格文件编号沿用脚本命名。论文中的消融图可使用 `table_4_ariel_ablation.md` 的数值绘制。自然长度分桶使用原始摘要词数：短摘要 ≤ 150 词，中等摘要为 151–225 词，长摘要 > 225 词。

Supported Ratio 和 Risk Ratio 来自句子级 NLI 标签，先在每篇摘要内计算，再对摘要取平均。ROUGE-1/2/L 使用仓库实现的 F1 分数；纠错结果还包含平均风险句数、Error Reduction Rate 与 Length Ratio。

## 7. 受控摘要长度实验

此实验复用第 2 步的文档文件，首次运行时取前 100 篇，保存为 `data/processed/documents_controlled_length_100.jsonl`。每篇文档在三个目标长度下分别生成摘要，并使用 ARIEL + NLI 验证。

```bash
CONFIG=configs/controlled_length_100_gpu.env \
  bash scripts/run_controlled_length_experiment.sh
```

| 设置 | 目标词数 | 分段摘要最大新增 tokens | 最终摘要最大新增 tokens |
|---|---:|---:|---:|
| short | 80 | 128 | 160 |
| medium | 240 | 256 | 384 |
| long | 520 | 640 | 900 |

提示中的目标词数容差为 15%；实际生成长度由结果表中的平均摘要词数报告。这组实验运行到 NLI 评估，`RUN_CORRECTION=0`。

三个设置的中间结果分别位于 `data/results/controlled_length_100_wide_gpu/short/`、`medium/`、`long/`；汇总表自动保存为：

```text
outputs/tables/controlled_length_100_wide_gpu/controlled_length.md
outputs/tables/controlled_length_100_wide_gpu/controlled_length.json
```

## 8. 调整配置与保留实验记录

配置文件会覆盖同名外部环境变量。修改参数时，可以直接编辑对应 `.env`；需要单次覆盖时，先加载配置，再设置覆盖值，并让脚本跳过再次加载：

```bash
(
  set -a
  source configs/main_600_pipeline_gpu.env
  set +a
  CONFIG= RUN_DIR=data/results/main_600_ariel_verify_only \
    RUN_CORRECTION=0 RUN_AFTER_VERIFY=0 bash scripts/run_pipeline.sh
)
```

句子标识包含文档和摘要 ID，纠错后沿用原句位置，具体格式见 [数据说明](../data/README.md)。
升级旧版结果时可保留生成摘要，但应重新生成分句及其下游结果。
案例导出只读取本次启用阶段的结果；表格使用评估文件已有的长度分桶。
指定的主方法、纠错结果目录或受控长度结果缺失时会报错。

每轮实验保留所用配置、规范化文档、生成摘要和相应结果目录。主实验数据与模型当前通过名称下载；保存运行时的数据和模型版本，有助于后续复用同一批输入。配置字段说明见 [configs/README.md](../configs/README.md)。

## 9. 不加载模型的回归检查

```bash
python3 -m unittest discover -s tests -v
```

测试使用标准库和小型合成 JSONL，覆盖证据融合、句子关联、局部纠错、评估及 Bash
实验编排。模型调用和下载边界使用替身，不需要安装模型依赖或下载数据集。
这些检查验证程序行为，不代表真实模型的生成质量或事实性效果。
