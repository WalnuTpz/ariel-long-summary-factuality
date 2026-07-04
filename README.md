# 面向长文档摘要事实性验证的 ARIEL

本仓库实现了一个轻量级流水线，用于长文档摘要中的证据驱动事实性验证和局部纠错。

项目最初是自然语言处理课程的大作业，当前版本整理为便于复现和二次开发的公开实现。

给定一篇源文档和一段生成摘要，系统会先将摘要拆分为句子级事实声明，再为每个声明从源文档中检索证据，使用 NLI 模型判断该声明是否被证据支持，并可选地对模型预测为高风险的句子进行证据约束的局部改写。

## 项目亮点

- 面向 GovReport 风格输入的长文档摘要事实性评估流水线。
- 支持 BM25、稠密向量检索、固定混合检索和 ARIEL 等句子级证据定位方法。
- ARIEL 证据定位包含三项主要机制：风险类型识别与风险自适应 BM25/向量融合权重、覆盖率感知重排序、自适应邻近证据扩展。
- 使用检索到的证据作为前提、摘要句子作为假设，通过 NLI 模型估计句子级事实一致性。
- 对模型预测的高风险句子进行保守的局部纠错，避免自由重写整段摘要。
- 提供 ROUGE、自动事实性估计、摘要长度分析、检索方法对比、ARIEL 消融和纠错前后对比等评估脚本。

## 仓库结构

```text
configs/       debug 和主实验使用的环境变量风格配置
data/          数据格式说明；大型数据文件不纳入仓库
scripts/       流水线、实验和表格生成入口
src/           核心实现代码
requirements.txt
```

公开仓库不包含完整数据集、生成摘要、实验输出、模型权重、论文草稿或展示文件。

## 环境安装

推荐使用 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次运行时可能需要下载 HuggingFace 数据集和模型。完整的摘要生成、NLI 验证和局部纠错建议使用 GPU。

## 快速检查

仓库没有提交真实 debug 数据。debug 脚本会临时下载一个小型 GovReport 样本到 `/tmp`，并使用参考摘要作为句子级查询，检查数据加载、文档切块、证据检索和验证输入构造是否正常：

```bash
bash scripts/run_debug.sh
```

## 数据准备

下载 GovReport 样本：

```bash
python scripts/download_govreport_sample.py \
  --split validation \
  --limit 600 \
  --output data/raw/govreport_main_600.jsonl
```

规范化下载得到的 JSONL 文件：

```bash
python -m src.load_data \
  --input data/raw/govreport_main_600.jsonl \
  --output data/processed/documents_main_600.jsonl \
  --min-document-chars 2000
```

完整实验还需要生成摘要。可以使用 `src.summarize` 单独生成，也可以在所选配置中设置 `RUN_SUMMARIZE=1` 后运行 `scripts/run_pipeline.sh`。

## 主要命令

比较检索方法并运行 ARIEL 消融实验：

```bash
CONFIG=configs/main_600_compare_gpu.env bash scripts/run_compare_retrieval.sh
```

运行 ARIEL 验证和局部纠错流水线：

```bash
CONFIG=configs/main_600_pipeline_gpu.env bash scripts/run_pipeline.sh
```

根据实验输出生成表格：

```bash
CONFIG=configs/main_600_tables_gpu.env bash scripts/make_tables.sh
```

运行受控摘要长度实验：

```bash
CONFIG=configs/controlled_length_100_gpu.env \
bash scripts/run_controlled_length_experiment.sh
```

`configs/` 中也提供 CPU 配置，但完整实验在没有 GPU 的情况下会明显更慢。

## 说明

- 自动事实性标签来自 NLI 模型估计，不是人工标注的金标准标签。
- ROUGE 衡量的是生成摘要与参考摘要的词面重叠，不应被解释为事实性指标。
- 生成数据和实验输出会被 git 忽略。
- 配置文件中的路径可以按本地数据集和模型缓存位置修改。
