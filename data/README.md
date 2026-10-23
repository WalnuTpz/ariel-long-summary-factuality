# 数据与结果格式

流水线以 UTF-8 JSONL 交换记录：每行是一个 JSON 对象。文档通过 `doc_id` 关联摘要与证据块，摘要通过 `(doc_id, summary_id)` 关联评估结果，句子通过 `query_id` 关联检索、验证和编辑记录。

`data/raw/` 保存导入文件，`data/processed/` 保存规范化文档和固定生成摘要，`data/results/<run>/` 保存一次运行的中间结果。表格写入 `outputs/tables/`。这些数据和运行产物保存在本地，Git 跟踪本目录的格式说明。

## 接入一组文档与已有摘要

文档记录的核心字段是 `doc_id` 和 `document`；`reference_summary` 用于 ROUGE，`title` 和 `meta` 保存标题与来源。下面建立一组可在本地检查的合成输入：

```bash
mkdir -p data/raw data/processed
cat > data/raw/example_documents.jsonl <<'JSONL'
{"doc_id":"d1","title":"Program costs","document":"The agency reported a 5% cost increase in 2021.\n\nThe program served 200 households.","reference_summary":"Costs rose 5% in 2021, and the program served 200 households.","meta":{"dataset":"example"}}
JSONL

cat > data/processed/example_summaries.jsonl <<'JSONL'
{"doc_id":"d1","summary_id":"d1_example","generated_summary":"The agency reported a 10% cost increase in 2021. The program served 200 households."}
JSONL

python3 -m src.load_data \
  --input data/raw/example_documents.jsonl \
  --output data/processed/example_documents.jsonl \
  --min-document-chars 1

python3 -m src.split_summary_sentences \
  --input data/processed/example_summaries.jsonl \
  --output data/processed/example_sentences.jsonl \
  --summary-field generated_summary

python3 -m src.chunking \
  --input data/processed/example_documents.jsonl \
  --output data/processed/example_chunks.jsonl \
  --chunk-type paragraph --max-tokens 256

python3 -m src.retrieval_bm25 \
  --chunks data/processed/example_chunks.jsonl \
  --queries data/processed/example_sentences.jsonl \
  --output data/results/example/retrieval.bm25.jsonl --top-k 2
```

这组命令在仓库根目录用 Python 标准库执行，产生两个摘要句和两个证据块。第一句将报告中的 `5%` 写为 `10%`，第二句沿用人数信息，便于观察句子与源文档的对应关系。

接入现有模型摘要时，按示例提供 `generated_summary`，设置 `SUMMARY_FIELD=generated_summary`。完整流水线读取规范化文档和这份摘要文件，使用 `RUN_SUMMARIZE=0` 保持摘要固定。若原文件的摘要字段名不同，`--summary-field` 或 `SUMMARY_FIELD` 指定实际字段名。

## 文档、摘要与句子如何关联

### 文档来源

规范化器把多种常见字段转换为 `document`、`reference_summary` 等统一字段，并整理段落内空白。GovReport 下载器生成的文档 ID 形如 `govreport_00042`，其中数字是从 0 开始的原数据集序号；`meta.source_index` 保留同一序号。按正文长度过滤后，后续阶段仍沿用这个 ID。

自有数据使用在本次输入中唯一、稳定的 `doc_id`。同一文档的所有摘要、文档块和验证结果使用相同的值。

### 多份摘要

`summary_id` 区分同一文档上的不同摘要。导入摘要可显式指定它；未提供时，分句和评估使用 `<doc_id>_summary`。摘要生成器按文档与模型名称生成 ID，例如 Qwen2.5-1.5B-Instruct 对应 `<doc_id>_qwen2p5_1p5b_instruct`，也可通过 `--summary-id-suffix` 指定后缀。

模型输出另含 `model`、`generation_method`、`summary_word_count` 和 `meta`。`generation_method` 为 `single_pass` 或 `chunk_summarize_then_merge`；`meta` 记录输入模型 token 数、窗口数量、目标词数、窗口宽度和重叠量。

### 稳定的验证单元

句子 ID 的格式为 `<doc_id>::<summary_id>::s<三位句序>`。第一条合成摘要句对应如下记录，省略了统计辅助字段：

```json
{
  "query_id": "d1::d1_example::s001",
  "doc_id": "d1",
  "summary_id": "d1_example",
  "sentence_id": 1,
  "text": "The agency reported a 10% cost increase in 2021.",
  "source_summary_field": "generated_summary"
}
```

`sentence_id` 从 1 开始，表示句子在完整原摘要中的位置。默认分句过滤少于八个字符的句子，因此保留下来的句序可以不连续。纠错后的分句结合 `original_summary` 与 `sentence_edits` 恢复原位置；短于八个字符的替换句也沿用原验证单元。

检索与验证按 `query_id` 对应，摘要评估按 `(doc_id, summary_id)` 对应。更换分句规则或导入另一份摘要时，在新的结果目录生成配套的检索、验证和评估文件。

## 从文档块追溯证据

文档块的 `chunk_index` 从 0 开始，`chunk_id` 形如 `d1_chunk_0000`。块记录包含 `text`、`chunk_type`、`token_count`、`char_start`、`char_end` 与 `meta`。

`chunk_type` 标识整段 `paragraph`、长段内部窗口 `paragraph_window` 或全文窗口 `token_window`。`token_count` 用转小写后的 `\w+` 词项计数。字符区间用于定位处理后文档中的来源；长段内部窗口的偏移基于整理过空白的段落文本。

四种检索方法共享下列记录结构。以下 JSON 是单条证据的格式示例，分数用于说明字段位置：

```json
{
  "query_id": "d1::d1_example::s001",
  "doc_id": "d1",
  "query": "The agency reported a 10% cost increase in 2021.",
  "retrieval_method": "bm25",
  "top_k": [
    {
      "rank": 1,
      "chunk_id": "d1_chunk_0000",
      "chunk_index": 0,
      "score": 1.2,
      "text": "The agency reported a 5% cost increase in 2021."
    }
  ]
}
```

`top_k` 按检索得分降序排列。ARIEL 在每项中补充 `base_score`、`coverage_bonus`、三种覆盖率及命中项、`risk_type`、`alpha` 和扩展信息。启用扩展后，`chunk_id` 仍标识原候选，`text` 可以包含相邻块；`expanded_chunk_ids` 列出组成这条证据的块 ID，`expanded_context` 表示是否采用了扩展。

因此阅读 ARIEL 证据时，应将 `text` 与 `expanded_chunk_ids` 一起使用。NLI 的 `selected_evidence_id` 指向这条候选证据的 `chunk_id`，该候选实际包含的全部文本保存在检索记录中。

## 验证记录与编辑记录

### NLI 输出

一条验证记录保存一条摘要句的判断。模型将证据作为 premise，将 `sentence` 作为 hypothesis。标签取 `Supported`、`Unsupported`、`Not Enough Information` 之一。主要字段的关系如下：

| 字段 | 含义 |
|---|---|
| `query_id`、`doc_id`、`summary_id`、`sentence_id` | 句子及摘要身份 |
| `sentence` | 被验证的摘要句 |
| `evidence`、`evidence_ids` | 按相同顺序保存的证据文本与候选块 ID |
| `label`、`raw_label`、`scores` | 项目标签、原始 NLI 标签及其分数 |
| `aggregation` | `per_chunk` 或 `concat` |
| `chunk_predictions` | 逐块 NLI 的分数、标签及检索排名，出现在逐块模式中 |
| `selected_evidence_id` | 逐块聚合选中的蕴含或矛盾证据 |
| `reason`、`model`、`retrieval_method` | 判定说明、验证模型和检索方法 |

逐块聚合先选择达到阈值的蕴含证据，再选择矛盾证据，剩余句子标为 NEI。选中证据时，`scores` 是该条证据的分数；NEI 时保存各标签跨证据的最大分数，`selected_evidence_id` 为空。零证据记录保留为 NEI，此时 `scores` 为空，`aggregation` 为空字符串。

### 局部纠错输出

每篇纠错摘要保存完整原文、新文和风险句的编辑记录。下面示意将合成例句中的百分比改回 5% 后的记录，省略了模型名称和统计字段：

```json
{
  "doc_id": "d1",
  "summary_id": "d1_example",
  "original_summary": "The agency reported a 10% cost increase in 2021. The program served 200 households.",
  "corrected_summary": "The agency reported a 5% cost increase in 2021. The program served 200 households.",
  "sentence_edits": [
    {
      "query_id": "d1::d1_example::s001",
      "sentence_id": 1,
      "original_sentence": "The agency reported a 10% cost increase in 2021.",
      "corrected_sentence": "The agency reported a 5% cost increase in 2021.",
      "verification_label": "Unsupported",
      "evidence_chunk_ids": ["d1_chunk_0000"],
      "selected_evidence_id": "d1_chunk_0000",
      "action": "replace",
      "reason": "Replaced the percentage using the source evidence."
    }
  ]
}
```

`replace` 表示接受模型返回的改写，`keep` 表示风险句保留原文。低置信度 NEI、模型原样返回的句子，以及解析或格式检查失败的句子都记录 `keep`，并在 `reason` 中说明原因；输出检查失败还记录 `fallback_reason`。Supported 句原样保留，省略其编辑项。

`num_sentences` 是参与该次处理的验证单元数，`num_sentence_edits` 是风险句编辑记录数。完整原摘要中的短句在 `corrected_summary` 中保留。前后词数记录在 `original_summary_word_count` 和 `corrected_summary_word_count` 中。

## 从结果文件读取指标

`evaluation.<method>.jsonl` 每行对应一个摘要版本。`summary_version` 为 `before_correction` 或 `after_correction`，同一摘要的前后记录共享 `doc_id`、`summary_id` 和基于原摘要长度得到的 `length_bucket`。

逐摘要记录包含 ROUGE-1/2/L、`supported_ratio`、`unsupported_or_nei_ratio`、`summary_word_count` 及来源信息。有验证结果时，还包含 `total_sentences`、三类句数、`risk_count` 和 `label_counts`。纠错后记录补充 `length_ratio` 和编辑记录数。缺少参考文本时 ROUGE 为 `null`，缺少匹配的验证结果时事实性比例为 `null`。

`evaluation.<method>.summary.json` 包含四部分：

| 字段 | 内容 |
|---|---|
| `stats` | 读取和写出的记录数，以及缺少参考文本、缺少验证结果的记录数 |
| `by_version` | 纠错前后分别汇总的指标 |
| `by_length_bucket` | short、medium、long 中各版本的指标 |
| `paired_error_reduction_rate` | 对有前后验证计数且原风险句数大于 0 的摘要，计算风险句数相对下降率后取平均 |

汇总层的 `count` 为摘要记录数，其余数值指标对该组中有值的记录求平均；其中 `risk_count` 和 `total_sentences` 表示平均句数。Supported Ratio 与 Risk Ratio 也先按摘要计算再平均。默认分桶依据为原摘要词数：short ≤150，medium 为 151–225，long >225。

`cases.<method>.jsonl` 将某个 `query_id` 的原句、编辑、新句标签、证据和摘要指标集中到一个对象，`cases.<method>.md` 是相应的阅读版本。使用完整验证文件统计句子分布，使用案例文件查看导出器选中的具体样例。
