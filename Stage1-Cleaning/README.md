# project

一个用于“数据清理 + 0.6B 模型 SFT 微调”的可维护、可复现项目脚手架。

## 1. 环境安装（默认：Conda）

```bash
conda create -n sft_project python=3.11 -y
conda activate sft_project
pip install -r requirements.txt
```

若不用 Conda，可自行用 `python -m venv` 创建虚拟环境后再执行 `pip install -r requirements.txt`。

## 2. 配置说明

所有可调参数统一放在 `configs/default.yaml`，包括：

- 数据路径（raw/interim/processed/samples）
- 模型与训练参数（模型名、batch size、epoch 等）
- 采样、划分和校验参数
- 输出与日志目录

建议通过脚本参数 `--config` 指定配置文件，避免在代码中硬编码。

## 3. Nemotron：从 `text` 抽取 Question / Answer

对 `data/raw/nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl` 中每条记录的 **`text`** 字段，按关键词切分问答：

1. **Markdown 标题**：最后一个匹配「行首 `#` 标题且含单词 `question`」的标题行之后，到第一个「行首 `#` 标题且含单词 `answer`」的标题行之前 → **question**；**answer** 为该 Answer 标题行之后至文末。（例如 `## Interview Question` 与 `## Answer`。）
2. **粗体标签**：`**Question:**` 之后到 `**Answer:**` 或 `**Anwer:**`（拼写容错）之前 → **question**；其后为 **answer**。

在项目根目录执行（需已激活 Conda 环境并安装依赖）：

```bash
# 全量处理，结果写入 data/interim/
python scripts/extract_nemotron_qna.py \
  --input data/raw/nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl \
  --output data/interim/nemotron_qna_extracted.jsonl
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--input` | 输入 JSONL，每行一个对象，需含 `text`；默认即为上述 Nemotron 路径 |
| `--output` | 输出 JSONL 路径；默认 `data/interim/nemotron_qna_extracted.jsonl` |
| `--max-records N` | 只处理前 N 条，用于试跑 |
| `--skip-unmatched` | 仅写出能解析出问答结构的行（`match_kind` 不为 `none`） |
| `--single-pair-only` | 每条 source record 只保留第一组问答（默认会尽量抽取多组） |

示例：小样本试跑、只保留匹配成功的行：

```bash
python scripts/extract_nemotron_qna.py --max-records 2000 \
  --output data/samples/nemotron_qna_sample.jsonl

python scripts/extract_nemotron_qna.py --skip-unmatched \
  --output data/interim/nemotron_qna_matched_only.jsonl
```

输出每行 JSON 字段含义：`source_id`（原记录 id）、`pair_index`（同一条文本的第几组问答）、`match_kind`（`markdown_headers` / `bold_labels` / `labeled_blocks` / `heading_question_solution` / `none`）、`question`、`answer`（无法匹配时为 `null`）。

在代码中复用解析逻辑：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from project_name.nemotron_qna import parse_question_answer

r = parse_question_answer(your_text)
# r.question, r.answer, r.kind
```

## 4. 用 Teacher 模型做 30 条抽样测评

该测评会从 `data/interim/nemotron_qna_matched_only.jsonl` 随机抽样，调用 `.env` 中的 teacher 模型参数（`TEACHER_BASE_URL`、`TEACHER_API_KEY`、`TEACHER_MODEL`）作为评审，输出：

- `extraction_ok`：是否正确抽取出问答对
- `answer_ok`：答案是否正确回答该问题

运行命令：

```bash
python scripts/06_eval_qna_teacher.py \
  --input data/interim/nemotron_qna_matched_only.jsonl \
  --sample-size 30 \
  --seed 42 \
  --env-file .env \
  --output-dir outputs/eval_qna_teacher
```

输出文件：

- `outputs/eval_qna_teacher/judged_details.jsonl`：逐条评测明细
- `outputs/eval_qna_teacher/summary.json`：汇总指标（含 both_ok_rate）

## 5. 二次筛选：按 answer 最后出现数字保留样本

该步骤会读取抽取后的 `question+answer` JSONL，从 `answer` 中取“最后出现的数字表达”作为最终答案候选，并仅保留候选值是纯数字形式的记录（支持整数、小数、科学计数法、普通分数、LaTeX 分数 `\\frac{a}{b}`）。

```bash
python scripts/07_filter_numeric_final_answer.py \
  --input data/interim/nemotron_qna_matched_only.jsonl \
  --output data/interim/nemotron_qna_numeric_only.jsonl \
  --keep-candidate-field
```

说明：
- 会过滤掉含变量表达式（如 `2x`、`3x+1`）以及完全无数字答案的样本。
- `--keep-candidate-field` 会在输出里追加 `final_answer_candidate`，方便人工 spot-check。

## 6. 长度分桶：small / middle / large / super-large

将 `question + answer` 的总 token 数（轻量近似 token 计数）按以下区间分桶，超出范围样本会丢弃：

- `small`: `[100, 400)`
- `middle`: `[400, 1000)`
- `large`: `[1000, 2000)`
- `super-large`: `[2000, 4096]`

```bash
python scripts/08_bucket_by_length.py \
  --input data/interim/nemotron_qna_numeric_only.jsonl \
  --output-dir data/interim/length_buckets \
  --keep-token-field
```

输出文件：

- `data/interim/length_buckets/small.jsonl`
- `data/interim/length_buckets/middle.jsonl`
- `data/interim/length_buckets/large.jsonl`
- `data/interim/length_buckets/super-large.jsonl`

## 7. 长度分桶抽样评测（每类 10 条）

从 `length_buckets` 四个类别各抽 10 条，通过 `.env` 中 teacher 模型评估 `answer` 是否正确回答 `question`。  
评估时由 LLM 自行从 `answer` 中判断最终答案，不使用 `final_answer_candidate` 字段。

```bash
python scripts/09_eval_length_buckets_answer.py \
  --input-dir data/interim/length_buckets \
  --sample-per-bucket 10 \
  --seed 42 \
  --env-file .env \
  --output-dir outputs/eval_length_buckets_answer
```

输出文件：

- `outputs/eval_length_buckets_answer/judged_details.jsonl`
- `outputs/eval_length_buckets_answer/summary.json`

## 8. 用本地 0.6B vLLM 生成 boxed SFT 数据

该步骤会读取问答数据集，调用本地 vLLM（OpenAI-compatible API）抽取最终答案，并输出 SFT 标准格式 JSON（`messages` 包含 `system/user/assistant`）。  
`assistant` 只输出 `\\boxed{...}`；找不到则 `\\boxed{None}`。

```bash
python scripts/10_build_boxed_sft_with_vllm.py \
  --input data/interim/nemotron_qna_numeric_only.jsonl \
  --output data/processed/sft_boxed_final_answer.json \
  --env-file .env \
  --temperature 0.0
```

环境变量（`.env`）：
- `STUDENT_BASE_URL`
- `STUDENT_API_KEY`
- `STUDENT_MODEL`

## 9. 评估 boxed answer 准确度

该步骤会读取 SFT JSON，先从 `assistant` 中抽取 `\\boxed{...}`，再调用 teacher 模型仅基于 `question` 独立生成 `\\boxed{...}`，最后比较两者一致性并统计准确率。

```bash
python scripts/11_eval_boxed_extraction.py \
  --input data/processed/sft_boxed_small.json \
  --env-file .env \
  --output-dir outputs/eval_boxed_extraction
```

可选：随机抽样 N 条评估（便于快速验证）

```bash
python scripts/11_eval_boxed_extraction.py \
  --input data/processed/sft_boxed_small.json \
  --sample-size 30 \
  --seed 42 \
  --env-file .env \
  --output-dir outputs/eval_boxed_extraction
```

输出文件：

- `outputs/eval_boxed_extraction/boxed_eval_details.jsonl`：逐条对比明细（`generated_boxed` / `extracted_boxed` / `is_match` / `reason`）
- `outputs/eval_boxed_extraction/boxed_eval_summary.json`：汇总指标（`match_count`、`total`、`match_rate`、`error_count`）

## 10. 完整运行命令

（对应脚本与训练入口实现后，在此填写具体命令。）

```bash
# 步骤 1 — 数据检视（inspect）
#
# 步骤 2 — 数据过滤（filter）
#
# 步骤 3 — 文档路由 / 分类（route）
#
# 步骤 4 — 生成 SFT 样本（generate）
#
# 步骤 5 — 校验 SFT（validate）
#
# 步骤 6 — 划分 train/val 或训练（split / train）
#
# Makefile 聚合命令（如有）
#
```

## 11. 数据流（Data Flow）



## 12. 输出文件约定

- `outputs/<timestamp>/`：每次实验的统计、报告、指标与快照
- `logs/`：运行日志
- `data/processed/`：最终可训练的数据文件（例如 `train.jsonl`、`val.jsonl`）

## 13. 目录职责

- `src/project_name/`：可复用逻辑，不直接承载完整流水线入口
- `scripts/`：单步 CLI 入口，每个脚本只做一个明确步骤
- `notebooks/`：仅用于探索分析和错误排查，不作为正式入口
- `tests/`：单元测试

