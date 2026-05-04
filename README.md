# project

一个用于“数据清理 + 0.6B 模型 SFT 微调”的可维护、可复现项目脚手架。

## 1. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 配置说明

所有可调参数统一放在 `configs/default.yaml`，包括：

- 数据路径（raw/interim/processed/samples）
- 模型与训练参数（模型名、batch size、epoch 等）
- 采样、划分和校验参数
- 输出与日志目录

建议通过脚本参数 `--config` 指定配置文件，避免在代码中硬编码。

## 3. 完整运行命令

按步骤执行（推荐）：

```bash
python scripts/00_inspect_data.py --config configs/default.yaml
python scripts/01_filter_data.py --config configs/default.yaml
python scripts/02_route_docs.py --config configs/default.yaml
python scripts/03_generate_sft.py --config configs/default.yaml
python scripts/04_validate_sft.py --config configs/default.yaml
python scripts/05_build_train_val.py --config configs/default.yaml
```

或使用 `Makefile`：

```bash
make pipeline
```

## 4. 数据流（Data Flow）

1. `data/raw/`：原始数据（只读）
2. `scripts/00_inspect_data.py`：统计字段、样本和质量概览，输出到 `outputs/`
3. `scripts/01_filter_data.py`：按规则清洗并写入 `data/interim/`
4. `scripts/02_route_docs.py`：文档路由（规则/LLM）并写入 `data/interim/`
5. `scripts/03_generate_sft.py`：构造 SFT 样本，输出到 `data/processed/`
6. `scripts/04_validate_sft.py`：执行结构与内容校验，报告写入 `outputs/`
7. `scripts/05_build_train_val.py`：划分训练/验证集，写入 `data/processed/`

## 5. 输出文件约定

- `outputs/<timestamp>/`：每次实验的统计、报告、指标与快照
- `logs/`：运行日志
- `data/processed/`：最终可训练的数据文件（例如 `train.jsonl`、`val.jsonl`）

## 6. 目录职责

- `src/project_name/`：可复用逻辑，不直接承载完整流水线入口
- `scripts/`：单步 CLI 入口，每个脚本只做一个明确步骤
- `notebooks/`：仅用于探索分析和错误排查，不作为正式入口
- `tests/`：单元测试

