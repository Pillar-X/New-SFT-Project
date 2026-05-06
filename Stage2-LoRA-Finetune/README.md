# Stage2 LoRA Finetune Project

本目录用于 Stage2：基于 Stage1 清洗得到的 SFT 数据，对 0.6B 模型进行 LoRA 微调。  
目标是保证结构清晰、流程可复现、后续可扩展。

## 项目目标

- 使用统一配置驱动的数据处理与训练流程。
- 将可复用逻辑沉淀到 `src/`，将可执行入口放在 `scripts/`。
- 保证每次实验可追溯（配置、命令、指标、样例、日志）。

## 目录结构说明

```text
Stage2-LoRA-Finetune/
├── README.md
├── requirements.txt
├── .gitignore
├── Makefile
├── configs/
│   ├── default.yaml
│   └── default_ray_dual_gpu.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── samples/
│   ├── sft/
│   ├── eval/
│   ├── train/
│   └── test/
├── models/
├── src/
│   └── project_name/
├── scripts/
├── evaluation/
├── notebooks/
├── tests/
├── outputs/
└── logs/
```

## 环境安装

```bash
conda create -n llm_project python=3.11 -y
conda activate llm_project
pip install -r requirements.txt
```

## W&B 配置

训练脚本统一只读取仓库根目录 `new-SFT-project/.env`，不再使用子目录 `.env`。

在 [wandb.ai/authorize](https://wandb.ai/authorize) 弹窗里：

- **API KEY**（灰色框内整串，点 **Copy**）才是要写入 `.env` 的值。
- **Key ID**（仅前半段、较短）不是完整密钥；若只填了 Key ID，会出现 `40+ characters, has 31` 这类报错。

请将配置写在仓库根目录 `new-SFT-project/.env`：

```bash
WANDB_API_KEY=粘贴官网灰色框内的完整_API_KEY
WANDB_PROJECT=stage2-qwen3-lora
WANDB_ENTITY=your_team_or_username
WANDB_NAME=qwen3-0.6b-lora-sft-middle
```

## Pipeline 运行命令

```bash
python scripts/00_inspect_data.py --config configs/default.yaml
python scripts/01_filter_data.py --config configs/default.yaml
python scripts/02_split_eval_test.py --seed 42
python scripts/03_train_lora.py --config configs/default.yaml
python scripts/04_eval_boxed_accuracy.py --config configs/default.yaml
python scripts/05_tpe_search.py --config configs/default.yaml --n-trials 30
```

## 如何启动 LoRA 微调

1. 进入项目目录并激活环境：

```bash
cd Stage2-LoRA-Finetune
conda activate llm_project
```

2. 检查 `configs/default.yaml` 中关键参数：
- `finetune.model_path: models/Qwen3-0.6B-Base`
- `finetune.dataset_path: data/sft/sft_boxed_small.json`
- `finetune.output_dir: outputs/qwen3-0.6b-lora`
- `finetune.dtype: auto`（可改为 `float16` / `bfloat16`）
- 单卡默认使用 `configs/default.yaml`

3. 配置 W&B（如需在线记录）：
- 在仓库根目录 `.env` 填好完整 `WANDB_API_KEY`（见上文「API KEY vs Key ID」）
- 在当前训练环境安装 `wandb`：`pip install wandb`
- 如果不想上报 W&B，可将 `finetune.use_wandb` 设为 `false`

4. 启动训练（二选一）：

```bash
python scripts/03_train_lora.py --config configs/default.yaml
```

```bash
make train-lora
```

5. 训练产物查看：
- LoRA adapter 权重默认在 `outputs/qwen3-0.6b-lora/`
- 训练指标会写入同目录下的 `trainer_state.json` 和 `train_results.json`（如有）
- 若启用 Ray 异步评估，还会输出：
  - `outputs/qwen3-0.6b-lora/boxed_eval_reports/boxed_eval_step_*.json`
  - `outputs/qwen3-0.6b-lora/async_eval_snapshots/`（评估完成后自动清理）

### 单机双卡：训练与评估并行（Ray）

如果你希望训练跑在第 1 张卡、评估跑在第 2 张卡，使用独立配置文件 `configs/default_ray_dual_gpu.yaml`，避免影响单卡默认配置。

1. 安装依赖：

```bash
pip install ray
```

2. 直接使用 `configs/default_ray_dual_gpu.yaml`（核心配置如下）：

```yaml
finetune:
  eval_split_ratio: 0.1  # 从训练集切出10%只用于eval loss，不参与训练
  eval_steps: 100        # 每100 step计算一次eval loss

evaluation:
  enable_during_training: true
  async_backend: ray
  every_n_steps: 100
  ray:
    address: null
    eval_device: cuda:1
    max_pending_jobs: 1
```

3. 启动训练：

```bash
python scripts/03_train_lora.py --config configs/default_ray_dual_gpu.yaml
```

运行后日志会出现：
- 提交异步任务：`[boxed-eval][ray] submitted step=...`
- 评估完成结果：`[boxed-eval][ray] step=... question_acc=... seed_acc=... combined_acc=...`
- 训练开始前基线评估：`[boxed-eval] init_step=0 ...`（Ray 模式会提交 `step=0` 的异步评估任务）

## 如何启动 TPE 参数搜索

1. 启动命令（二选一）：

```bash
python scripts/05_tpe_search.py --config configs/default.yaml --n-trials 30
```

```bash
make tpe-search
```

2. 当前搜索设置（来自 `configs/default.yaml`）：
- 搜索算法：`hparam_search.method = tpe`
- 搜索参数：仅 `learning_rate`
- 范围：`1e-6 ~ 1e-3`（log-uniform）
- 每个 trial 训练步数：`hparam_search.trial_max_steps = 199`
- trial 打分方式：训练到 199 step 后，基于 boxed accuracy 评估

3. 搜参评估逻辑：
- 数据集：`data/eval/valid_800.jsonl`
- 每次评估随机抽样：`evaluation.sample_size = 50`
- 输入模型：`question`
- 每条样本会评测两个问题：`question` 与 `seed_question`，分别和 `answer` / `seed_answer` 比较 `\boxed{}` 结果
- 在 TPE 搜参模式下，会关闭训练中每 100 step 的自动评估，只在 trial 结束时评估一次

4. 搜参输出文件：
- `outputs/hparam_search/leaderboard.csv`
- `outputs/hparam_search/best_config.yaml`
- `outputs/hparam_search/summary.md`
- `outputs/hparam_search/trial_xxxx/trial_report.json`
- `outputs/hparam_search/lr_vs_accuracy.png`（可视化图）

5. 对照验证（固定学习率，和搜参同流程）：

```bash
python scripts/05_tpe_search.py --config configs/default.yaml --fixed-lr 2e-4 --debug-single-trial
```

- 用途：排查“value 为 0”是超参问题还是流程问题
- 行为：固定 `learning_rate`、只跑 1 个 trial、其余逻辑与 TPE trial 一致

6. 可视化学习率与结果关系：

```bash
python scripts/06_plot_tpe_lr_vs_accuracy.py --leaderboard outputs/hparam_search/leaderboard.csv --output outputs/hparam_search/lr_vs_accuracy.png
```

或：

```bash
make plot-tpe
```

6. 测试集评估（boxed 准确率）：
- 测试集：`data/eval/valid_800.jsonl`
- 评测方式：每次随机抽取 50 条样本；每条样本依次评测 `question` 与 `seed_question` 两个问题，分别统计准确率并给出合并准确率
- 训练中会按 `evaluation.every_n_steps`（默认 100）自动执行一次并打印结果，例如：
  - `[boxed-eval] step=100 sampled=50 question_acc=0.xxxx seed_acc=0.xxxx combined_acc=0.xxxx`
- 训练开始前会先做一轮 step=0 的 boxed 评估，作为基线。
- 若启用 `evaluation.async_backend: ray`，评估会异步在 `evaluation.ray.eval_device` 指定 GPU 上运行，训练不会等待评估结束。
- 若 `finetune.eval_split_ratio > 0`（例如 0.1），会从训练数据中切出对应比例作为 `eval_dataset`，该部分不参与梯度更新；Trainer 会按 `finetune.eval_steps` 输出 `eval_loss`，可用于观察过拟合趋势。
- 在 TPE 搜参模式下会关闭上述“每100步评估”，统一改为训练到 199 step 后仅评估一次作为 trial 分数。

```bash
python scripts/04_eval_boxed_accuracy.py --config configs/default.yaml
```

评估结果会保存到：
- `outputs/qwen3-0.6b-lora/boxed_eval_report.json`

## 数据流说明

1. `data/raw/` 原始输入。
2. `scripts/00_inspect_data.py` 做数据概览，输出统计到 `outputs/`。
3. `scripts/01_filter_data.py` 做基础过滤，输出到 `data/interim/` 与 `data/processed/`。
4. `scripts/02_split_eval_test.py` 将 `data/raw/valid_1000.jsonl` 按固定随机种子切分为 `data/eval/valid_800.jsonl` 与 `data/test/valid_200.jsonl`。
5. `scripts/03_train_lora.py` 读取 `data/sft/sft_boxed_small.json`，使用 PEFT 对 `models/Qwen3-0.6B-Base` 进行 LoRA 微调，并将日志上报到 W&B。
6. `scripts/04_eval_boxed_accuracy.py` 使用 `data/eval/valid_800.jsonl` 作为测试集，每次随机抽样 50 条；每条样本评测 `question` 和 `seed_question`，输出 question/seed/combined 三个准确率。
7. LoRA 适配器权重、训练指标和评测结果默认输出到 `outputs/qwen3-0.6b-lora/`。

## 输出文件说明

每次实验建议在 `outputs/<run_name_or_timestamp>/` 下至少包含：

- `config.yaml`：本次实验配置副本
- `command.txt`：运行命令
- `metrics.json`：核心指标
- `samples.jsonl`：样例输出
- `log.txt`：运行日志

## 常见注意事项

- 可复用逻辑写入 `src/`，避免散落在 `scripts/`。
- `scripts/` 每个脚本只负责一个明确步骤，并保留 `main()` 入口。
- 参数尽量走 `configs/default.yaml`，不要硬编码路径和模型名。
- 双卡 Ray 并行评估请使用 `configs/default_ray_dual_gpu.yaml`，不要直接改 `configs/default.yaml`。
- W&B 的 `API Key/Project/Entity/Run Name` 放在 `.env`，代码里通过 `python-dotenv` 自动加载。
- 临时调试代码放在 `notebooks/`、`tests/` 或临时文件，不要混进正式脚本。
- 数据构建阶段若样本长度超过 `finetune.max_seq_length`，会打印 `[dataset-truncate] ... truncated_tokens=...`；未截断样本不输出。
- 训练日志新增 `avg_loss_50`（最近 50 个 loss 的滑动平均）用于更稳定地观察 loss 趋势。
