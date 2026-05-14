# Stage4 GRPO

本目录用于在已有 LoRA 权重基础上，使用 GRPO（Group Relative Policy Optimization）继续训练。

当前默认输入：

- 基础模型：`../Stage2-LoRA-Finetune/models/Qwen3-0.6B-Base`
- 初始 LoRA：`../Stage3-Evaluation/lora/lora2-500steps`
- 训练数据：`../Stage2-LoRA-Finetune/data/sft/sft_boxed_small.json`

## 目录结构

```text
Stage4-GRPO/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   └── 00_train_grpo.py
└── src/project_name/
    ├── __init__.py
    ├── config.py
    ├── grpo.py
    └── wandb_util.py
```

## 环境安装

```bash
cd Stage4-GRPO
conda activate llm_project
pip install -r requirements.txt
```

## W&B 配置

训练脚本会加载仓库根目录 `.env`，例如：

```bash
WANDB_API_KEY=xxx
WANDB_PROJECT=stage4-qwen3-grpo
WANDB_ENTITY=your_entity
```

你也可以在 `configs/default.yaml` 中设置：

- `training.use_wandb: true/false`
- `training.wandb_mode: online/offline/disabled`
- `training.wandb.project/entity/name`

## 一键运行

```bash
cd Stage4-GRPO
python scripts/00_train_grpo.py --config configs/default.yaml
```

## 配置说明（核心）

`configs/default.yaml` 分为三块：

- `model`：基础模型、初始 LoRA、训练数据路径
- `training`：学习率、步数、batch、日志与保存
- `grpo`：group size、采样温度、KL 系数等

推荐先用小步数验证：

```yaml
training:
  max_steps: 20
  save_steps: 10
grpo:
  group_size: 2
  max_new_tokens: 64
```

## 训练输出

默认输出目录：`outputs/qwen3-0.6b-grpo/`

- `adapter_config.json`
- `adapter_model.safetensors`
- `checkpoint-<step>/`（按 `save_steps` 保存）
- `training_summary.json`

## GRPO 实现说明

- 每个 prompt 采样 `group_size` 个回答；
- 根据 `\boxed{}` 最终答案与标注答案是否一致计算 reward（0 或 1）；
- 在组内做 reward 标准化，得到优势值；
- 用 `-adv * logprob` 做策略梯度更新；
- 可选参考模型 KL 约束（`grpo.use_reference_model=true` 且 `kl_coef>0` 时启用）。

## 常见问题

1. 显存不足：
   - 降低 `grpo.group_size`
   - 降低 `grpo.max_new_tokens`
   - 降低 `training.max_prompt_length`
   - 开启更大的 `gradient_accumulation_steps`

2. reward 基本为 0：
   - 提高 `temperature` 或增加 `max_new_tokens`
   - 检查数据中的 `\boxed{}` 标注是否完整
   - 先把 `max_steps` 调小排查数据与路径

3. 想关掉 W&B：
   - 设置 `training.use_wandb: false`
   - 或设置 `training.wandb_mode: disabled`
