# Stage2/Stage3 Boxed-Eval 一致性验证

这个仓库根目录下提供了验证脚本：

- `verify_stage2_boxed_eval_with_stage3_lora.py`

它用于复现 Stage2 训练中 boxed-eval 的抽样与评测逻辑，并加载你指定的 LoRA（可来自 `Stage3-Evaluation/lora`）进行打分，帮助判断：

- “训练中每 100 step 的 boxed 准确率” 与
- “离线复算同一批样本的 boxed 准确率”

是否一致。

## 1) 适用场景

当你怀疑以下问题时可使用本脚本：

- Stage2 训练中指标偏高是否来自抽样（64 条）；
- Stage3 全量评估偏低是否来自评测集合变化；
- 训练时评测逻辑是否存在实现错误。

## 2) 脚本做了什么

脚本会：

1. 读取你指定的 Stage2 配置文件（例如 `Stage2-LoRA-Finetune/configs/lr_e-6.yaml`）。
2. 使用其中的 `evaluation.test_data_path`、`sample_size`、`seed` 复现同一组样本。
3. 加载：
   - base model：`finetune.model_path`
   - LoRA adapter：`--lora-path` 或 `--lora-name` 指定
4. 使用 Stage2 的 `evaluate_boxed_accuracy` 计算 `question/seed/combined` 准确率。
5. 可选地将结果保存为 JSON，或与期望准确率做严格比对。

## 3) 基本用法

### 方式 A：按 Stage3 的 LoRA 名称指定

```bash
python verify_stage2_boxed_eval_with_stage3_lora.py \
  --stage2-config Stage2-LoRA-Finetune/configs/lr_e-6.yaml \
  --lora-name lora-e-5-1700steps
```

`--lora-name` 会自动解析到：

- `Stage3-Evaluation/lora/<lora-name>`

### 方式 B：直接指定 LoRA 路径

```bash
python verify_stage2_boxed_eval_with_stage3_lora.py \
  --stage2-config Stage2-LoRA-Finetune/configs/lr_e-6.yaml \
  --lora-path Stage3-Evaluation/lora/lora-e-5-1700steps
```

## 4) 常用增强参数

- `--expected-combined-accuracy`：传入训练时记录的目标准确率，做一致性校验。
- `--atol`：比较容差（默认 `1e-12`）。
- `--output-json`：输出完整报告到指定路径。
- `--device`：手动指定设备（例如 `cuda:1`）。
- `--dtype`：手动指定 dtype（`auto`/`float16`/`bfloat16`/`float32`）。
- `--sample-size`：覆盖 config 内样本数。
- `--seed`：覆盖 config 内随机种子。

示例（带严格比对 + 输出）：

```bash
python verify_stage2_boxed_eval_with_stage3_lora.py \
  --stage2-config Stage2-LoRA-Finetune/configs/lr_e-6.yaml \
  --lora-name lora-e-5-1700steps \
  --expected-combined-accuracy 0.609375 \
  --atol 1e-12 \
  --output-json outputs/verify_stage2_boxed_eval.json
```

## 5) 如何判断“逻辑一致”

若要和训练过程某一步的 boxed-eval **完全一致**，请确保以下条件一致：

- 使用同一步对应的 LoRA 权重；
- `sample_size`、`seed`、`test_data_path` 一致；
- 推理参数一致（`max_new_tokens`、`temperature`、`do_sample`、`batch_size`）；
- 设备/精度设置尽量一致（`device`、`dtype`）。

脚本会打印 `sample_fingerprint`（抽样题目哈希），可快速确认抽样集合是否一致。

## 6) 常见问题

- `adapter_config.json not found`  
  说明传入的 LoRA 路径不是完整 adapter 目录。

- 结果与训练时有微小偏差  
  先检查 `device/dtype` 与 LoRA 步数是否一致，再检查是否真的使用了同一 `seed + sample_size + test_data_path`。
