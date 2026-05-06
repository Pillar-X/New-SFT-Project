# Stage2 LoRA Hyperparameter Search Plan (TPE)

## 1) Objective

Use TPE (Tree-structured Parzen Estimator) to search for better finetuning hyperparameters for:

- Model: `models/Qwen3-0.6B-Base`
- Train set: `data/sft/sft_boxed_small.json`
- Trial score eval: boxed accuracy (random 30 samples from `data/eval/valid_800.jsonl`) measured after step 199

Primary objective:

- Maximize boxed accuracy

Secondary objective:

- Maintain stable training (no NaN/Inf, no strong loss divergence)

## 2) Current Scope (This Round)

This round only searches **one** hyperparameter:

1. `learning_rate`

All other parameters remain fixed to baseline in `configs/default.yaml`.

## 3) Search Space

### `learning_rate` (continuous, log scale)

- Range: `1e-6` to `1e-3`
- Sampling: log-uniform

Reason:

- Covers practical LR magnitudes for LoRA on small models; TPE handles log-scale sensitivity well.

## 4) TPE Study Design

- Sampler: TPE
- Direction: maximize
- Pruner: median-based early pruning (after enough warmup trials)
- Random seed for sampler: `42`

Suggested trial budget:

- Initial exploration: 30 trials
- If variance is high: extend to 50 trials

## 5) Per-Trial Training/Eval Protocol

For each trial:

1. Update only:
   - `finetune.learning_rate`
2. Keep all other config fields unchanged.
3. Train until `max_steps = 199`.
4. During hyperparameter search, disable in-training boxed eval callback (no every-100-step eval).
5. Right after step 199, run boxed evaluation once on random 30 samples.
6. Use this boxed accuracy as the trial score.

## 6) Early Stop / Pruning Rules

A trial can be stopped early when:

- Loss becomes NaN/Inf.
- Training is clearly unstable (repeated exploding gradients).
- Pruner decides the trial is significantly below median trajectory.

Hard fail trials should be recorded with failure reason.

## 7) Selection Rule

Choose best trial by:

1. Highest best boxed accuracy
2. If tied: lower training instability (smoother loss / no gradient spikes)

Then run confirmation:

- Re-run top-1 config with seeds `[42, 43, 44]`
- Still use step 199 + 30-sample boxed accuracy for consistency
- Compare mean and std of boxed accuracy

## 8) Tracking and Output

Each trial should log to W&B:

- `learning_rate`
- boxed accuracy at step 199
- train loss / grad norm / lr curve
- trial id and status (completed/pruned/failed)

Artifacts to produce:

- `outputs/hparam_search/leaderboard.csv`
  - columns: `trial_id`, `learning_rate`, `boxed_acc_step199`, `status`
- `outputs/hparam_search/best_config.yaml`
- `outputs/hparam_search/summary.md`

## 9) Notes

- Keep `evaluation.sample_size=30` during this search round.
- Keep trial stopping rule fixed at `max_steps=199` for fair comparison.
- After selecting best config, run one final larger-sample evaluation (e.g. 200 or full 800) for a more stable estimate.
- If best `learning_rate` is pinned at boundary (`1e-6` or `1e-3`), expand search range in the next round.
