# Stage3 Evaluation

Stage3 evaluates LoRA adapters produced in Stage2 against the eval dataset using the same boxed-accuracy rule (`question / seed / combined`).

## Project Goal

- Compose a deployable model from `models/` base model + `lora/` adapter.
- Start a local vLLM OpenAI-compatible service for that composed model.
- Evaluate the full eval file in `data/eval/` and report boxed accuracy.

## Directory Structure

- `configs/`: runtime parameters for vLLM service and evaluation.
- `data/eval/`: evaluation dataset (`jsonl` with `question/answer/seed_question/seed_answer`).
- `lora/`: LoRA adapter files (`adapter_config.json`, `adapter_model.safetensors`).
- `models/`: base model files.
- `src/project_name/`: reusable config, boxed-eval logic, vLLM client.
- `scripts/`: CLI entrypoints for serving and evaluating.
- `evaluation/`: per-run outputs (`metrics.json`, `samples.jsonl`, `config.yaml`, `command.txt`).
- `logs/`, `outputs/`: runtime logs and extra artifacts.

## Environment Setup

```bash
conda create -n llm_project python=3.11
conda activate llm_project
pip install -r requirements.txt
pip install vllm
```

## Pipeline Commands

### 1) Start vLLM service (base model + LoRA)

```bash
python scripts/00_serve_vllm.py --config configs/default.yaml
```

Dry-run (print exact vLLM startup command):

```bash
python scripts/00_serve_vllm.py --config configs/default.yaml --dry-run
```

### 2) Evaluate boxed accuracy on full eval file

```bash
python scripts/01_eval_boxed_accuracy.py --config configs/default.yaml
```

## Data Flow

1. `00_serve_vllm.py` reads config and launches vLLM with:
   - `--model <base_model_path>`
   - `--enable-lora --lora-modules <alias>=<lora_path>`
2. `01_eval_boxed_accuracy.py` sends concurrent eval prompts to `/v1/chat/completions`.
3. Responses are parsed with boxed extraction and answer matching.
4. Final metrics and row-level samples are persisted under `evaluation/boxed-eval-<timestamp>/`.

## Output Files

Each run directory under `evaluation/` contains:

- `config.yaml`: config snapshot used in this run.
- `command.txt`: exact command used to run the script.
- `metrics.json`: aggregate `question/seed/combined` accuracy.
- `samples.jsonl`: row-level model outputs and correctness flags.

## Common Notes

- Start vLLM service first, then run evaluation.
- Ensure `model.served_model_name` in config matches the serving side.
- Throughput knobs: `vllm.tensor_parallel_size` and `evaluation.concurrent_requests`.
- If you change code or config defaults, update this README accordingly.

