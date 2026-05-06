from __future__ import annotations

import json
import os
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainerControl,
    Trainer,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from src.project_name.eval_boxed import (
    EvalItem,
    evaluate_boxed_accuracy,
    load_eval_items,
    load_model_and_tokenizer,
    sample_eval_items,
)

try:
    import ray
except ImportError:
    ray = None


def _to_int(value: Any) -> int:
    return int(value)


def _to_float(value: Any) -> float:
    return float(value)


def _resolve_output_dir(ft_cfg: dict[str, Any]) -> str:
    """Append a wall-clock timestamp so repeated runs do not overwrite adapters."""
    base = Path(str(ft_cfg["output_dir"]))
    if not bool(ft_cfg.get("timestamp_output_dir", True)):
        return str(base)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resolved = base.parent / f"{base.name}-{stamp}"
    out = str(resolved)
    print(f"[output-dir] {base} -> {out}")
    return out


def _evaluate_checkpoint_boxed_accuracy(
    *,
    step: int,
    model_path: str,
    adapter_path: str,
    dtype: str,
    test_data_path: str,
    sample_size: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
    report_dir: str,
) -> dict[str, Any]:
    items = load_eval_items(test_data_path)
    sampled = sample_eval_items(items=items, sample_size=sample_size, seed=seed + step)
    requested_device = device
    candidate_devices: list[str] = [requested_device]
    if requested_device.startswith("cuda:") and requested_device != "cuda:0":
        candidate_devices.append("cuda:0")

    model = None
    tokenizer = None
    actual_device = requested_device
    last_error: Exception | None = None

    for candidate in candidate_devices:
        try:
            mapped_device = candidate
            # Ray task may hide all accelerators for num_gpus=0. Make the target
            # physical GPU visible, then use local cuda:0 in this subprocess.
            if candidate.startswith("cuda:"):
                gpu_idx = int(candidate.split(":", maxsplit=1)[1])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                mapped_device = "cuda:0"

            model, tokenizer = load_model_and_tokenizer(
                model_path=model_path,
                adapter_path=adapter_path,
                dtype=dtype,
                device=mapped_device,
            )
            actual_device = candidate
            if candidate != requested_device:
                print(
                    f"[boxed-eval][ray] fallback device applied: "
                    f"requested={requested_device} actual={actual_device}"
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if model is None or tokenizer is None:
        raise RuntimeError(
            f"Failed to load eval model on devices {candidate_devices}; "
            f"last_error={last_error}"
        )

    with torch.no_grad():
        report = evaluate_boxed_accuracy(
            model=model,
            tokenizer=tokenizer,
            items=sampled,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
    del model
    if actual_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"boxed_eval_step_{step}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "step": step,
        "question_accuracy": report["question_accuracy"],
        "seed_accuracy": report["seed_accuracy"],
        "combined_accuracy": report["combined_accuracy"],
        "question_total": report["question_total"],
        "seed_total": report["seed_total"],
        "combined_total": report["combined_total"],
        "report_path": str(out_file),
        "adapter_path": adapter_path,
        "eval_device_used": actual_device,
    }


class BoxedEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer: AutoTokenizer,
        eval_items: list[EvalItem],
        sample_size: int,
        every_n_steps: int,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_items = eval_items
        self.sample_size = sample_size
        self.every_n_steps = every_n_steps
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    def _run_boxed_eval(
        self,
        *,
        model: AutoModelForCausalLM,
        step_seed: int,
        step_label: str,
        step_value: int,
    ) -> None:
        sampled = sample_eval_items(
            items=self.eval_items,
            sample_size=self.sample_size,
            seed=step_seed,
        )

        was_training = model.training
        model.eval()
        with torch.no_grad():
            report = evaluate_boxed_accuracy(
                model=model,
                tokenizer=self.tokenizer,
                items=sampled,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
            )
        if was_training:
            model.train()

        print(
            f"[boxed-eval] {step_label}={step_value} "
            f"sampled={len(sampled)} question_acc={report['question_accuracy']:.4f} "
            f"seed_acc={report['seed_accuracy']:.4f} "
            f"combined_acc={report['combined_accuracy']:.4f}"
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if model is None or state.global_step <= 0:
            return control
        if state.global_step % self.every_n_steps != 0:
            return control
        if args.process_index != 0:
            return control

        self._run_boxed_eval(
            model=model,
            step_seed=self.seed + int(state.global_step),
            step_label="step",
            step_value=int(state.global_step),
        )
        return control

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if model is None:
            return control
        if args.process_index != 0:
            return control
        self._run_boxed_eval(
            model=model,
            step_seed=self.seed,
            step_label="init_step",
            step_value=0,
        )
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if model is None:
            return control
        if args.process_index != 0:
            return control
        self._run_boxed_eval(
            model=model,
            step_seed=self.seed + int(state.global_step) + 99991,
            step_label="train_end_step",
            step_value=int(state.global_step),
        )
        return control


class AsyncRayBoxedEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        model_path: str,
        dtype: str,
        output_dir: str,
        test_data_path: str,
        sample_size: int,
        every_n_steps: int,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        eval_device: str,
        max_pending_jobs: int,
        shutdown_ray_on_end: bool,
    ) -> None:
        if ray is None:
            raise ImportError("ray is required for async Ray evaluation. Install with: pip install ray")
        self.model_path = model_path
        self.dtype = dtype
        self.output_dir = Path(output_dir)
        self.test_data_path = test_data_path
        self.sample_size = sample_size
        self.every_n_steps = every_n_steps
        self.seed = seed
        self.max_new_jobs = max(1, max_pending_jobs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.eval_device = eval_device
        self.shutdown_ray_on_end = shutdown_ray_on_end
        self.snapshot_root = self.output_dir / "async_eval_snapshots"
        self.report_root = self.output_dir / "boxed_eval_reports"
        self.pending_jobs: list[tuple[Any, int, Path]] = []
        self.remote_eval = ray.remote(num_cpus=1)(_evaluate_checkpoint_boxed_accuracy)

    def _collect_finished_jobs(self) -> None:
        if not self.pending_jobs:
            return
        pending_refs = [job[0] for job in self.pending_jobs]
        ready_refs, _ = ray.wait(
            pending_refs,
            num_returns=len(pending_refs),
            timeout=0,
        )
        if not ready_refs:
            return
        ready_set = set(ready_refs)
        remaining: list[tuple[Any, int, Path]] = []
        for ref, step, snapshot_dir in self.pending_jobs:
            if ref not in ready_set:
                remaining.append((ref, step, snapshot_dir))
                continue
            try:
                result = ray.get(ref)
                print(
                    f"[boxed-eval][ray] step={result['step']} "
                    f"question_acc={result['question_accuracy']:.4f} "
                    f"seed_acc={result['seed_accuracy']:.4f} "
                    f"combined_acc={result['combined_accuracy']:.4f} "
                    f"report={result['report_path']}"
                )
            except Exception as exc:
                print(f"[boxed-eval][ray] step={step} failed: {exc}")
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        self.pending_jobs = remaining

    def _submit_job(self, *, model: AutoModelForCausalLM, step: int) -> None:
        if len(self.pending_jobs) >= self.max_new_jobs:
            print(
                f"[boxed-eval][ray] skip step={step}: "
                f"pending_jobs={len(self.pending_jobs)} reached limit={self.max_new_jobs}"
            )
            return
        snapshot_dir = self.snapshot_root / f"step-{step}"
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(snapshot_dir))
        ref = self.remote_eval.remote(
            step=step,
            model_path=self.model_path,
            adapter_path=str(snapshot_dir),
            dtype=self.dtype,
            test_data_path=self.test_data_path,
            sample_size=self.sample_size,
            seed=self.seed,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            device=self.eval_device,
            report_dir=str(self.report_root),
        )
        self.pending_jobs.append((ref, step, snapshot_dir))
        print(
            f"[boxed-eval][ray] submitted step={step} "
            f"pending_jobs={len(self.pending_jobs)} snapshot={snapshot_dir}"
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if model is None or state.global_step <= 0:
            return control
        if args.process_index != 0:
            return control
        self._collect_finished_jobs()
        step = int(state.global_step)
        if step % self.every_n_steps != 0:
            return control
        self._submit_job(model=model, step=step)
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if args.process_index == 0:
            self._collect_finished_jobs()
        return control

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: AutoModelForCausalLM | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if model is None:
            return control
        if args.process_index != 0:
            return control
        self._collect_finished_jobs()
        self._submit_job(model=model, step=0)
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if args.process_index != 0:
            return control
        while self.pending_jobs:
            ref, step, snapshot_dir = self.pending_jobs.pop(0)
            try:
                result = ray.get(ref)
                print(
                    f"[boxed-eval][ray][finalized] step={result['step']} "
                    f"question_acc={result['question_accuracy']:.4f} "
                    f"seed_acc={result['seed_accuracy']:.4f} "
                    f"combined_acc={result['combined_accuracy']:.4f} "
                    f"report={result['report_path']}"
                )
            except Exception as exc:
                print(f"[boxed-eval][ray][finalized] step={step} failed: {exc}")
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        if self.shutdown_ray_on_end and ray.is_initialized():
            ray.shutdown()
        return control


class AvgLossCallback(TrainerCallback):
    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.loss_window: deque[float] = deque(maxlen=window_size)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if args.process_index != 0:
            return control
        if not logs:
            return control
        loss_value = logs.get("loss")
        if loss_value is None:
            return control
        loss_float = float(loss_value)
        self.loss_window.append(loss_float)
        avg_loss = sum(self.loss_window) / len(self.loss_window)
        logs["avg_loss_50"] = avg_loss
        print(
            f"[train-metric] step={int(state.global_step)} "
            f"loss={loss_float:.6f} avg_loss_50={avg_loss:.6f} "
            f"window={len(self.loss_window)}"
        )
        return control


def load_sft_json(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"SFT dataset not found: {path}")
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("SFT dataset must be a JSON list.")
    return data


def _extract_prompt_and_answer(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    last_assistant_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            last_assistant_idx = idx
            break

    if last_assistant_idx is None:
        raise ValueError("Sample has no assistant message.")

    prompt_messages = messages[:last_assistant_idx]
    answer = messages[last_assistant_idx].get("content", "")
    if not prompt_messages:
        raise ValueError("Sample has no prompt messages before assistant.")
    return prompt_messages, answer


def build_tokenized_dataset(
    raw_data: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dataset:
    features: list[dict[str, list[int]]] = []

    for sample_idx, sample in enumerate(raw_data):
        messages = sample.get("messages")
        if not isinstance(messages, list):
            continue

        try:
            prompt_messages, answer = _extract_prompt_and_answer(messages)
        except ValueError:
            continue

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        answer_text = answer + (tokenizer.eos_token or "")

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids

        original_len = len(input_ids)
        if original_len > max_seq_length:
            truncated_tokens = original_len - max_seq_length
            print(
                f"[dataset-truncate] sample_idx={sample_idx} "
                f"original_tokens={original_len} max_seq_length={max_seq_length} "
                f"truncated_tokens={truncated_tokens}"
            )
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]

        attention_mask = [1] * len(input_ids)
        if not input_ids:
            continue

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    if not features:
        raise ValueError("No valid training samples were built from dataset.")

    return Dataset.from_list(features)


def create_trainer(config: dict[str, Any]) -> tuple[Trainer, AutoTokenizer]:
    seed = config["project"]["seed"]
    set_seed(seed)

    ft_cfg = config["finetune"]
    ft_cfg["output_dir"] = _resolve_output_dir(ft_cfg)
    lora_cfg = ft_cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(ft_cfg["model_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_data = load_sft_json(ft_cfg["dataset_path"])
    tokenized = build_tokenized_dataset(
        raw_data=raw_data,
        tokenizer=tokenizer,
        max_seq_length=ft_cfg["max_seq_length"],
    )

    eval_ratio = float(ft_cfg.get("loss_eval_split_ratio", ft_cfg.get("eval_split_ratio", 0.0)))
    if not (0.0 <= eval_ratio < 1.0):
        raise ValueError(
            f"loss_eval_split_ratio/eval_split_ratio must be in [0, 1), got: {eval_ratio}"
        )
    if eval_ratio > 0:
        split = tokenized.train_test_split(test_size=eval_ratio, seed=seed, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(
            f"[loss-eval] enabled holdout_ratio={eval_ratio:.3f} "
            f"train_samples={len(train_dataset)} eval_samples={len(eval_dataset)} "
            f"eval_steps={_to_int(ft_cfg.get('eval_steps', 100))}"
        )
    else:
        train_dataset = tokenized
        eval_dataset = None

    model_dtype = ft_cfg.get("dtype", ft_cfg.get("torch_dtype", "auto"))
    model = AutoModelForCausalLM.from_pretrained(
        ft_cfg["model_path"],
        dtype=model_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(model, peft_config)

    report_to = ["wandb"] if ft_cfg.get("use_wandb", True) else []
    training_args = TrainingArguments(
        output_dir=ft_cfg["output_dir"],
        per_device_train_batch_size=_to_int(ft_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=_to_int(ft_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=_to_int(ft_cfg["gradient_accumulation_steps"]),
        learning_rate=_to_float(ft_cfg["learning_rate"]),
        num_train_epochs=_to_float(ft_cfg["num_train_epochs"]),
        lr_scheduler_type=ft_cfg["lr_scheduler_type"],
        warmup_ratio=_to_float(ft_cfg["warmup_ratio"]),
        logging_steps=_to_int(ft_cfg["logging_steps"]),
        save_steps=_to_int(ft_cfg["save_steps"]),
        save_total_limit=_to_int(ft_cfg["save_total_limit"]),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=_to_int(ft_cfg["eval_steps"]) if eval_dataset is not None else None,
        bf16=ft_cfg.get("bf16", False),
        fp16=ft_cfg.get("fp16", False),
        gradient_checkpointing=ft_cfg.get("gradient_checkpointing", True),
        report_to=report_to,
        run_name=ft_cfg.get("run_name"),
        dataloader_num_workers=_to_int(ft_cfg.get("dataloader_num_workers", 0)),
        remove_unused_columns=False,
        max_steps=_to_int(ft_cfg["max_steps"]) if ft_cfg.get("max_steps") is not None else -1,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    callbacks: list[TrainerCallback] = []
    callbacks.append(AvgLossCallback(window_size=50))
    eval_cfg = config.get("evaluation", {})
    test_data_path = eval_cfg.get("test_data_path")
    enable_train_time_boxed_eval = bool(eval_cfg.get("enable_during_training", True))
    async_backend = str(eval_cfg.get("async_backend", "none")).lower()
    if test_data_path and enable_train_time_boxed_eval:
        if async_backend == "ray":
            if ray is None:
                raise ImportError(
                    "evaluation.async_backend=ray requires ray. Install with: pip install ray"
                )
            ray_cfg = eval_cfg.get("ray", {})
            started_ray = False
            if not ray.is_initialized():
                ray.init(
                    address=ray_cfg.get("address"),
                    ignore_reinit_error=True,
                    include_dashboard=False,
                )
                started_ray = True
            callbacks.append(
                AsyncRayBoxedEvalCallback(
                    model_path=ft_cfg["model_path"],
                    dtype=str(ft_cfg.get("dtype", ft_cfg.get("torch_dtype", "auto"))),
                    output_dir=ft_cfg["output_dir"],
                    test_data_path=str(test_data_path),
                    sample_size=_to_int(eval_cfg.get("sample_size", 30)),
                    every_n_steps=_to_int(eval_cfg.get("every_n_steps", ft_cfg.get("eval_steps", 100))),
                    seed=_to_int(eval_cfg.get("seed", seed)),
                    max_new_tokens=_to_int(eval_cfg.get("max_new_tokens", 256)),
                    temperature=_to_float(eval_cfg.get("temperature", 0.0)),
                    do_sample=bool(eval_cfg.get("do_sample", False)),
                    eval_device=str(ray_cfg.get("eval_device", "cuda:1")),
                    max_pending_jobs=_to_int(ray_cfg.get("max_pending_jobs", 1)),
                    shutdown_ray_on_end=started_ray,
                )
            )
        else:
            eval_items = load_eval_items(test_data_path)
            callbacks.append(
                BoxedEvalCallback(
                    tokenizer=tokenizer,
                    eval_items=eval_items,
                    sample_size=_to_int(eval_cfg.get("sample_size", 30)),
                    every_n_steps=_to_int(eval_cfg.get("every_n_steps", ft_cfg.get("eval_steps", 100))),
                    seed=_to_int(eval_cfg.get("seed", seed)),
                    max_new_tokens=_to_int(eval_cfg.get("max_new_tokens", 256)),
                    temperature=_to_float(eval_cfg.get("temperature", 0.0)),
                    do_sample=bool(eval_cfg.get("do_sample", False)),
                )
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    return trainer, tokenizer

