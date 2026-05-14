from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype_name = dtype.lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
    if not matches:
        return ""
    return matches[-1].strip()


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _compute_reward(generated_text: str, target_boxed: str) -> float:
    pred_boxed = _extract_boxed_answer(generated_text)
    if not pred_boxed:
        return 0.0
    pred_norm = _normalize_answer(pred_boxed)
    target_norm = _normalize_answer(target_boxed)
    return 1.0 if pred_norm == target_norm else 0.0


@dataclass
class TrainSample:
    prompt: str
    target_boxed: str


def _build_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    chunks: list[str] = []
    for item in messages:
        role = item.get("role", "user")
        content = item.get("content", "")
        chunks.append(f"<|{role}|>\n{content}")
    chunks.append("<|assistant|>\n")
    return "\n".join(chunks)


def load_grpo_dataset(dataset_path: str, tokenizer: AutoTokenizer) -> list[TrainSample]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON list")

    items: list[TrainSample] = []
    for row in records:
        messages = row.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_msgs:
            continue
        target_boxed = _extract_boxed_answer(str(assistant_msgs[-1].get("content", "")))
        if not target_boxed:
            continue
        prompt = _build_prompt(tokenizer, messages[:-1])
        items.append(TrainSample(prompt=prompt, target_boxed=target_boxed))
    if not items:
        raise ValueError("No usable training rows found after preprocessing.")
    return items


def _sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    seq_len = labels.size(1)
    positions = torch.arange(seq_len, device=labels.device).unsqueeze(0)
    completion_mask = (positions >= (prompt_lengths.unsqueeze(1) - 1)).float()
    completion_mask = completion_mask * attention_mask[:, 1:].float()
    token_count = completion_mask.sum(dim=1).clamp(min=1.0)
    return (token_log_probs * completion_mask).sum(dim=1) / token_count


def _prepare_batched_sequences(
    prompt_ids_list: list[torch.Tensor],
    completion_ids_list: list[torch.Tensor],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_ids: list[torch.Tensor] = []
    prompt_lens: list[int] = []
    for prompt_ids, completion_ids in zip(prompt_ids_list, completion_ids_list):
        full_ids.append(torch.cat([prompt_ids, completion_ids], dim=0))
        prompt_lens.append(int(prompt_ids.numel()))

    max_len = max(x.numel() for x in full_ids)
    padded_ids: list[torch.Tensor] = []
    padded_mask: list[torch.Tensor] = []
    for ids in full_ids:
        pad_len = max_len - ids.numel()
        if pad_len > 0:
            pad = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
        mask = torch.ones((max_len,), dtype=torch.long)
        if pad_len > 0:
            mask[-pad_len:] = 0
        padded_ids.append(ids)
        padded_mask.append(mask)

    input_ids = torch.stack(padded_ids, dim=0).to(device)
    attention_mask = torch.stack(padded_mask, dim=0).to(device)
    prompt_lengths = torch.tensor(prompt_lens, dtype=torch.long, device=device)
    return input_ids, attention_mask, prompt_lengths


def _save_checkpoint(model: PeftModel, tokenizer: AutoTokenizer, output_dir: Path, step: int) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))


def train_grpo(config: dict[str, Any]) -> None:
    model_cfg = config["model"]
    train_cfg = config["training"]
    grpo_cfg = config["grpo"]

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    random.seed(seed)

    dtype = _resolve_dtype(str(train_cfg.get("dtype", "bf16")))
    model_path = str(model_cfg["base_model_path"])
    adapter_path = str(model_cfg["lora_adapter_path"])

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    policy_base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    policy_model = PeftModel.from_pretrained(policy_base, adapter_path, is_trainable=True)
    policy_model.train()
    policy_model.print_trainable_parameters()

    use_ref = bool(grpo_cfg.get("use_reference_model", False)) and float(grpo_cfg.get("kl_coef", 0.0)) > 0
    ref_model: PeftModel | None = None
    if use_ref:
        ref_base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        ref_model = PeftModel.from_pretrained(ref_base, adapter_path, is_trainable=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    data = load_grpo_dataset(str(model_cfg["dataset_path"]), tokenizer)
    max_samples = train_cfg.get("max_samples")
    if isinstance(max_samples, int) and max_samples > 0:
        data = data[:max_samples]

    if not data:
        raise ValueError("No training samples after max_samples filtering.")

    output_dir = Path(str(train_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log: dict[str, Any] = {"steps": []}

    per_device_batch_size = int(train_cfg["per_device_batch_size"])
    grad_acc_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    lr = float(train_cfg["learning_rate"])
    max_steps = int(train_cfg["max_steps"])
    save_steps = int(train_cfg.get("save_steps", 100))
    log_steps = int(train_cfg.get("logging_steps", 1))

    group_size = int(grpo_cfg["group_size"])
    max_new_tokens = int(grpo_cfg["max_new_tokens"])
    temperature = float(grpo_cfg["temperature"])
    top_p = float(grpo_cfg["top_p"])
    kl_coef = float(grpo_cfg.get("kl_coef", 0.0))

    optim = AdamW(policy_model.parameters(), lr=lr)

    maybe_wandb = None
    if train_cfg.get("use_wandb", True):
        import wandb  # type: ignore

        maybe_wandb = wandb
        run_name = str(train_cfg.get("run_name", "grpo-lora-continue"))
        wandb.init(
            project=str(train_cfg.get("wandb", {}).get("project", "")) or None,
            entity=str(train_cfg.get("wandb", {}).get("entity", "")) or None,
            name=run_name,
            config=config,
        )

    device = next(policy_model.parameters()).device
    global_step = 0
    sample_cursor = 0
    grad_acc_counter = 0
    progress = tqdm(total=max_steps, desc="GRPO training")

    while global_step < max_steps:
        batch = []
        for _ in range(per_device_batch_size):
            batch.append(data[sample_cursor % len(data)])
            sample_cursor += 1

        prompts = [row.prompt for row in batch]
        targets = [row.target_boxed for row in batch]
        expanded_prompts = [p for p in prompts for _ in range(group_size)]
        expanded_targets = [t for t in targets for _ in range(group_size)]

        encoded = tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(train_cfg["max_prompt_length"]),
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_lengths = encoded["attention_mask"].sum(dim=1)

        with torch.no_grad():
            generated = policy_model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_ids_list: list[torch.Tensor] = []
        completion_ids_list: list[torch.Tensor] = []
        decoded_outputs: list[str] = []

        for i in range(generated.size(0)):
            full_ids = generated[i]
            p_len = int(prompt_lengths[i].item())
            prompt_ids = full_ids[:p_len].detach().cpu()
            completion_ids = full_ids[p_len:].detach().cpu()
            if completion_ids.numel() == 0:
                completion_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
            prompt_ids_list.append(prompt_ids)
            completion_ids_list.append(completion_ids)
            decoded_outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True))

        rewards = torch.tensor(
            [_compute_reward(pred, tgt) for pred, tgt in zip(decoded_outputs, expanded_targets)],
            dtype=torch.float32,
            device=device,
        )

        advantages = torch.zeros_like(rewards)
        for i in range(per_device_batch_size):
            s = i * group_size
            e = s + group_size
            group_rewards = rewards[s:e]
            mean = group_rewards.mean()
            std = group_rewards.std(unbiased=False).clamp(min=1e-4)
            advantages[s:e] = (group_rewards - mean) / std

        input_ids, attention_mask, prompt_lens = _prepare_batched_sequences(
            prompt_ids_list=prompt_ids_list,
            completion_ids_list=completion_ids_list,
            pad_token_id=int(tokenizer.pad_token_id),
            device=device,
        )
        policy_logprob = _sequence_logprob(policy_model, input_ids, attention_mask, prompt_lens)

        loss_pg = -(advantages.detach() * policy_logprob).mean()
        loss = loss_pg
        mean_kl = torch.tensor(0.0, device=device)
        if use_ref and ref_model is not None:
            with torch.no_grad():
                ref_logprob = _sequence_logprob(ref_model, input_ids, attention_mask, prompt_lens)
            mean_kl = (policy_logprob - ref_logprob).mean()
            loss = loss + kl_coef * mean_kl

        (loss / grad_acc_steps).backward()
        grad_acc_counter += 1

        if grad_acc_counter >= grad_acc_steps:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), float(train_cfg["max_grad_norm"]))
            optim.step()
            optim.zero_grad(set_to_none=True)
            grad_acc_counter = 0
            global_step += 1
            progress.update(1)

            mean_reward = float(rewards.mean().item())
            reward_std = float(rewards.std(unbiased=False).item())
            metrics = {
                "step": global_step,
                "train/loss": float(loss.item()),
                "train/loss_pg": float(loss_pg.item()),
                "train/reward_mean": mean_reward,
                "train/reward_std": reward_std,
                "train/kl_mean": float(mean_kl.item()),
                "train/adv_mean": float(advantages.mean().item()),
            }
            training_log["steps"].append(metrics)

            if global_step % log_steps == 0:
                print(
                    "[step={}] loss={:.6f} reward_mean={:.4f} reward_std={:.4f} kl={:.6f}".format(
                        global_step,
                        metrics["train/loss"],
                        metrics["train/reward_mean"],
                        metrics["train/reward_std"],
                        metrics["train/kl_mean"],
                    )
                )
            if maybe_wandb is not None:
                maybe_wandb.log(metrics, step=global_step)
            if global_step % save_steps == 0:
                _save_checkpoint(policy_model, tokenizer, output_dir, global_step)

    progress.close()

    policy_model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if maybe_wandb is not None:
        maybe_wandb.finish()

    summary_path = output_dir / "training_summary.json"
    summary = {
        "max_steps": max_steps,
        "num_samples": len(data),
        "group_size": group_size,
        "seed": seed,
        "best_reward_mean": max((x["train/reward_mean"] for x in training_log["steps"]), default=0.0),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "logs": training_log}, f, ensure_ascii=False, indent=2)

    print(f"GRPO training done. Adapter saved to: {output_dir}")
