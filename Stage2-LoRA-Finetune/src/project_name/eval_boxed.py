from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvalItem:
    question: str
    gold_answer: str
    seed_question: str
    seed_answer: str


def load_eval_items(path: str) -> list[EvalItem]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    items: list[EvalItem] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            seed_question = str(row.get("seed_question", "")).strip()
            seed_answer = str(row.get("seed_answer", "")).strip()
            if not question:
                continue
            items.append(
                EvalItem(
                    question=question,
                    gold_answer=answer,
                    seed_question=seed_question,
                    seed_answer=seed_answer,
                )
            )
    if not items:
        raise ValueError(f"No valid eval items in: {path}")
    return items


def extract_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def _normalize_text(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace("$", "").replace("\\,", "")
    ans = re.sub(r"\s+", "", ans)
    ans = ans.rstrip(".")
    return ans


def _to_decimal(s: str) -> Decimal | None:
    s = s.strip()
    if not s:
        return None
    if "/" in s and not s.startswith("http"):
        parts = s.split("/")
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            try:
                denominator = Decimal(right)
                if denominator == 0:
                    return None
                return Decimal(left) / denominator
            except InvalidOperation:
                return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    pred_norm = _normalize_text(pred)
    gold_norm = _normalize_text(gold)

    if pred_norm == gold_norm:
        return True

    pred_num = _to_decimal(pred_norm)
    gold_num = _to_decimal(gold_norm)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num

    return False


def sample_eval_items(items: list[EvalItem], sample_size: int, seed: int) -> list[EvalItem]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if sample_size > len(items):
        raise ValueError(f"sample_size={sample_size} exceeds dataset size={len(items)}")
    rng = random.Random(seed)
    return rng.sample(items, sample_size)


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: str | None,
    dtype: str | None = "auto",
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map.get(str(dtype).lower(), "auto")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=resolved_dtype,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    if device:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> str:
    messages: list[dict[str, str]] = [
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **gen_kwargs)

    generated_ids = output_ids[0][model_inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_boxed_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: list[EvalItem],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    question_correct = 0
    seed_correct = 0
    seed_total = 0

    for idx, item in enumerate(items):
        question_response = generate_response(
            model=model,
            tokenizer=tokenizer,
            question=item.question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        question_pred_boxed = extract_boxed_content(question_response)
        question_is_correct = answers_match(question_pred_boxed, item.gold_answer)
        question_correct += int(question_is_correct)

        seed_response = None
        seed_pred_boxed = None
        seed_is_correct = None
        if item.seed_question:
            seed_total += 1
            seed_response = generate_response(
                model=model,
                tokenizer=tokenizer,
                question=item.seed_question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            seed_pred_boxed = extract_boxed_content(seed_response)
            seed_is_correct = answers_match(seed_pred_boxed, item.seed_answer)
            seed_correct += int(seed_is_correct)

        results.append(
            {
                "index": idx,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "question_model_response": question_response,
                "question_pred_boxed": question_pred_boxed,
                "question_is_correct": question_is_correct,
                "seed_question": item.seed_question,
                "seed_answer": item.seed_answer,
                "seed_model_response": seed_response,
                "seed_pred_boxed": seed_pred_boxed,
                "seed_is_correct": seed_is_correct,
            }
        )

    question_total = len(items)
    question_accuracy = question_correct / question_total if question_total else 0.0
    seed_accuracy = seed_correct / seed_total if seed_total else 0.0
    combined_total = question_total + seed_total
    combined_correct = question_correct + seed_correct
    combined_accuracy = combined_correct / combined_total if combined_total else 0.0
    return {
        "question_total": question_total,
        "question_correct": question_correct,
        "question_accuracy": question_accuracy,
        "seed_total": seed_total,
        "seed_correct": seed_correct,
        "seed_accuracy": seed_accuracy,
        "combined_total": combined_total,
        "combined_correct": combined_correct,
        "combined_accuracy": combined_accuracy,
        # Keep compatibility with existing downstream printing.
        "total": combined_total,
        "correct": combined_correct,
        "accuracy": combined_accuracy,
        "results": results,
    }

