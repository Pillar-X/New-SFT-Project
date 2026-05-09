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

ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"


def _boxed_eval_msg(message: str) -> str:
    return f"{ANSI_BLUE}{message}{ANSI_RESET}"


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
    # Decoder-only generation with padding is typically more stable with left padding.
    tokenizer.padding_side = "left"

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
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        questions=[question],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    return responses[0]


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[str],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> list[str]:
    if not questions:
        return []
    messages: list[dict[str, str]] = [
        {"role": "user", "content": ""},
    ]
    prompt_texts: list[str] = []
    for q in questions:
        messages[0]["content"] = q
        prompt_texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    model_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, **gen_kwargs)

    input_lens = model_inputs["attention_mask"].sum(dim=1).tolist()
    decoded: list[str] = []
    for i, out_ids in enumerate(output_ids):
        generated_ids = out_ids[int(input_lens[i]) :]
        decoded.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return decoded


def evaluate_boxed_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: list[EvalItem],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    *,
    eval_batch_size: int = 1,
    progress_log_every: int | None = None,
    progress_label: str = "",
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    question_correct = 0
    seed_correct = 0
    seed_total = 0

    bs = max(1, int(eval_batch_size))
    for start in range(0, len(items), bs):
        chunk = items[start : start + bs]
        question_responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            questions=[x.question for x in chunk],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        seed_positions: list[int] = []
        seed_questions: list[str] = []
        for local_idx, item in enumerate(chunk):
            if item.seed_question:
                seed_positions.append(local_idx)
                seed_questions.append(item.seed_question)
                seed_total += 1

        seed_response_by_pos: dict[int, str] = {}
        if seed_questions:
            seed_responses = generate_responses(
                model=model,
                tokenizer=tokenizer,
                questions=seed_questions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            seed_response_by_pos = {
                pos: response for pos, response in zip(seed_positions, seed_responses)
            }

        for local_idx, item in enumerate(chunk):
            idx = start + local_idx
            question_response = question_responses[local_idx]
            question_pred_boxed = extract_boxed_content(question_response)
            question_is_correct = answers_match(question_pred_boxed, item.gold_answer)
            question_correct += int(question_is_correct)

            seed_response = seed_response_by_pos.get(local_idx)
            seed_pred_boxed = extract_boxed_content(seed_response) if seed_response else None
            seed_is_correct = (
                answers_match(seed_pred_boxed, item.seed_answer) if seed_response is not None else None
            )
            if seed_is_correct is not None:
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

        if progress_log_every is not None and progress_log_every > 0:
            done_rows = min(start + len(chunk), len(items))
            at_interval = done_rows % progress_log_every == 0
            at_end = done_rows == len(items) and (len(items) % progress_log_every != 0)
            if at_interval or at_end:
                n_rows = done_rows
                run_q = question_correct / n_rows if n_rows else 0.0
                run_s = seed_correct / seed_total if seed_total else 0.0
                denom = n_rows + seed_total
                run_c = (question_correct + seed_correct) / denom if denom else 0.0
                tag = f" {progress_label}" if progress_label else ""
                print(
                    _boxed_eval_msg(
                        f"[boxed-eval]{tag} progress rows={n_rows}/{len(items)} "
                        f"running_q_acc={run_q:.4f} running_seed_acc={run_s:.4f} "
                        f"running_combined_acc={run_c:.4f}"
                    ),
                    flush=True,
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

