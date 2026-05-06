from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable


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


def maybe_limit_items(items: list[EvalItem], max_samples: int | None, seed: int) -> list[EvalItem]:
    if max_samples is None:
        return items
    if max_samples <= 0:
        raise ValueError("evaluation.max_samples must be > 0 when provided")
    if max_samples >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, max_samples)


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


def evaluate_boxed_accuracy(
    items: list[EvalItem],
    generate_fn: Callable[[str], str],
    *,
    concurrent_requests: int = 1,
    progress_every_requests: int | None = 100,
    progress_every_rows: int | None = 50,
) -> dict[str, Any]:
    max_workers = max(1, int(concurrent_requests))
    total_requests = len(items) + sum(1 for x in items if x.seed_question)
    response_map: dict[tuple[int, str], str] = {}

    def _run_all_requests() -> None:
        nonlocal response_map
        if max_workers == 1:
            done = 0
            for idx, item in enumerate(items):
                response_map[(idx, "question")] = generate_fn(item.question)
                done += 1
                if progress_every_requests and done % progress_every_requests == 0:
                    print(f"[boxed-eval] request_progress={done}/{total_requests}", flush=True)
                if item.seed_question:
                    response_map[(idx, "seed")] = generate_fn(item.seed_question)
                    done += 1
                    if progress_every_requests and done % progress_every_requests == 0:
                        print(f"[boxed-eval] request_progress={done}/{total_requests}", flush=True)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map: dict[Future[str], tuple[int, str]] = {}
            for idx, item in enumerate(items):
                future_map[executor.submit(generate_fn, item.question)] = (idx, "question")
                if item.seed_question:
                    future_map[executor.submit(generate_fn, item.seed_question)] = (idx, "seed")

            done = 0
            for future in as_completed(future_map):
                idx, req_type = future_map[future]
                try:
                    response_map[(idx, req_type)] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"vLLM request failed at idx={idx}, type={req_type}"
                    ) from exc
                done += 1
                if progress_every_requests and done % progress_every_requests == 0:
                    print(f"[boxed-eval] request_progress={done}/{total_requests}", flush=True)

    _run_all_requests()

    results: list[dict[str, Any]] = []
    question_correct = 0
    seed_correct = 0
    seed_total = 0

    for idx, item in enumerate(items):
        question_response = response_map[(idx, "question")]
        question_pred_boxed = extract_boxed_content(question_response)
        question_is_correct = answers_match(question_pred_boxed, item.gold_answer)
        question_correct += int(question_is_correct)

        seed_response = None
        seed_pred_boxed = None
        seed_is_correct = None
        if item.seed_question:
            seed_total += 1
            seed_response = response_map[(idx, "seed")]
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

        if progress_every_rows is not None and progress_every_rows > 0:
            done_rows = idx + 1
            if done_rows % progress_every_rows == 0:
                run_q = question_correct / done_rows
                run_s = seed_correct / seed_total if seed_total else 0.0
                run_c = (question_correct + seed_correct) / (done_rows + seed_total)
                print(
                    f"[boxed-eval] progress rows={done_rows}/{len(items)} "
                    f"running_q_acc={run_q:.4f} running_seed_acc={run_s:.4f} "
                    f"running_combined_acc={run_c:.4f}",
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
        "total": combined_total,
        "correct": combined_correct,
        "accuracy": combined_accuracy,
        "results": results,
    }

