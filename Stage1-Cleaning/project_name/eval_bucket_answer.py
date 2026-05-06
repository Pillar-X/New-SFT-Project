from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .eval_teacher import TeacherConfig, load_teacher_config, read_jsonl

BUCKET_NAMES = ["small", "middle", "large", "super-large"]

SYSTEM_PROMPT = """You are a strict math QA evaluator.
Your job is to determine whether the answer correctly answers the question.
Important:
- Infer the final answer ONLY from the provided answer text.
- Do NOT use any external fields or metadata.
- Focus on final-answer correctness first.
Return ONLY valid JSON.
"""


def _post_chat_completion(cfg: TeacherConfig, user_prompt: str) -> str:
    url = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def _build_user_prompt(record: Dict[str, Any], bucket: str) -> str:
    question = record.get("question") or ""
    answer = record.get("answer") or ""
    return f"""Evaluate whether the answer correctly solves the question.

Bucket: {bucket}

Question:
{question}

Answer:
{answer}

Rules:
1) Infer the final answer from the answer text itself.
2) Do not trust style/length; judge correctness.
3) If unclear or contradictory, mark false.

Output strict JSON:
{{
  "answer_ok": true/false,
  "confidence": 0.0-1.0,
  "extracted_final_answer_from_answer": "string",
  "reason": "short reason"
}}
"""


def sample_bucket_rows(
    input_dir: str | Path,
    sample_per_bucket: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    base = Path(input_dir)
    for i, bucket in enumerate(BUCKET_NAMES):
        bucket_path = base / f"{bucket}.jsonl"
        rows = read_jsonl(bucket_path)
        if not rows:
            out[bucket] = []
            continue
        rng = random.Random(seed + i * 9973)
        if len(rows) <= sample_per_bucket:
            sampled = rows
        else:
            sampled = rng.sample(rows, sample_per_bucket)
        out[bucket] = sampled
    return out


def judge_one(
    record: Dict[str, Any],
    bucket: str,
    cfg: TeacherConfig,
    max_retry: int = 2,
) -> Dict[str, Any]:
    prompt = _build_user_prompt(record, bucket)
    last_err = ""
    for _ in range(max_retry + 1):
        try:
            content = _post_chat_completion(cfg, prompt)
            result = json.loads(content)
            return {
                "bucket": bucket,
                "source_id": record.get("source_id"),
                "pair_index": record.get("pair_index"),
                "match_kind": record.get("match_kind"),
                "total_tokens": record.get("total_tokens"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "answer_ok": bool(result.get("answer_ok", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "extracted_final_answer_from_answer": str(
                    result.get("extracted_final_answer_from_answer", "")
                ),
                "reason": str(result.get("reason", "")),
                "error": None,
            }
        except (json.JSONDecodeError, KeyError, ValueError, urllib.error.URLError) as e:
            last_err = str(e)
            time.sleep(1.0)

    return {
        "bucket": bucket,
        "source_id": record.get("source_id"),
        "pair_index": record.get("pair_index"),
        "match_kind": record.get("match_kind"),
        "total_tokens": record.get("total_tokens"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "answer_ok": False,
        "confidence": 0.0,
        "extracted_final_answer_from_answer": "",
        "reason": "",
        "error": last_err or "unknown_error",
    }


def summarize_by_bucket(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"overall": {}, "buckets": {}}
    total = len(results)
    ok = sum(1 for r in results if r.get("answer_ok") is True)
    errors = sum(1 for r in results if r.get("error"))
    avg_conf = (
        sum(float(r.get("confidence", 0.0)) for r in results) / total if total else 0.0
    )
    summary["overall"] = {
        "total": total,
        "answer_ok_count": ok,
        "answer_ok_rate": ok / total if total else 0.0,
        "error_count": errors,
        "avg_confidence": avg_conf,
    }

    for bucket in BUCKET_NAMES:
        part = [r for r in results if r.get("bucket") == bucket]
        b_total = len(part)
        b_ok = sum(1 for r in part if r.get("answer_ok") is True)
        b_errors = sum(1 for r in part if r.get("error"))
        b_avg_conf = (
            sum(float(r.get("confidence", 0.0)) for r in part) / b_total if b_total else 0.0
        )
        summary["buckets"][bucket] = {
            "total": b_total,
            "answer_ok_count": b_ok,
            "answer_ok_rate": b_ok / b_total if b_total else 0.0,
            "error_count": b_errors,
            "avg_confidence": b_avg_conf,
        }
    return summary


def evaluate_bucket_samples(
    input_dir: str | Path,
    sample_per_bucket: int,
    seed: int,
    env_path: str = ".env",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = load_teacher_config(env_path=env_path)
    sampled = sample_bucket_rows(input_dir, sample_per_bucket=sample_per_bucket, seed=seed)
    results: List[Dict[str, Any]] = []
    for bucket in BUCKET_NAMES:
        for row in sampled.get(bucket, []):
            results.append(judge_one(row, bucket=bucket, cfg=cfg))
    summary = summarize_by_bucket(results)
    summary["sample_per_bucket_requested"] = sample_per_bucket
    summary["seed"] = seed
    summary["input_dir"] = str(input_dir)
    return results, summary
