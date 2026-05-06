from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .eval_teacher import TeacherConfig, load_teacher_config, write_json, write_jsonl

_RE_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")

GEN_SYSTEM_PROMPT = (
    "You are a careful math solver. Return ONLY one final boxed answer: \\boxed{...}. "
    "If impossible to determine, return \\boxed{None}."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Determine whether two boxed final answers are equivalent. "
    "Return ONLY JSON."
)


def read_sft_json(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("SFT JSON must be a list.")
    return data


def normalize_boxed(text: str) -> str:
    if not text:
        return r"\boxed{None}"
    m = _RE_BOXED.search(text)
    if not m:
        return r"\boxed{None}"
    inner = (m.group(1) or "").strip()
    if not inner:
        return r"\boxed{None}"
    return f"\\boxed{{{inner}}}"


def extract_question_and_boxed(sample: Dict[str, Any]) -> Tuple[Optional[str], str]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return None, r"\boxed{None}"
    question = None
    assistant_content = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "user" and isinstance(content, str):
            question = content
        elif role == "assistant" and isinstance(content, str):
            assistant_content = content
    extracted_boxed = normalize_boxed(assistant_content)
    return question, extracted_boxed


def _post_chat(cfg: TeacherConfig, system_prompt: str, user_prompt: str) -> str:
    url = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
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


def generate_boxed_from_question(
    question: str,
    cfg: TeacherConfig,
    max_retry: int = 2,
) -> str:
    user_prompt = (
        "Solve this question and output only one boxed final answer.\n\n"
        f"Question:\n{question}\n"
    )
    last_err = ""
    for _ in range(max_retry + 1):
        try:
            content = _post_chat(cfg, GEN_SYSTEM_PROMPT, user_prompt)
            return normalize_boxed(content)
        except (json.JSONDecodeError, KeyError, ValueError, urllib.error.URLError) as e:
            last_err = str(e)
            time.sleep(1.0)
    raise RuntimeError(f"Failed to generate boxed answer: {last_err}")


def judge_boxed_match(
    extracted_boxed: str,
    generated_boxed: str,
    cfg: TeacherConfig,
    max_retry: int = 2,
) -> Dict[str, Any]:
    user_prompt = f"""Decide whether the two boxed answers are mathematically equivalent.

Extracted boxed answer:
{extracted_boxed}

Generated boxed answer:
{generated_boxed}

Return strict JSON:
{{
  "is_match": true/false,
  "reason": "short reason"
}}
"""
    last_err = ""
    for _ in range(max_retry + 1):
        try:
            content = _post_chat(cfg, JUDGE_SYSTEM_PROMPT, user_prompt)
            result = json.loads(content)
            return {
                "is_match": bool(result.get("is_match", False)),
                "reason": str(result.get("reason", "")),
                "error": None,
            }
        except (json.JSONDecodeError, KeyError, ValueError, urllib.error.URLError) as e:
            last_err = str(e)
            time.sleep(1.0)
    return {"is_match": False, "reason": "", "error": last_err or "unknown_error"}


def evaluate_boxed_extraction(
    input_path: str | Path,
    env_path: str = ".env",
    sample_size: int = 0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = read_sft_json(input_path)
    if sample_size > 0 and sample_size < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, sample_size)

    cfg = load_teacher_config(env_path=env_path)
    details: List[Dict[str, Any]] = []
    total = len(data)
    for idx, sample in enumerate(data, start=1):
        question, extracted_boxed = extract_question_and_boxed(sample)
        if not question:
            details.append(
                {
                    "index": idx - 1,
                    "question": None,
                    "generated_boxed": r"\boxed{None}",
                    "extracted_boxed": extracted_boxed,
                    "is_match": False,
                    "reason": "missing_question",
                    "error": "missing_question",
                }
            )
            continue
        try:
            generated_boxed = generate_boxed_from_question(question, cfg)
            judge = judge_boxed_match(extracted_boxed, generated_boxed, cfg)
            details.append(
                {
                    "index": idx - 1,
                    "question": question,
                    "generated_boxed": generated_boxed,
                    "extracted_boxed": extracted_boxed,
                    "is_match": bool(judge["is_match"]),
                    "reason": judge["reason"],
                    "error": judge["error"],
                }
            )
        except Exception as e:  # noqa: BLE001
            details.append(
                {
                    "index": idx - 1,
                    "question": question,
                    "generated_boxed": r"\boxed{None}",
                    "extracted_boxed": extracted_boxed,
                    "is_match": False,
                    "reason": "",
                    "error": str(e),
                }
            )
        if idx % 20 == 0 or idx == total:
            print(f"[boxed-eval] progress={idx}/{total}", flush=True)

    ok = sum(1 for d in details if d.get("is_match") is True)
    errors = sum(1 for d in details if d.get("error"))
    summary = {
        "total": len(details),
        "match_count": ok,
        "match_rate": ok / len(details) if details else 0.0,
        "error_count": errors,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(details),
        "seed": seed,
        "input_path": str(input_path),
    }
    return details, summary


def write_eval_outputs(
    output_dir: str | Path,
    details: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    details_path = out / "boxed_eval_details.jsonl"
    summary_path = out / "boxed_eval_summary.json"
    write_jsonl(details_path, details)
    write_json(summary_path, summary)
    return details_path, summary_path
