from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


@dataclass
class TeacherConfig:
    base_url: str
    api_key: str
    model: str
    timeout_sec: int = 90


SYSTEM_PROMPT = """You are a strict evaluator for Q/A extraction quality.
Return ONLY valid JSON.
No markdown, no extra text.
"""


def load_teacher_config(env_path: str = ".env") -> TeacherConfig:
    load_dotenv(env_path, override=False)

    base_url = os.getenv("TEACHER_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("TEACHER_API_KEY", "").strip()
    model = os.getenv("TEACHER_MODEL", "").strip()
    timeout_sec = int(os.getenv("TEACHER_TIMEOUT_SEC", "90"))

    if not base_url or not api_key or not model:
        raise ValueError(
            "Missing TEACHER_BASE_URL / TEACHER_API_KEY / TEACHER_MODEL in environment."
        )

    return TeacherConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_sec=timeout_sec,
    )


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def _build_user_prompt(record: Dict[str, Any]) -> str:
    source_id = record.get("source_id")
    pair_index = record.get("pair_index")
    match_kind = record.get("match_kind")
    question = record.get("question") or ""
    answer = record.get("answer") or ""

    return f"""Evaluate this extracted QA pair.

You must score two labels:
1) extraction_ok: whether this looks like a correctly extracted "question + answer" pair (not broken split, not random text)
2) answer_ok: whether the answer correctly addresses the question mathematically/logically.

Data:
- source_id: {source_id}
- pair_index: {pair_index}
- match_kind: {match_kind}

Question:
{question}

Answer:
{answer}

Output strict JSON with fields:
{{
  "extraction_ok": true/false,
  "answer_ok": true/false,
  "confidence": 0.0-1.0,
  "reason": "short reason"
}}
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


def judge_one(record: Dict[str, Any], cfg: TeacherConfig, max_retry: int = 2) -> Dict[str, Any]:
    prompt = _build_user_prompt(record)
    last_err: Optional[str] = None

    for _ in range(max_retry + 1):
        try:
            content = _post_chat_completion(cfg, prompt)
            result = json.loads(content)
            return {
                "source_id": record.get("source_id"),
                "pair_index": record.get("pair_index"),
                "match_kind": record.get("match_kind"),
                "question": record.get("question"),
                "answer": record.get("answer"),
                "extraction_ok": bool(result.get("extraction_ok", False)),
                "answer_ok": bool(result.get("answer_ok", False)),
                "confidence": float(result.get("confidence", 0.0)),
                "reason": str(result.get("reason", "")),
                "error": None,
            }
        except (json.JSONDecodeError, KeyError, ValueError, urllib.error.URLError) as e:
            last_err = str(e)
            time.sleep(1.0)

    return {
        "source_id": record.get("source_id"),
        "pair_index": record.get("pair_index"),
        "match_kind": record.get("match_kind"),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "extraction_ok": False,
        "answer_ok": False,
        "confidence": 0.0,
        "reason": "",
        "error": last_err or "unknown_error",
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    ok_extraction = sum(1 for r in results if r.get("extraction_ok") is True)
    ok_answer = sum(1 for r in results if r.get("answer_ok") is True)
    both_ok = sum(
        1
        for r in results
        if r.get("extraction_ok") is True and r.get("answer_ok") is True
    )
    error_count = sum(1 for r in results if r.get("error"))
    avg_conf = (
        sum(float(r.get("confidence", 0.0)) for r in results) / total if total else 0.0
    )
    return {
        "total": total,
        "extraction_ok_count": ok_extraction,
        "answer_ok_count": ok_answer,
        "both_ok_count": both_ok,
        "extraction_ok_rate": ok_extraction / total if total else 0.0,
        "answer_ok_rate": ok_answer / total if total else 0.0,
        "both_ok_rate": both_ok / total if total else 0.0,
        "error_count": error_count,
        "avg_confidence": avg_conf,
    }


def evaluate_sample(
    input_path: str | Path,
    sample_size: int,
    seed: int,
    env_path: str = ".env",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No records found in {input_path}")

    # 默认跳过 none，因为这类通常不是有效问答抽取结果。
    valid_rows = [r for r in rows if r.get("match_kind") != "none"]
    if not valid_rows:
        raise ValueError("No valid rows after filtering match_kind != 'none'.")

    sampled = sample_rows(valid_rows, sample_size=sample_size, seed=seed)
    cfg = load_teacher_config(env_path=env_path)
    judged = [judge_one(r, cfg) for r in sampled]
    summary = summarize_results(judged)
    summary["sample_size_requested"] = sample_size
    summary["sample_size_actual"] = len(sampled)
    return judged, summary
