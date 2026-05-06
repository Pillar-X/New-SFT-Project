from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

SYSTEM_PROMPT = "You are a helpful math tutor. Extract the final answer from the given text."

EXTRACT_PROMPT_TEMPLATE = """Given a math question and an existing answer, extract the final answer.

Rules:
1) Return ONLY one LaTeX boxed expression: \\boxed{{...}}.
2) If you are sure the final answer cannot be found, return \\boxed{{None}}.
3) Do not output any extra text.

Given text:
{answer}
"""

_RE_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")


@dataclass
class StudentConfig:
    base_url: str
    api_key: str
    model: str
    timeout_sec: int = 90


def _is_local_endpoint(base_url: str) -> bool:
    host = (urllib.parse.urlparse(base_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "0.0.0.0"}


def _service_root_url(base_url: str) -> str:
    """
    将 OpenAI 兼容前缀（通常是 /v1）还原到服务根路径，便于访问 /health。
    """
    if base_url.endswith("/v1"):
        return base_url[: -len("/v1")]
    return base_url


def _urlopen(req: urllib.request.Request, cfg: StudentConfig):
    """
    对本地 vLLM 端点自动绕过系统代理，避免 localhost 请求被代理成 502。
    """
    if _is_local_endpoint(cfg.base_url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=cfg.timeout_sec)
    return urllib.request.urlopen(req, timeout=cfg.timeout_sec)


def load_student_config(env_path: str = ".env") -> StudentConfig:
    load_dotenv(env_path, override=False)
    base_url = os.getenv("STUDENT_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("STUDENT_API_KEY", "").strip() or "dummy"
    model = os.getenv("STUDENT_MODEL", "").strip()
    timeout_sec = int(os.getenv("STUDENT_TIMEOUT_SEC", "90"))
    if not base_url or not model:
        raise ValueError("Missing STUDENT_BASE_URL / STUDENT_MODEL in environment.")
    return StudentConfig(
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


def normalize_boxed(text: str) -> str:
    if not text:
        return r"\boxed{None}"
    match = _RE_BOXED.search(text)
    if not match:
        return r"\boxed{None}"
    inner = (match.group(1) or "").strip()
    if not inner:
        return r"\boxed{None}"
    return f"\\boxed{{{inner}}}"


def _post_chat_completion(
    cfg: StudentConfig,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    url = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
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
    with _urlopen(req, cfg) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def check_student_api_ready(cfg: StudentConfig) -> None:
    """
    Fail-fast 连通性检查，避免“脚本跑了但其实没打到本地 vLLM”。
    """
    # 1) health
    health_url = f"{_service_root_url(cfg.base_url)}/health"
    req_health = urllib.request.Request(url=health_url, method="GET")
    with _urlopen(req_health, cfg):
        pass

    # 2) models
    models_url = f"{cfg.base_url}/models"
    req_models = urllib.request.Request(
        url=models_url,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        method="GET",
    )
    with _urlopen(req_models, cfg) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    data = parsed.get("data", [])
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"No models returned from {models_url}")


def extract_boxed_answer_with_retry(
    answer: str,
    cfg: StudentConfig,
    temperature: float = 0.0,
    max_retry: int = 2,
    fail_on_error: bool = True,
) -> str:
    user_prompt = EXTRACT_PROMPT_TEMPLATE.format(answer=answer)
    last_err = ""
    for _ in range(max_retry + 1):
        try:
            content = _post_chat_completion(cfg, user_prompt=user_prompt, temperature=temperature)
            boxed = normalize_boxed(content)
            return boxed
        except (
            json.JSONDecodeError,
            KeyError,
            ValueError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as e:
            last_err = str(e)
            time.sleep(1.0)
    if fail_on_error:
        raise RuntimeError(f"vLLM request failed after retries: {last_err}")
    return r"\boxed{None}"


def build_sft_messages(question: str, answer: str, boxed_final: str) -> List[Dict[str, str]]:
    user_content = question
    assistant_content = answer.rstrip() + "\n" + boxed_final
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def build_sft_dataset(
    rows: List[Dict[str, Any]],
    cfg: StudentConfig,
    temperature: float = 0.0,
    max_retry: int = 2,
    fail_on_error: bool = True,
    progress_every: int = 50,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        boxed = extract_boxed_answer_with_retry(
            answer=answer,
            cfg=cfg,
            temperature=temperature,
            max_retry=max_retry,
            fail_on_error=fail_on_error,
        )
        out.append(
            {
                "messages": build_sft_messages(question=question, answer=answer, boxed_final=boxed),
            }
        )
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            print(f"[build-boxed-sft] progress={idx}/{total}", flush=True)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
