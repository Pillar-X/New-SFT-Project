from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# 轻量 token 近似：单词/数字块 + 单个标点符号
_RE_APPROX_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class BucketRule:
    name: str
    min_tokens: int
    max_tokens: int


DEFAULT_BUCKETS = [
    BucketRule("small", 100, 400),
    BucketRule("middle", 400, 1000),
    BucketRule("large", 1000, 2000),
    BucketRule("super-large", 2000, 4096),
]


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(_RE_APPROX_TOKEN.findall(text))


def combined_qa_token_count(question: Optional[str], answer: Optional[str]) -> int:
    q = question or ""
    a = answer or ""
    return approx_token_count(q + "\n" + a)


def assign_bucket(total_tokens: int) -> Optional[str]:
    # 区间定义：
    # small: [100, 400)
    # middle: [400, 1000)
    # large: [1000, 2000)
    # super-large: [2000, 4096]
    for rule in DEFAULT_BUCKETS:
        if rule.name == "super-large":
            if rule.min_tokens <= total_tokens <= rule.max_tokens:
                return rule.name
            continue
        if rule.min_tokens <= total_tokens < rule.max_tokens:
            return rule.name
    return None
