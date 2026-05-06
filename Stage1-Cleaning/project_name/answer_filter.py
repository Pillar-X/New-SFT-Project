from __future__ import annotations

import re
from typing import Optional

# 允许的“纯数字表达”
# 1) 整数/小数/科学计数法
_RE_PURE_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
# 2) 普通分数（无变量）
_RE_PURE_PLAIN_FRACTION = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*/\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)$"
)
# 3) LaTeX 分数（无变量）
_RE_PURE_LATEX_FRACTION = re.compile(
    r"^\\frac\s*\{\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*\}\s*\{\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*\}$"
)

# 用于在 answer 中扫描“最后出现数字”
_RE_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_RE_LATEX_FRACTION_ANY = re.compile(
    r"\\frac\s*\{\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*\}\s*\{\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*\}"
)
_RE_PLAIN_FRACTION_ANY = re.compile(
    r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*/\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
)
_RE_NUMBER_ANY = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")
_RE_WORD_TOKEN = re.compile(r"[a-z]+")

_NUM_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_NUM_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_NUM_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUM_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
_NUM_SIGN_WORDS = {"negative", "minus"}
_NUM_DECIMAL_WORDS = {"point"}
_NUM_FRACTION_WORDS = {"over"}
_NUM_JOIN_WORDS = {"and"}
_NUM_DIGIT_WORDS = {
    **_NUM_UNITS,
    "oh": 0,
    "o": 0,
}
_ALL_NUMBER_WORDS = (
    set(_NUM_UNITS)
    | set(_NUM_TEENS)
    | set(_NUM_TENS)
    | set(_NUM_SCALES)
    | _NUM_SIGN_WORDS
    | _NUM_DECIMAL_WORDS
    | _NUM_FRACTION_WORDS
    | _NUM_JOIN_WORDS
    | {"a", "an"}
)


def _cleanup_candidate(text: str) -> str:
    s = text.strip()
    s = re.sub(r"`+", "", s)
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = s.rstrip(".,;:!?)").strip()
    s = s.lstrip("(").strip()
    return s


def _contains_alpha_in_local_token(text: str, start: int, end: int) -> bool:
    """
    取命中点周围的“连续非空白 token”，若 token 含字母，则认为它可能是变量表达式（如 2x, 3x+1）。
    """
    left = start
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and not text[right].isspace():
        right += 1
    token = text[left:right]
    token = token.strip("`\"'()[]{}.,;:!?$")
    return any(ch.isalpha() for ch in token)


def _in_spans(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if start >= s and end <= e:
            return True
    return False


def _parse_integer_words(words: list[str]) -> Optional[int]:
    if not words:
        return None
    total = 0
    current = 0
    seen = False
    for w in words:
        if w in _NUM_JOIN_WORDS:
            continue
        if w in ("a", "an"):
            # 仅在 hundred/thousand 前使用 "a/an" 时才有意义，这里先按 1 处理。
            current += 1
            seen = True
            continue
        if w in _NUM_UNITS:
            current += _NUM_UNITS[w]
            seen = True
            continue
        if w in _NUM_TEENS:
            current += _NUM_TEENS[w]
            seen = True
            continue
        if w in _NUM_TENS:
            current += _NUM_TENS[w]
            seen = True
            continue
        if w == "hundred":
            if current == 0:
                current = 1
            current *= 100
            seen = True
            continue
        if w in ("thousand", "million", "billion"):
            scale = _NUM_SCALES[w]
            if current == 0:
                current = 1
            total += current * scale
            current = 0
            seen = True
            continue
        return None
    if not seen:
        return None
    return total + current


def _parse_decimal_words(words: list[str]) -> Optional[str]:
    if "point" not in words:
        return None
    idx = words.index("point")
    left_words = words[:idx]
    right_words = words[idx + 1 :]
    if not right_words:
        return None

    if left_words:
        left_value = _parse_integer_words(left_words)
        if left_value is None:
            return None
    else:
        left_value = 0

    digits: list[str] = []
    for w in right_words:
        if w in _NUM_JOIN_WORDS:
            continue
        if w not in _NUM_DIGIT_WORDS:
            return None
        digits.append(str(_NUM_DIGIT_WORDS[w]))
    if not digits:
        return None
    return f"{left_value}.{''.join(digits)}"


def _parse_number_word_phrase(words: list[str]) -> Optional[str]:
    if not words:
        return None
    sign = ""
    if words and words[0] in _NUM_SIGN_WORDS:
        sign = "-"
        words = words[1:]
        if not words:
            return None

    # 形如: two over three
    over_count = words.count("over")
    if over_count == 1:
        idx = words.index("over")
        left = _parse_integer_words(words[:idx])
        right = _parse_integer_words(words[idx + 1 :])
        if left is None or right is None:
            return None
        return f"{sign}{left}/{right}"
    if over_count > 1:
        return None

    dec = _parse_decimal_words(words)
    if dec is not None:
        return f"{sign}{dec}" if sign else dec

    integer = _parse_integer_words(words)
    if integer is None:
        return None
    return f"{sign}{integer}" if sign else str(integer)


def _extract_word_number_candidates(answer: str) -> list[tuple[int, str]]:
    """
    扫描英文数字词（如 twenty two thousand, two over three），返回候选及其出现位置。
    """
    text = answer.lower().replace("-", " ")
    tokens = list(_RE_WORD_TOKEN.finditer(text))
    if not tokens:
        return []

    words = [m.group(0) for m in tokens]
    starts = [m.start() for m in tokens]

    candidates: list[tuple[int, str]] = []
    i = 0
    n = len(words)
    max_span = 16
    while i < n:
        if words[i] not in _ALL_NUMBER_WORDS:
            i += 1
            continue

        best_j = -1
        best_value: Optional[str] = None
        upper = min(n, i + max_span)
        for j in range(upper, i, -1):
            seg = words[i:j]
            if not seg:
                continue
            if any(w not in _ALL_NUMBER_WORDS for w in seg):
                continue
            parsed = _parse_number_word_phrase(seg)
            if parsed is None:
                continue
            best_j = j
            best_value = parsed
            break

        if best_j != -1 and best_value is not None:
            candidates.append((starts[i], best_value))
            i = best_j
        else:
            i += 1

    return candidates


def is_pure_numeric_answer(candidate: str | None) -> bool:
    if candidate is None:
        return False
    c = candidate.strip()
    return bool(
        _RE_PURE_NUMBER.fullmatch(c)
        or _RE_PURE_PLAIN_FRACTION.fullmatch(c)
        or _RE_PURE_LATEX_FRACTION.fullmatch(c)
    )


def extract_final_answer_candidate(answer: str) -> Optional[str]:
    """
    从 answer 文本提取“最终答案候选”：
    取 answer 中“最后出现”的纯数字表达（数值/分数），并尽量排除变量表达式片段。
    """
    if not answer or not answer.strip():
        return None

    candidates: list[tuple[int, str]] = []
    occupied_spans: list[tuple[int, int]] = []

    # 1) \\boxed{...}
    for m in _RE_BOXED.finditer(answer):
        inner = _cleanup_candidate(m.group(1))
        if is_pure_numeric_answer(inner):
            candidates.append((m.start(), inner))
            occupied_spans.append((m.start(), m.end()))

    # 2) LaTeX fraction
    for m in _RE_LATEX_FRACTION_ANY.finditer(answer):
        value = _cleanup_candidate(m.group(0))
        if is_pure_numeric_answer(value):
            candidates.append((m.start(), value))
            occupied_spans.append((m.start(), m.end()))

    # 3) Plain fraction
    for m in _RE_PLAIN_FRACTION_ANY.finditer(answer):
        if _in_spans(m.start(), m.end(), occupied_spans):
            continue
        value = _cleanup_candidate(m.group(0))
        if is_pure_numeric_answer(value) and not _contains_alpha_in_local_token(
            answer, m.start(), m.end()
        ):
            candidates.append((m.start(), value))
            occupied_spans.append((m.start(), m.end()))

    # 4) Plain number
    for m in _RE_NUMBER_ANY.finditer(answer):
        if _in_spans(m.start(), m.end(), occupied_spans):
            continue
        value = _cleanup_candidate(m.group(0))
        if not is_pure_numeric_answer(value):
            continue
        if _contains_alpha_in_local_token(answer, m.start(), m.end()):
            continue
        candidates.append((m.start(), value))

    # 5) English number words (e.g., twenty two thousand, two over three)
    for pos, value in _extract_word_number_candidates(answer):
        if is_pure_numeric_answer(value):
            candidates.append((pos, value))

    if not candidates:
        return None

    # 规则：默认最后出现的数字是最终答案
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]
