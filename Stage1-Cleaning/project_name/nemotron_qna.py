"""从 Nemotron/CC 类文档的 `text` 字段中抽取问题+解答对。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

MatchKind = Literal[
    "markdown_headers",
    "bold_labels",
    "labeled_blocks",
    "heading_question_solution",
    "none",
]


@dataclass
class QnAParseResult:
    question: Optional[str]
    answer: Optional[str]
    kind: MatchKind


# Markdown：行首 # 标题且行内含单词 question / answer（不区分大小写）
_RE_MD_QUESTION_LINE = re.compile(r"^#{1,6}\s+[^\n]*\bquestion\b[^\n]*\s*$", re.MULTILINE | re.IGNORECASE)
_RE_MD_ANSWER_LINE = re.compile(r"^#{1,6}\s+[^\n]*\banswer\b[^\n]*\s*$", re.MULTILINE | re.IGNORECASE)

# 粗体标签：**Question:** / **Answer:**（兼容常见 typo：anwer）
_RE_BOLD_QUESTION = re.compile(r"\*\*question:\*\*", re.IGNORECASE)
_RE_BOLD_ANSWER = re.compile(r"\*\*(?:answer|anwer):\*\*", re.IGNORECASE)
_RE_INLINE_BOLD_QA = re.compile(
    r"(?is)(?:^|\n)\s*(?:\d+\.\s*)?\*\*question:\*\*\s*(.*?)\s*(?:^|\n)\s*(?:\d+\.\s*)?\*\*(?:answer|anwer):\*\*\s*(.*?)(?=(?:\n\s*(?:\d+\.\s*)?\*\*question:\*\*)|\Z)"
)

# 标签式块：Question/Problem/Prompt ... + Answer/Solution/Explanation ...
_RE_LABEL_Q = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:question|q|problem(?:\s+statement)?|prompt)\s*(?:\d+)?\s*:\s*(?:\*\*)?\s*(.*)$"
)
_RE_LABEL_A = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:answer|anwer|best answer|final answer|solution|explanation)\s*(?:\d+)?\s*:\s*(?:\*\*)?\s*(.*)$"
)

# 标题问句 + 解题标签（如：# How to calculate ... ? + **Solution:**）
_RE_HEADING_QUESTION = re.compile(r"(?im)^#{1,6}\s+(.{3,300}\?)\s*$")
_RE_SOLUTION_LINE = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:answer|anwer|best answer|final answer|solution|explanation)\s*:?\s*(?:\*\*)?\s*$"
)
_RE_SEPARATOR_LINE = re.compile(r"(?m)^\s*---+\s*$")


def _clean_block(value: str) -> Optional[str]:
    s = value.strip()
    if not s:
        return None
    # 去掉常见分隔线，减少脏噪声。
    s = re.sub(r"(?m)^\s*---+\s*$", "", s).strip()
    return s or None


def _dedup_pairs(pairs: list[QnAParseResult]) -> list[QnAParseResult]:
    seen: set[tuple[str, str]] = set()
    out: list[QnAParseResult] = []
    for p in pairs:
        key = ((p.question or "").strip(), (p.answer or "").strip())
        if not key[0] and not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _extract_markdown_pairs(text: str) -> list[QnAParseResult]:
    pairs: list[QnAParseResult] = []
    q_lines = list(_RE_MD_QUESTION_LINE.finditer(text))
    a_lines = list(_RE_MD_ANSWER_LINE.finditer(text))
    if not q_lines or not a_lines:
        return pairs

    for i, mq in enumerate(q_lines):
        next_q_start = q_lines[i + 1].start() if i + 1 < len(q_lines) else len(text)
        ma = next((m for m in a_lines if m.start() > mq.end() and m.start() < next_q_start), None)
        if not ma:
            continue
        question = _clean_block(text[mq.end() : ma.start()])
        answer = _clean_block(_clip_answer_tail(text, ma.end(), next_q_start))
        if question or answer:
            pairs.append(QnAParseResult(question, answer, "markdown_headers"))

    return pairs


def _extract_bold_labels(text: str) -> Optional[QnAParseResult]:
    mq = _RE_BOLD_QUESTION.search(text)
    ma = _RE_BOLD_ANSWER.search(text)
    if not (mq and ma and ma.start() > mq.end()):
        return None
    question = _clean_block(text[mq.end() : ma.start()])
    answer = _clean_block(text[ma.end() :])
    if question or answer:
        return QnAParseResult(question, answer, "bold_labels")
    return None


def _extract_inline_bold_pairs(text: str) -> list[QnAParseResult]:
    pairs: list[QnAParseResult] = []
    for m in _RE_INLINE_BOLD_QA.finditer(text):
        question = _clean_block(m.group(1) or "")
        answer = _clean_block(m.group(2) or "")
        if question or answer:
            pairs.append(QnAParseResult(question, answer, "bold_labels"))
    return pairs


def _clip_answer_tail(text: str, start: int, end: int) -> str:
    """
    在已知 answer 区间内，遇到明显分段线时提前截断，降低把后续不相关题目吞进同一答案的概率。
    """
    seg = text[start:end]
    sep = _RE_SEPARATOR_LINE.search(seg)
    if sep:
        return seg[: sep.start()]
    return seg


def _extract_labeled_blocks(text: str) -> list[QnAParseResult]:
    pairs: list[QnAParseResult] = []
    q_matches = list(_RE_LABEL_Q.finditer(text))
    a_matches = list(_RE_LABEL_A.finditer(text))
    if not q_matches or not a_matches:
        return pairs

    for i, mq in enumerate(q_matches):
        q_end = q_matches[i + 1].start() if i + 1 < len(q_matches) else len(text)
        ma = next((m for m in a_matches if m.start() > mq.start() and m.start() < q_end), None)
        if not ma:
            continue

        q_inline = (mq.group(1) or "").strip()
        q_between = text[mq.end() : ma.start()]
        question = _clean_block((q_inline + "\n" + q_between) if q_inline else q_between)

        next_q_start = q_end
        answer_tail = _clip_answer_tail(text, ma.end(), next_q_start)
        a_inline = (ma.group(1) or "").strip()
        answer = _clean_block((a_inline + "\n" + answer_tail) if a_inline else answer_tail)

        if question or answer:
            pairs.append(QnAParseResult(question, answer, "labeled_blocks"))

    return pairs


def _extract_heading_question_solution(text: str) -> Optional[QnAParseResult]:
    hq = _RE_HEADING_QUESTION.search(text)
    if not hq:
        return None
    sol = _RE_SOLUTION_LINE.search(text, hq.end())
    if not sol:
        return None
    if sol.start() - hq.end() > 1500:
        # 标题问句距离解答标签太远，通常不是单个问答体，避免误抽。
        return None

    # 标题中的问句作为问题主体，标题后到 solution 之前的段落拼到问题上下文。
    q_head = (hq.group(1) or "").strip()
    q_context = _clean_block(text[hq.end() : sol.start()])
    if q_context:
        question = _clean_block(q_head + "\n\n" + q_context)
    else:
        question = _clean_block(q_head)
    answer = _clean_block(_clip_answer_tail(text, sol.end(), len(text)))
    if question or answer:
        return QnAParseResult(question, answer, "heading_question_solution")
    return None


def parse_question_answer_pairs(text: str) -> list[QnAParseResult]:
    """
    尽量全面抽取一条 `text` 内的问答对，支持多组输出。

    策略顺序：
    1) Question/Answer 风格标签块（可抽多组，如 Question 1..n）
    2) Markdown Question/Answer 标题对
    3) **Question:** / **Answer:** 标签对
    4) 标题问句（含 ?）+ Solution/Answer/Explanation 标签
    """
    if not text or not text.strip():
        return []

    pairs: list[QnAParseResult] = []
    pairs.extend(_extract_labeled_blocks(text))
    pairs.extend(_extract_inline_bold_pairs(text))

    pairs.extend(_extract_markdown_pairs(text))

    # 对于只有单组 **Question:** / **Answer:** 的文本，inline 匹配可能不命中，
    # 再补一个单组兜底。
    if not pairs:
        bold_pair = _extract_bold_labels(text)
        if bold_pair is not None:
            pairs.append(bold_pair)

    # 标题问句 + Solution 是兜底策略，仅在更显式标签都无法命中时启用，减少重复。
    if not pairs:
        hq_pair = _extract_heading_question_solution(text)
        if hq_pair is not None:
            pairs.append(hq_pair)

    return _dedup_pairs(pairs)


def parse_question_answer(text: str) -> QnAParseResult:
    """
    从整段 `text` 中尽量切出「问题」与「解答」正文（不含标题行本身）。

    返回首个问答对；若需同一条 text 的全部问答对，请使用 `parse_question_answer_pairs`。
    """
    pairs = parse_question_answer_pairs(text)
    if pairs:
        return pairs[0]
    return QnAParseResult(None, None, "none")


def parse_question_answer_tuple(text: str) -> Tuple[Optional[str], Optional[str]]:
    r = parse_question_answer(text)
    return r.question, r.answer
