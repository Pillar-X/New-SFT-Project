#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_name.answer_filter import extract_final_answer_candidate, is_pure_numeric_answer


def normalize_answer_by_hash(answer: str) -> str:
    """
    规范化：若 answer 中出现 '#', 则截断到首个 '#' 之前。
    """
    idx = answer.find("#")
    if idx == -1:
        return answer
    return answer[:idx].rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter extracted QA pairs and keep only pure numeric final answers."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/nemotron_qna_matched_only.jsonl",
        help="Input extracted QA JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/nemotron_qna_numeric_only.jsonl",
        help="Output JSONL containing only numeric final answers.",
    )
    parser.add_argument(
        "--keep-candidate-field",
        action="store_true",
        help="Keep `final_answer_candidate` field in output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    hash_truncated = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            answer = row.get("answer")
            if not isinstance(answer, str):
                continue

            normalized_answer = normalize_answer_by_hash(answer)
            if normalized_answer != answer:
                hash_truncated += 1
            row["answer"] = normalized_answer

            candidate = extract_final_answer_candidate(normalized_answer)
            if not is_pure_numeric_answer(candidate):
                continue

            if args.keep_candidate_field:
                row["final_answer_candidate"] = candidate
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    rate = kept / total if total else 0.0
    print(
        f"[numeric-filter] total={total} kept={kept} truncated_by_hash={hash_truncated} "
        f"kept_rate={rate:.2%} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
