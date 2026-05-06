#!/usr/bin/env python3
"""从 Nemotron JSONL 的 `text` 字段中尽量全面抽取问题+解答，写出 JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_name.nemotron_qna import parse_question_answer_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read JSONL records, split `text` into question/answer by keywords."
    )
    p.add_argument(
        "--input",
        type=str,
        default="data/raw/nv-community_Nemotron-CC-Math-v1_4plus_first100000.jsonl",
        help="Input JSONL path (each line: object with `text`).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/interim/nemotron_qna_extracted.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="If >0, only process first N records (for debug).",
    )
    p.add_argument(
        "--skip-unmatched",
        action="store_true",
        help="If set, do not write rows where question/answer could not be parsed.",
    )
    p.add_argument(
        "--single-pair-only",
        action="store_true",
        help="If set, only write first extracted pair for each source record.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_match_records = 0
    n_match_pairs = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.max_records and n_in >= args.max_records:
                break
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if not isinstance(text, str):
                continue

            source_id = obj.get("id")
            pairs = parse_question_answer_pairs(text)
            if args.single_pair_only and pairs:
                pairs = pairs[:1]

            if pairs:
                n_match_records += 1
                n_match_pairs += len(pairs)
                for idx, pair in enumerate(pairs):
                    out_obj = {
                        "source_id": source_id,
                        "pair_index": idx,
                        "match_kind": pair.kind,
                        "question": pair.question,
                        "answer": pair.answer,
                    }
                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    n_out += 1
                continue

            if args.skip_unmatched:
                continue
            out_obj = {
                "source_id": source_id,
                "pair_index": None,
                "match_kind": "none",
                "question": None,
                "answer": None,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        " ".join(
            [
                f"read={n_in}",
                f"written={n_out}",
                f"matched_records={n_match_records}",
                f"matched_pairs={n_match_pairs}",
                f"output={out_path}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
