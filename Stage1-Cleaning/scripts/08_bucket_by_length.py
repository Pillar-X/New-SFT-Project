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

from project_name.length_bucket import assign_bucket, combined_qa_token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bucket QA pairs by total question+answer token length."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/nemotron_qna_numeric_only.jsonl",
        help="Input JSONL after numeric-only filtering.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/length_buckets",
        help="Directory to save bucketed JSONL files.",
    )
    parser.add_argument(
        "--keep-token-field",
        action="store_true",
        help="If set, append `total_tokens` to each output row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {
        "small": output_dir / "small.jsonl",
        "middle": output_dir / "middle.jsonl",
        "large": output_dir / "large.jsonl",
        "super-large": output_dir / "super-large.jsonl",
    }
    writers = {k: p.open("w", encoding="utf-8") for k, p in out_paths.items()}

    total = 0
    kept = 0
    dropped = 0
    bucket_counts = {k: 0 for k in out_paths}

    try:
        with input_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue

                question = row.get("question")
                answer = row.get("answer")
                if not isinstance(question, str) or not isinstance(answer, str):
                    dropped += 1
                    continue

                total_tokens = combined_qa_token_count(question, answer)
                bucket = assign_bucket(total_tokens)
                if bucket is None:
                    dropped += 1
                    continue

                if args.keep_token_field:
                    row["total_tokens"] = total_tokens
                writers[bucket].write(json.dumps(row, ensure_ascii=False) + "\n")
                bucket_counts[bucket] += 1
                kept += 1
    finally:
        for f in writers.values():
            f.close()

    kept_rate = kept / total if total else 0.0
    print(
        " ".join(
            [
                "[length-bucket]",
                f"total={total}",
                f"kept={kept}",
                f"dropped={dropped}",
                f"kept_rate={kept_rate:.2%}",
                f"small={bucket_counts['small']}",
                f"middle={bucket_counts['middle']}",
                f"large={bucket_counts['large']}",
                f"super-large={bucket_counts['super-large']}",
                f"output_dir={output_dir}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
