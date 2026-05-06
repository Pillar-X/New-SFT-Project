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

from project_name.eval_bucket_answer import evaluate_bucket_samples
from project_name.eval_teacher import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample each length bucket and evaluate whether answer correctly "
            "answers question using teacher LLM."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/interim/length_buckets",
        help="Directory containing small/middle/large/super-large JSONL files.",
    )
    parser.add_argument(
        "--sample-per-bucket",
        type=int,
        default=10,
        help="Number of samples per bucket.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to env file containing teacher model configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_length_buckets_answer",
        help="Directory for output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details, summary = evaluate_bucket_samples(
        input_dir=args.input_dir,
        sample_per_bucket=args.sample_per_bucket,
        seed=args.seed,
        env_path=args.env_file,
    )

    details_path = output_dir / "judged_details.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(details_path, details)
    write_json(summary_path, summary)

    overall = summary.get("overall", {})
    print(f"[bucket-eval] wrote details: {details_path}")
    print(f"[bucket-eval] wrote summary: {summary_path}")
    print(
        "[bucket-eval] total={total} answer_ok={ok}/{total} ({rate:.2%}) errors={err}".format(
            total=overall.get("total", 0),
            ok=overall.get("answer_ok_count", 0),
            rate=float(overall.get("answer_ok_rate", 0.0)),
            err=overall.get("error_count", 0),
        )
    )
    print("[bucket-eval] bucket breakdown:")
    for name in ("small", "middle", "large", "super-large"):
        b = summary.get("buckets", {}).get(name, {})
        total = int(b.get("total", 0))
        ok = int(b.get("answer_ok_count", 0))
        rate = float(b.get("answer_ok_rate", 0.0))
        print(f"  - {name}: {ok}/{total} ({rate:.2%})")


if __name__ == "__main__":
    main()
