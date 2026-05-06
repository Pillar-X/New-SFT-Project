#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_name.eval_boxed_extraction import evaluate_boxed_extraction, write_eval_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate boxed-answer extraction by comparing extracted \\boxed{} against "
            "teacher model's independently generated \\boxed{} from question."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/sft_boxed_small.json",
        help="Input SFT JSON path.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env containing TEACHER_BASE_URL/API_KEY/MODEL.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="If >0, evaluate a random sample of N examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_boxed_extraction",
        help="Directory for evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details, summary = evaluate_boxed_extraction(
        input_path=args.input,
        env_path=args.env_file,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    details_path, summary_path = write_eval_outputs(
        output_dir=args.output_dir,
        details=details,
        summary=summary,
    )
    print(f"[boxed-eval] wrote details: {details_path}")
    print(f"[boxed-eval] wrote summary: {summary_path}")
    print(
        "[boxed-eval] match={ok}/{total} ({rate:.2%}) errors={err}".format(
            ok=summary["match_count"],
            total=summary["total"],
            rate=summary["match_rate"],
            err=summary["error_count"],
        )
    )


if __name__ == "__main__":
    main()
