from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_name.eval_teacher import evaluate_sample, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample extracted QA pairs and evaluate with teacher LLM."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/nemotron_qna_matched_only.jsonl",
        help="Input extracted QA JSONL.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of rows to sample for teacher evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env that contains teacher model settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_qna_teacher",
        help="Directory to save evaluation results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    judged, summary = evaluate_sample(
        input_path=args.input,
        sample_size=args.sample_size,
        seed=args.seed,
        env_path=args.env_file,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "judged_details.jsonl"
    summary_path = out_dir / "summary.json"

    write_jsonl(details_path, judged)
    write_json(summary_path, summary)

    print(f"[eval] wrote details: {details_path}")
    print(f"[eval] wrote summary: {summary_path}")
    print(
        "[eval] both_ok={both_ok}/{total} ({rate:.2%}), extraction_ok={eok}/{total}, answer_ok={aok}/{total}, errors={err}".format(
            both_ok=summary["both_ok_count"],
            total=summary["total"],
            rate=summary["both_ok_rate"],
            eok=summary["extraction_ok_count"],
            aok=summary["answer_ok_count"],
            err=summary["error_count"],
        )
    )


if __name__ == "__main__":
    main()
