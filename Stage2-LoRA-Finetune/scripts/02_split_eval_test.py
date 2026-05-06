#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split jsonl samples into eval and test with fixed seed."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/valid_1000.jsonl"),
        help="Input jsonl file path.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/eval/valid_800.jsonl"),
        help="Output jsonl file path for eval split.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/test/valid_200.jsonl"),
        help="Output jsonl file path for test split.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.8,
        help="Eval split ratio. The rest goes to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic shuffling.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=1000,
        help="Expected number of input samples for safety check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    if args.expected_total > 0 and total != args.expected_total:
        raise ValueError(
            f"Expected {args.expected_total} samples, but got {total}: {args.input}"
        )

    if not (0 < args.eval_ratio < 1):
        raise ValueError("--eval-ratio must be between 0 and 1.")

    shuffled = lines[:]
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)

    eval_count = int(total * args.eval_ratio)
    eval_lines = shuffled[:eval_count]
    test_lines = shuffled[eval_count:]

    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)

    args.eval_output.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
    args.test_output.write_text("\n".join(test_lines) + "\n", encoding="utf-8")

    print(f"Input: {args.input} ({total} lines)")
    print(f"Seed: {args.seed}")
    print(f"Eval: {args.eval_output} ({len(eval_lines)} lines)")
    print(f"Test: {args.test_output} ({len(test_lines)} lines)")


if __name__ == "__main__":
    main()
