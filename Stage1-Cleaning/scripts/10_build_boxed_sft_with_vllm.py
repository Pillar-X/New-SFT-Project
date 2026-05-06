#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_name.vllm_boxed_sft import (  # noqa: E402
    build_sft_dataset,
    check_student_api_ready,
    load_student_config,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SFT JSON (messages only) by extracting boxed final answers with local vLLM model."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/interim/nemotron_qna_numeric_only.jsonl",
        help="Input JSONL dataset path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/sft_boxed_final_answer.json",
        help="Output SFT JSON path.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env containing STUDENT_BASE_URL / STUDENT_API_KEY / STUDENT_MODEL.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, only process first N samples (for debug).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for local model API.",
    )
    parser.add_argument(
        "--max-retry",
        type=int,
        default=2,
        help="Max retry per sample when API call fails.",
    )
    parser.add_argument(
        "--allow-api-fail",
        action="store_true",
        help="If set, API failures fallback to \\boxed{None} instead of raising error.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_student_config(env_path=args.env_file)
    print(
        f"[build-boxed-sft] using endpoint={cfg.base_url} model={cfg.model}",
        flush=True,
    )
    check_student_api_ready(cfg)
    print("[build-boxed-sft] vLLM API health check passed.", flush=True)

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    sft_rows = build_sft_dataset(
        rows=rows,
        cfg=cfg,
        temperature=args.temperature,
        max_retry=args.max_retry,
        fail_on_error=not args.allow_api_fail,
        progress_every=args.progress_every,
    )
    write_json(args.output, sft_rows)

    print(
        f"[build-boxed-sft] input={args.input} total_in={len(rows)} "
        f"total_out={len(sft_rows)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
