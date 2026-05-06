#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_name.config import load_yaml_config
from src.project_name.eval_boxed import (
    evaluate_boxed_accuracy,
    load_eval_items,
    load_model_and_tokenizer,
    sample_eval_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate boxed answer accuracy on random sampled test set."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter path. Defaults to finetune.output_dir.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Random sample size. Defaults to evaluation.sample_size in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed. Defaults to evaluation.seed in config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT.parent / ".env")
    config = load_yaml_config(args.config)

    ft_cfg = config["finetune"]
    eval_cfg = config["evaluation"]

    adapter_path = args.adapter_path if args.adapter_path else ft_cfg["output_dir"]
    sample_size = args.sample_size if args.sample_size is not None else int(eval_cfg["sample_size"])
    seed = args.seed if args.seed is not None else int(eval_cfg["seed"])

    items = load_eval_items(eval_cfg["test_data_path"])
    sampled_items = sample_eval_items(items=items, sample_size=sample_size, seed=seed)

    model, tokenizer = load_model_and_tokenizer(
        model_path=ft_cfg["model_path"],
        adapter_path=adapter_path,
        dtype=ft_cfg.get("dtype", "auto"),
    )

    report = evaluate_boxed_accuracy(
        model=model,
        tokenizer=tokenizer,
        items=sampled_items,
        max_new_tokens=int(eval_cfg["max_new_tokens"]),
        temperature=float(eval_cfg["temperature"]),
        do_sample=bool(eval_cfg["do_sample"]),
    )

    output_dir = Path(ft_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "boxed_eval_report.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Sampled: {report['total']}")
    print(f"Correct: {report['correct']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Saved report: {out_file}")


if __name__ == "__main__":
    main()

