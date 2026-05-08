#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
from src.project_name.wandb_util import configure_wandb_environment, log_boxed_eval_metrics


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
    configure_wandb_environment(config)

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

    wandb_active = False
    if ft_cfg.get("use_wandb", True):
        try:
            import wandb
        except ImportError:
            wandb = None  # type: ignore[assignment]
        if wandb is not None and not os.environ.get("WANDB_DISABLED"):
            wandb_cfg = ft_cfg.get("wandb", {}) or {}
            entity_raw = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
            entity = entity_raw if entity_raw else None
            project = (
                wandb_cfg.get("project")
                or os.environ.get("WANDB_PROJECT")
                or "stage2-lora"
            )
            base_name = wandb_cfg.get("name") or ft_cfg.get("run_name") or "boxed-eval"
            wandb.init(
                project=str(project),
                entity=entity,
                name=f"{base_name}-boxed-eval",
                mode=os.environ.get("WANDB_MODE", "online"),
                job_type="boxed_eval",
                config={
                    "adapter_path": adapter_path,
                    "sample_size": sample_size,
                    "seed": seed,
                    "script": "04_eval_boxed_accuracy",
                },
            )
            wandb_active = True
            log_boxed_eval_metrics(report, step=0, extra={"adapter_path": adapter_path})

    try:
        output_dir = Path(ft_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "boxed_eval_report.json"
        out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Sampled rows: {len(sampled_items)}")
        print(
            f"Question accuracy: {report['question_accuracy']:.4f} "
            f"({report['question_correct']}/{report['question_total']})"
        )
        print(
            f"Seed accuracy: {report['seed_accuracy']:.4f} "
            f"({report['seed_correct']}/{report['seed_total']})"
        )
        print(
            f"Combined accuracy: {report['combined_accuracy']:.4f} "
            f"({report['combined_correct']}/{report['combined_total']})"
        )
        print(f"Saved report: {out_file}")
    finally:
        if wandb_active:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()

