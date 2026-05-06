#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_name.config import load_yaml_config
from src.project_name.lora_sft import create_trainer


def _load_dotenv_files() -> None:
    """Load environment variables from repository root `.env` only."""
    load_dotenv(PROJECT_ROOT.parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen 0.6B LoRA with PEFT.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _setup_wandb_env(config: dict) -> None:
    ft_cfg = config["finetune"]
    if not ft_cfg.get("use_wandb", True):
        os.environ["WANDB_DISABLED"] = "true"
        return
    if importlib.util.find_spec("wandb") is None:
        ft_cfg["use_wandb"] = False
        os.environ["WANDB_DISABLED"] = "true"
        print(
            "wandb is not installed in current environment. "
            "Continue without wandb tracking. "
            "Install it with: pip install wandb"
        )
        return

    wandb_cfg = ft_cfg.get("wandb", {})
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
    if wandb_cfg.get("name"):
        os.environ["WANDB_NAME"] = str(wandb_cfg["name"])


def main() -> None:
    args = parse_args()
    _load_dotenv_files()
    config = load_yaml_config(args.config)
    _setup_wandb_env(config)

    trainer, tokenizer = create_trainer(config)
    train_result = trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"LoRA finetune done. Adapter saved to: {trainer.args.output_dir}")


if __name__ == "__main__":
    main()

