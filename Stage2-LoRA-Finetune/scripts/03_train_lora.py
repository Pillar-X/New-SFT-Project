#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_name.config import load_yaml_config
from src.project_name.lora_sft import create_trainer
from src.project_name.wandb_util import configure_wandb_environment


def _load_dotenv_files() -> None:
    """Load environment variables from repository root `.env` only."""
    load_dotenv(PROJECT_ROOT.parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen 0.6B LoRA with PEFT.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_dotenv_files()
    config = load_yaml_config(args.config)
    configure_wandb_environment(config)

    trainer, tokenizer = create_trainer(config)
    train_result = trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(trainer.args.output_dir)

    ft_cfg = config["finetune"]
    stamp = str(ft_cfg.get("_lora_run_stamp", "")).strip()
    if stamp and bool(ft_cfg.get("lora_checkpoint_symlinks", False)):
        gs = int(trainer.state.global_step)
        final_named = Path(trainer.args.output_dir) / f"lora-{stamp}-step-{gs}-final"
        trainer.save_model(str(final_named))
        tokenizer.save_pretrained(str(final_named))
        print(f"LoRA named copy (final): {final_named}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print(f"LoRA finetune done. Adapter saved to: {trainer.args.output_dir}")


if __name__ == "__main__":
    main()

