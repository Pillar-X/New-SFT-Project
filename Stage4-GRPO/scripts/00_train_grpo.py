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
from src.project_name.grpo import train_grpo
from src.project_name.wandb_util import configure_wandb_environment


def _load_dotenv_files() -> None:
    load_dotenv(PROJECT_ROOT.parent / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue LoRA training with Group Relative Policy Optimization (GRPO)."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_dotenv_files()
    config = load_yaml_config(args.config)
    configure_wandb_environment(config)
    train_grpo(config)


if __name__ == "__main__":
    main()
