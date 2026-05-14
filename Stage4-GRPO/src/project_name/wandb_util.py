from __future__ import annotations

import importlib.util
import os
from typing import Any


def configure_wandb_environment(config: dict[str, Any]) -> None:
    train_cfg = config["training"]
    wandb_mode = str(train_cfg.get("wandb_mode", "online")).lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("training.wandb_mode must be one of: online, offline, disabled")
    if not train_cfg.get("use_wandb", True) or wandb_mode == "disabled":
        train_cfg["use_wandb"] = False
        os.environ["WANDB_DISABLED"] = "true"
        return

    if importlib.util.find_spec("wandb") is None:
        train_cfg["use_wandb"] = False
        os.environ["WANDB_DISABLED"] = "true"
        print(
            "wandb is not installed in current environment. "
            "Continue without wandb tracking. "
            "Install it with: pip install wandb"
        )
        return

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_MODE"] = "offline" if wandb_mode == "offline" else "online"

    wandb_cfg = train_cfg.get("wandb", {}) or {}
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
    if wandb_cfg.get("name"):
        os.environ["WANDB_NAME"] = str(wandb_cfg["name"])
