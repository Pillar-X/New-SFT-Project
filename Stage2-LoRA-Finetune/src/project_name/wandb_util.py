from __future__ import annotations

import importlib.util
import os
from typing import Any


def configure_wandb_environment(config: dict[str, Any]) -> None:
    """Set ``WANDB_*`` env vars and possibly disable wandb, mirroring ``scripts/03_train_lora.py``."""
    ft_cfg = config["finetune"]
    wandb_mode = str(ft_cfg.get("wandb_mode", "online")).lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("finetune.wandb_mode must be one of: online, offline, disabled")
    if not ft_cfg.get("use_wandb", True) or wandb_mode == "disabled":
        ft_cfg["use_wandb"] = False
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

    os.environ.pop("WANDB_DISABLED", None)
    if wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        print("wandb offline mode enabled: logs are saved locally and not synced.")
    else:
        os.environ["WANDB_MODE"] = "online"

    wandb_cfg = ft_cfg.get("wandb", {}) or {}
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
    if wandb_cfg.get("name"):
        os.environ["WANDB_NAME"] = str(wandb_cfg["name"])


def log_boxed_eval_metrics(
    report: dict[str, Any],
    *,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log boxed-accuracy summary to the active wandb run (training or standalone)."""
    if importlib.util.find_spec("wandb") is None:
        return
    import wandb

    if wandb.run is None:
        return

    metrics: dict[str, Any] = {
        "boxed_eval/question_accuracy": float(report["question_accuracy"]),
        "boxed_eval/seed_accuracy": float(report["seed_accuracy"]),
        "boxed_eval/combined_accuracy": float(report["combined_accuracy"]),
        "boxed_eval/question_correct": int(report["question_correct"]),
        "boxed_eval/question_total": int(report["question_total"]),
        "boxed_eval/seed_correct": int(report["seed_correct"]),
        "boxed_eval/seed_total": int(report["seed_total"]),
        "boxed_eval/combined_correct": int(report["combined_correct"]),
        "boxed_eval/combined_total": int(report["combined_total"]),
    }
    if extra:
        for k, v in extra.items():
            if v is not None:
                metrics[f"boxed_eval/{k}"] = v
    wandb.log(metrics, step=int(step))
