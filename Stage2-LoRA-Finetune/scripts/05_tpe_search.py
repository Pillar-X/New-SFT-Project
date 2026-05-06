#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import optuna
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_name.config import load_yaml_config
from src.project_name.eval_boxed import evaluate_boxed_accuracy, load_eval_items, sample_eval_items
from src.project_name.lora_sft import create_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPE hyperparameter search for LoRA training.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="lora_tpe_lr_search")
    parser.add_argument(
        "--fixed-lr",
        type=float,
        default=None,
        help="Debug mode: use a fixed learning rate instead of TPE sampling.",
    )
    parser.add_argument(
        "--debug-single-trial",
        action="store_true",
        help="Run exactly one trial. Recommended with --fixed-lr for A/B verification.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL. Default: sqlite under outputs/hparam_search.",
    )
    return parser.parse_args()


def _load_root_env() -> None:
    load_dotenv(PROJECT_ROOT.parent / ".env")


def _setup_wandb_env(config: dict[str, Any]) -> None:
    ft_cfg = config["finetune"]
    if not ft_cfg.get("use_wandb", True):
        os.environ["WANDB_DISABLED"] = "true"
        return
    if importlib.util.find_spec("wandb") is None:
        ft_cfg["use_wandb"] = False
        os.environ["WANDB_DISABLED"] = "true"
        print("wandb not installed. Continue search without wandb tracking.")
        return
    wandb_cfg = ft_cfg.get("wandb", {})
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])


def _boxed_acc_column_name(trial_max_steps: int) -> str:
    return f"boxed_acc_step{trial_max_steps}"


def _trial_learning_rate(trial: optuna.trial.FrozenTrial) -> float | None:
    if "learning_rate" in trial.params:
        return float(trial.params["learning_rate"])
    if "learning_rate" in trial.user_attrs:
        return float(trial.user_attrs["learning_rate"])
    return None


def _write_leaderboard(study: optuna.Study, path: Path, trial_max_steps: int) -> None:
    acc_col = _boxed_acc_column_name(trial_max_steps)
    rows: list[dict[str, Any]] = []
    for t in study.trials:
        rows.append(
            {
                "trial_id": t.number,
                "learning_rate": _trial_learning_rate(t),
                acc_col: t.value,
                "status": t.state.name.lower(),
            }
        )
    rows.sort(
        key=lambda r: (
            -1.0 if r[acc_col] is None else -float(r[acc_col]),
            r["trial_id"],
        )
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial_id", "learning_rate", acc_col, "status"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(study: optuna.Study, path: Path, n_trials: int, trial_max_steps: int) -> None:
    best = study.best_trial
    acc_label = _boxed_acc_column_name(trial_max_steps)
    best_lr = _trial_learning_rate(best)
    lines = [
        "# TPE Search Summary",
        "",
        f"- Trials requested: {n_trials}",
        f"- Trials finished: {len(study.trials)}",
        f"- Best trial id: {best.number}",
        f"- Best learning_rate: {best_lr}",
        f"- Trial max steps: {trial_max_steps}",
        f"- Best {acc_label}: {best.value:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _load_root_env()

    base_config = load_yaml_config(args.config)
    _setup_wandb_env(base_config)

    search_cfg = base_config.get("hparam_search", {})
    if str(search_cfg.get("method", "")).lower() != "tpe":
        raise ValueError("Config hparam_search.method must be 'tpe'.")

    trial_max_steps = int(search_cfg.get("trial_max_steps", 199))
    disable_train_time_boxed_eval = bool(search_cfg.get("disable_train_time_boxed_eval", True))
    lr_cfg = search_cfg.get("search_space", {}).get("learning_rate", {})
    lr_min = float(lr_cfg.get("min", 1e-6))
    lr_max = float(lr_cfg.get("max", 1e-3))
    lr_log = bool(lr_cfg.get("log", True))
    fixed_lr = args.fixed_lr
    if fixed_lr is not None and fixed_lr <= 0:
        raise ValueError("--fixed-lr must be > 0.")
    if args.debug_single_trial and fixed_lr is None:
        raise ValueError("--debug-single-trial requires --fixed-lr.")

    eval_cfg = base_config["evaluation"]
    eval_items = load_eval_items(eval_cfg["test_data_path"])
    eval_sample_size = int(eval_cfg.get("sample_size", 30))
    eval_seed = int(eval_cfg.get("seed", 42))
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))
    temperature = float(eval_cfg.get("temperature", 0.0))
    do_sample = bool(eval_cfg.get("do_sample", False))

    search_root = PROJECT_ROOT / "outputs" / "hparam_search"
    search_root.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{search_root / 'tpe_study.db'}"

    sampler = optuna.samplers.TPESampler(seed=int(base_config["project"]["seed"]))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    original_epochs = float(base_config["finetune"]["num_train_epochs"])
    original_max_steps = base_config["finetune"].get("max_steps")
    acc_key = _boxed_acc_column_name(trial_max_steps)

    def objective(trial: optuna.Trial) -> float:
        trial_config = copy.deepcopy(base_config)
        if fixed_lr is not None:
            learning_rate = float(fixed_lr)
            trial.set_user_attr("fixed_lr_mode", True)
        else:
            learning_rate = trial.suggest_float("learning_rate", lr_min, lr_max, log=lr_log)

        trial_output_dir = search_root / f"trial_{trial.number:04d}"
        if trial_output_dir.exists():
            # Hard reset trial workspace to guarantee fresh training.
            shutil.rmtree(trial_output_dir)
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        trial_config["finetune"]["learning_rate"] = learning_rate
        trial_config["finetune"]["max_steps"] = trial_max_steps
        trial_config["finetune"]["output_dir"] = str(trial_output_dir)
        run_prefix = "debug-fixedlr" if fixed_lr is not None else "tpe-lr"
        trial_config["finetune"]["run_name"] = (
            f"{run_prefix}-{learning_rate:.2e}-s{trial_max_steps}-trial-{trial.number:04d}"
        )
        if disable_train_time_boxed_eval:
            trial_config.setdefault("evaluation", {})
            trial_config["evaluation"]["enable_during_training"] = False
        if trial_config["finetune"].get("wandb", {}):
            trial_config["finetune"]["wandb"]["name"] = trial_config["finetune"]["run_name"]

        trainer, tokenizer = create_trainer(trial_config)
        # Explicitly disable resume for strict from-scratch trials.
        train_result = trainer.train(resume_from_checkpoint=False)

        trainer.save_model()
        tokenizer.save_pretrained(trainer.args.output_dir)
        trainer.save_state()

        sampled_items = sample_eval_items(
            items=eval_items,
            sample_size=eval_sample_size,
            seed=eval_seed + trial.number,
        )
        # Keep search-time eval behavior aligned with training callback:
        # generation must run under eval mode.
        was_training = trainer.model.training
        trainer.model.eval()
        try:
            report = evaluate_boxed_accuracy(
                model=trainer.model,
                tokenizer=tokenizer,
                items=sampled_items,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        finally:
            if was_training:
                trainer.model.train()
        accuracy = float(report["accuracy"])

        trial_report = {
            "trial_id": trial.number,
            "learning_rate": learning_rate,
            "max_steps": trial_max_steps,
            acc_key: accuracy,
            "train_metrics": train_result.metrics,
            "boxed_eval_report": report,
        }
        (trial_output_dir / "trial_report.json").write_text(
            json.dumps(trial_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        trial.set_user_attr("output_dir", str(trial_output_dir))
        trial.set_user_attr("learning_rate", learning_rate)
        trial.set_user_attr(acc_key, accuracy)
        return accuracy

    n_trials_to_run = 1 if args.debug_single_trial else args.n_trials
    if fixed_lr is not None:
        print(f"[debug] fixed learning_rate mode enabled: {fixed_lr}")
        if args.debug_single_trial:
            print("[debug] running one trial for controlled comparison")
    study.optimize(objective, n_trials=n_trials_to_run)

    _write_leaderboard(study, search_root / "leaderboard.csv", trial_max_steps)
    _write_summary(study, search_root / "summary.md", n_trials_to_run, trial_max_steps)

    best_config = copy.deepcopy(base_config)
    best_config["finetune"]["learning_rate"] = float(study.best_trial.params["learning_rate"])
    best_config["finetune"]["num_train_epochs"] = original_epochs
    if original_max_steps is not None:
        best_config["finetune"]["max_steps"] = original_max_steps
    elif "max_steps" in best_config["finetune"]:
        del best_config["finetune"]["max_steps"]
    (search_root / "best_config.yaml").write_text(
        yaml.safe_dump(best_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best learning_rate: {_trial_learning_rate(study.best_trial)}")
    print(f"Best {acc_key}: {study.best_trial.value:.6f}")
    print(f"Leaderboard: {search_root / 'leaderboard.csv'}")
    print(f"Best config: {search_root / 'best_config.yaml'}")
    print(f"Summary: {search_root / 'summary.md'}")


if __name__ == "__main__":
    main()

