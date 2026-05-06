#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_name.boxed_eval import evaluate_boxed_accuracy, load_eval_items, maybe_limit_items
from src.project_name.config import load_yaml_config
from src.project_name.vllm_openai import VllmOpenAIClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full eval file with boxed accuracy via vLLM.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def _build_run_dir(output_root: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{prefix}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = load_yaml_config(str(config_path))

    eval_cfg = config["evaluation"]
    model_cfg = config["model"]
    output_root = PROJECT_ROOT / eval_cfg.get("output_dir", "evaluation")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _build_run_dir(output_root, "boxed-eval")

    items = load_eval_items(str(PROJECT_ROOT / eval_cfg["input_path"]))
    items = maybe_limit_items(
        items=items,
        max_samples=eval_cfg.get("max_samples"),
        seed=int(config["project"].get("seed", 42)),
    )
    print(f"[eval] loaded_rows={len(items)}")

    client = VllmOpenAIClient(
        base_url=str(model_cfg["base_url"]),
        api_key=str(model_cfg.get("api_key", "EMPTY")),
        model=str(model_cfg["served_model_name"]),
        max_new_tokens=int(eval_cfg.get("max_new_tokens", 512)),
        temperature=float(eval_cfg.get("temperature", 0.0)),
        do_sample=bool(eval_cfg.get("do_sample", False)),
        timeout_seconds=int(eval_cfg.get("timeout_seconds", 180)),
    )

    report = evaluate_boxed_accuracy(
        items=items,
        generate_fn=client.generate,
        progress_every_rows=eval_cfg.get("progress_every_rows"),
    )

    metrics = {
        "question_accuracy": report["question_accuracy"],
        "seed_accuracy": report["seed_accuracy"],
        "combined_accuracy": report["combined_accuracy"],
        "question_total": report["question_total"],
        "seed_total": report["seed_total"],
        "combined_total": report["combined_total"],
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in report["results"]),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    command_text = f"python {' '.join(shlex.quote(x) for x in sys.argv)}"
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    print(
        "[boxed-eval] "
        f"question_acc={report['question_accuracy']:.4f} "
        f"seed_acc={report['seed_accuracy']:.4f} "
        f"combined_acc={report['combined_accuracy']:.4f} "
        f"rows={report['question_total']} seed_rows={report['seed_total']} "
        f"out={run_dir}"
    )


if __name__ == "__main__":
    main()

