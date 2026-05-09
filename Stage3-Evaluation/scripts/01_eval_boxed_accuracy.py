#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _compact_review_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row.get("index"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "question_pred_boxed": row.get("question_pred_boxed"),
        "question_model_response": row.get("question_model_response"),
        "seed_question": row.get("seed_question"),
        "seed_answer": row.get("seed_answer"),
        "seed_pred_boxed": row.get("seed_pred_boxed"),
        "seed_is_correct": row.get("seed_is_correct"),
        "seed_model_response": row.get("seed_model_response"),
    }


def _select_review_rows(
    results: list[dict[str, Any]],
    *,
    n_each: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pick random incorrect/correct rows by ``question_is_correct``."""
    wrong_pool = [r for r in results if not r.get("question_is_correct")]
    right_pool = [r for r in results if r.get("question_is_correct")]
    k_w = min(max(0, n_each), len(wrong_pool))
    k_r = min(max(0, n_each), len(right_pool))
    wrong_pick = rng.sample(wrong_pool, k_w) if k_w else []
    right_pick = rng.sample(right_pool, k_r) if k_r else []
    return wrong_pick, right_pick


def _write_review_markdown(
    path: Path,
    *,
    incorrect: list[dict[str, Any]],
    correct: list[dict[str, Any]],
    criterion: str,
) -> None:
    lines: list[str] = [
        "# Boxed eval — review samples",
        "",
        f"Main criterion: **{criterion}** (seed question metrics are shown for context when present).",
        "",
        "---",
        "",
        "## Incorrect samples",
        "",
    ]
    if not incorrect:
        lines.append("_No incorrect rows under this criterion._\n")
    for i, row in enumerate(incorrect, start=1):
        idx = row.get("index")
        lines.append(f"### {i}. Row index `{idx}` — question wrong")
        lines.extend(
            [
                "",
                "**Question**",
                "",
                "```text",
                str(row.get("question", "")),
                "```",
                "",
                "**Gold answer**",
                "",
                "```text",
                str(row.get("gold_answer", "")),
                "```",
                "",
                "**Predicted (extracted \\boxed{})**",
                "",
                "```text",
                str(row.get("question_pred_boxed")),
                "```",
                "",
                "**Full model response (question)**",
                "",
                "```text",
                str(row.get("question_model_response", "")),
                "```",
                "",
            ]
        )
        if row.get("seed_question"):
            lines.extend(
                [
                    "**Seed question**",
                    "",
                    "```text",
                    str(row.get("seed_question", "")),
                    "```",
                    "",
                    "**Seed gold**",
                    "",
                    "```text",
                    str(row.get("seed_answer", "")),
                    "```",
                    "",
                    "**Seed predicted \\boxed{}**",
                    "",
                    "```text",
                    str(row.get("seed_pred_boxed")),
                    "```",
                    "",
                    f"**Seed correct:** `{row.get('seed_is_correct')}`",
                    "",
                    "**Full model response (seed)**",
                    "",
                    "```text",
                    str(row.get("seed_model_response") or ""),
                    "```",
                    "",
                ]
            )
        lines.append("---")
        lines.append("")

    lines.extend(["## Correct samples", ""])
    if not correct:
        lines.append("_No correct rows under this criterion._\n")
    for i, row in enumerate(correct, start=1):
        idx = row.get("index")
        lines.append(f"### {i}. Row index `{idx}` — question correct")
        lines.extend(
            [
                "",
                "**Question**",
                "",
                "```text",
                str(row.get("question", "")),
                "```",
                "",
                "**Gold answer**",
                "",
                "```text",
                str(row.get("gold_answer", "")),
                "```",
                "",
                "**Predicted (extracted \\boxed{})**",
                "",
                "```text",
                str(row.get("question_pred_boxed")),
                "```",
                "",
                "**Full model response (question)**",
                "",
                "```text",
                str(row.get("question_model_response", "")),
                "```",
                "",
            ]
        )
        if row.get("seed_question"):
            lines.extend(
                [
                    "**Seed question / gold / predicted**",
                    "",
                    "```text",
                    str(row.get("seed_question", "")),
                    "```",
                    "",
                    "```text",
                    str(row.get("seed_answer", "")),
                    "```",
                    "",
                    "```text",
                    str(row.get("seed_pred_boxed")),
                    "```",
                    "",
                    f"**Seed correct:** `{row.get('seed_is_correct')}`",
                    "",
                ]
            )
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
        concurrent_requests=int(eval_cfg.get("concurrent_requests", 1)),
        progress_every_requests=eval_cfg.get("progress_every_requests", 100),
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

    review_each = int(eval_cfg.get("review_samples_each", 5))
    review_seed = int(eval_cfg.get("review_samples_seed", config["project"].get("seed", 42)))
    rng = random.Random(review_seed)
    wrong_pick, right_pick = _select_review_rows(report["results"], n_each=review_each, rng=rng)
    review_payload = {
        "selection": {
            "criterion": "question_is_correct",
            "n_requested_each": review_each,
            "n_incorrect_selected": len(wrong_pick),
            "n_correct_selected": len(right_pick),
            "random_seed": review_seed,
        },
        "incorrect": [_compact_review_row(dict(r)) for r in wrong_pick],
        "correct": [_compact_review_row(dict(r)) for r in right_pick],
    }
    (run_dir / "review_samples.json").write_text(
        json.dumps(review_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_review_markdown(
        run_dir / "review_samples.md",
        incorrect=[_compact_review_row(dict(r)) for r in wrong_pick],
        correct=[_compact_review_row(dict(r)) for r in right_pick],
        criterion="question_is_correct",
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
        f"review={run_dir / 'review_samples.md'} "
        f"out={run_dir}"
    )


if __name__ == "__main__":
    main()

