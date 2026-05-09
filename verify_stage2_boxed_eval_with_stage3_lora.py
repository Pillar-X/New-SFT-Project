#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STAGE2_ROOT = ROOT / "Stage2-LoRA-Finetune"
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))

from src.project_name.config import load_yaml_config
from src.project_name.eval_boxed import (
    evaluate_boxed_accuracy,
    load_eval_items,
    load_model_and_tokenizer,
    sample_eval_items,
)


def _resolve(path_like: str, *, base: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Stage2 boxed-eval by using the same sampled eval set "
            "and loading a specified LoRA adapter (for example from Stage3-Evaluation/lora)."
        )
    )
    parser.add_argument(
        "--stage2-config",
        type=str,
        required=True,
        help=(
            "Stage2 config path used in training, e.g. "
            "Stage2-LoRA-Finetune/configs/lr_e-6.yaml"
        ),
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help=(
            "Adapter directory path. Can be absolute, or relative to repo root. "
            "Example: Stage3-Evaluation/lora/lora-e-5-1700steps"
        ),
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default=None,
        help=(
            "Shortcut for Stage3 adapter name under Stage3-Evaluation/lora/. "
            "Ignored when --lora-path is provided."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Override torch device, e.g. cuda:1")
    parser.add_argument("--dtype", type=str, default=None, help="Override dtype, e.g. auto/bfloat16/float16")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override evaluation.sample_size from Stage2 config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override evaluation.seed from Stage2 config.",
    )
    parser.add_argument(
        "--expected-combined-accuracy",
        type=float,
        default=None,
        help="Optional expected combined accuracy for consistency check.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for expected accuracy check.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full report JSON.",
    )
    return parser.parse_args()


def _choose_adapter_path(args: argparse.Namespace) -> Path:
    if args.lora_path:
        adapter = _resolve(args.lora_path, base=ROOT)
    elif args.lora_name:
        adapter = (ROOT / "Stage3-Evaluation" / "lora" / args.lora_name).resolve()
    else:
        raise ValueError("Provide either --lora-path or --lora-name.")
    if not adapter.exists():
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter}")
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"adapter_config.json not found under adapter path: {adapter}"
        )
    return adapter


def main() -> None:
    args = _parse_args()
    stage2_config_path = _resolve(args.stage2_config, base=ROOT)
    if not stage2_config_path.exists():
        raise FileNotFoundError(f"Stage2 config not found: {stage2_config_path}")

    config = load_yaml_config(str(stage2_config_path))
    ft_cfg: dict[str, Any] = dict(config.get("finetune", {}))
    eval_cfg: dict[str, Any] = dict(config.get("evaluation", {}))
    if not ft_cfg or not eval_cfg:
        raise ValueError("Invalid config: missing finetune/evaluation sections")

    model_path = _resolve(str(ft_cfg["model_path"]), base=STAGE2_ROOT)
    adapter_path = _choose_adapter_path(args)
    test_data_path = _resolve(str(eval_cfg["test_data_path"]), base=STAGE2_ROOT)
    sample_size = int(args.sample_size if args.sample_size is not None else eval_cfg["sample_size"])
    seed = int(args.seed if args.seed is not None else eval_cfg["seed"])

    max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))
    temperature = float(eval_cfg.get("temperature", 0.0))
    do_sample = bool(eval_cfg.get("do_sample", False))
    eval_batch_size = int(eval_cfg.get("batch_size", 1))
    dtype = str(args.dtype if args.dtype is not None else ft_cfg.get("dtype", ft_cfg.get("torch_dtype", "auto")))
    device = args.device
    if device is None:
        if str(eval_cfg.get("async_backend", "none")).lower() == "ray":
            ray_cfg = eval_cfg.get("ray", {}) or {}
            device = str(ray_cfg.get("eval_device", "cuda:1"))
        else:
            device = "cuda"

    print("=== Stage2-aligned boxed-eval verification ===")
    print(f"stage2_config: {stage2_config_path}")
    print(f"model_path: {model_path}")
    print(f"adapter_path: {adapter_path}")
    print(f"test_data_path: {test_data_path}")
    print(f"sample_size: {sample_size}")
    print(f"seed: {seed}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"temperature: {temperature}")
    print(f"do_sample: {do_sample}")
    print(f"eval_batch_size: {eval_batch_size}")
    print(f"dtype: {dtype}")
    print(f"device: {device}")

    items = load_eval_items(str(test_data_path))
    sampled_items = sample_eval_items(items=items, sample_size=sample_size, seed=seed)
    sample_fingerprint = hashlib.sha256(
        "\n".join(x.question for x in sampled_items).encode("utf-8")
    ).hexdigest()[:16]
    print(f"sample_fingerprint: {sample_fingerprint}")

    model, tokenizer = load_model_and_tokenizer(
        model_path=str(model_path),
        adapter_path=str(adapter_path),
        dtype=dtype,
        device=device,
    )
    report = evaluate_boxed_accuracy(
        model=model,
        tokenizer=tokenizer,
        items=sampled_items,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        eval_batch_size=eval_batch_size,
    )

    metrics = {
        "question_accuracy": report["question_accuracy"],
        "seed_accuracy": report["seed_accuracy"],
        "combined_accuracy": report["combined_accuracy"],
        "question_correct": report["question_correct"],
        "question_total": report["question_total"],
        "seed_correct": report["seed_correct"],
        "seed_total": report["seed_total"],
        "combined_correct": report["combined_correct"],
        "combined_total": report["combined_total"],
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output_json:
        out_path = _resolve(args.output_json, base=ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "stage2_config": str(stage2_config_path),
                "model_path": str(model_path),
                "adapter_path": str(adapter_path),
                "test_data_path": str(test_data_path),
                "sample_size": sample_size,
                "seed": seed,
                "sample_fingerprint": sample_fingerprint,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "eval_batch_size": eval_batch_size,
                "dtype": dtype,
                "device": device,
            },
            "metrics": metrics,
            "report": report,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_output: {out_path}")

    if args.expected_combined_accuracy is not None:
        diff = abs(float(report["combined_accuracy"]) - float(args.expected_combined_accuracy))
        print(f"expected_combined_accuracy: {args.expected_combined_accuracy}")
        print(f"abs_diff: {diff}")
        if diff > float(args.atol):
            raise SystemExit(
                f"FAILED: combined_accuracy mismatch (diff={diff} > atol={args.atol})"
            )
        print("PASS: combined_accuracy matches expected value within tolerance.")


if __name__ == "__main__":
    main()
