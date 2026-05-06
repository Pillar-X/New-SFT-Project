#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_name.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start vLLM OpenAI server with base model + LoRA.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print command; do not launch server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(str(PROJECT_ROOT / args.config))
    vllm_cfg = config["vllm"]
    model_cfg = config["model"]

    base_model = str(PROJECT_ROOT / model_cfg["base_model_path"])
    lora_path = str(PROJECT_ROOT / model_cfg["lora_path"])
    if not Path(base_model).exists():
        raise FileNotFoundError(f"Base model path not found: {base_model}")
    if not Path(lora_path).exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    served_model_name = str(model_cfg["served_model_name"])
    lora_alias = str(model_cfg.get("lora_alias", "lora_adapter"))

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        str(vllm_cfg.get("host", "0.0.0.0")),
        "--port",
        str(vllm_cfg.get("port", 8000)),
        "--model",
        base_model,
        "--served-model-name",
        served_model_name,
        "--dtype",
        str(vllm_cfg.get("dtype", "auto")),
        "--tensor-parallel-size",
        str(vllm_cfg.get("tensor_parallel_size", 1)),
        "--max-model-len",
        str(vllm_cfg.get("max_model_len", 4096)),
        "--gpu-memory-utilization",
        str(vllm_cfg.get("gpu_memory_utilization", 0.9)),
        "--trust-remote-code",
        "--enable-lora",
        "--lora-modules",
        f"{lora_alias}={lora_path}",
    ]

    if bool(vllm_cfg.get("enforce_eager", False)):
        command.append("--enforce-eager")
    if bool(vllm_cfg.get("disable_log_stats", True)):
        command.append("--disable-log-stats")

    extra_args = vllm_cfg.get("extra_args", [])
    if not isinstance(extra_args, list):
        raise ValueError("vllm.extra_args must be a list")
    command.extend(str(x) for x in extra_args)

    printable = " ".join(shlex.quote(x) for x in command)
    print(f"[vllm] command:\n{printable}")
    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

