.PHONY: help serve serve-dry-run eval

help:
	@echo "Common commands:"
	@echo "  make serve         # start vLLM service with base+LoRA"
	@echo "  make serve-dry-run # print vLLM command only"
	@echo "  make eval          # run full-file boxed accuracy evaluation"

serve:
	python scripts/00_serve_vllm.py --config configs/default.yaml

serve-dry-run:
	python scripts/00_serve_vllm.py --config configs/default.yaml --dry-run

eval:
	python scripts/01_eval_boxed_accuracy.py --config configs/default.yaml

