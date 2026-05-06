PYTHON ?= python
CONFIG ?= configs/default.yaml

.PHONY: help setup inspect filter route generate validate split pipeline test

help:
	@echo "Available targets:"
	@echo "  setup     - install dependencies"
	@echo "  inspect   - inspect raw data"
	@echo "  filter    - filter raw data"
	@echo "  route     - route documents"
	@echo "  generate  - generate SFT samples"
	@echo "  validate  - validate SFT dataset"
	@echo "  split     - build train/val split"
	@echo "  pipeline  - run all steps"
	@echo "  test      - run unit tests"

setup:
	$(PYTHON) -m pip install -r requirements.txt

inspect:
	$(PYTHON) scripts/00_inspect_data.py --config $(CONFIG)

filter:
	$(PYTHON) scripts/01_filter_data.py --config $(CONFIG)

route:
	$(PYTHON) scripts/02_route_docs.py --config $(CONFIG)

generate:
	$(PYTHON) scripts/03_generate_sft.py --config $(CONFIG)

validate:
	$(PYTHON) scripts/04_validate_sft.py --config $(CONFIG)

split:
	$(PYTHON) scripts/05_build_train_val.py --config $(CONFIG)

pipeline: inspect filter route generate validate split

test:
	$(PYTHON) -m pytest -q
