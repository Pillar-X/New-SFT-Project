.PHONY: inspect filter split-eval-test train-lora eval-boxed tpe-search plot-tpe

inspect:
	python scripts/00_inspect_data.py --config configs/default.yaml

filter:
	python scripts/01_filter_data.py --config configs/default.yaml

split-eval-test:
	python scripts/02_split_eval_test.py --seed 42

train-lora:
	python scripts/03_train_lora.py --config configs/default.yaml

eval-boxed:
	python scripts/04_eval_boxed_accuracy.py --config configs/default.yaml

tpe-search:
	python scripts/05_tpe_search.py --config configs/default.yaml --n-trials 30

plot-tpe:
	python scripts/06_plot_tpe_lr_vs_accuracy.py --leaderboard outputs/hparam_search/leaderboard.csv --output outputs/hparam_search/lr_vs_accuracy.png
