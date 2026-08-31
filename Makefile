export HF_HOME := $(CURDIR)/.cache/hf
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export SMOLVLA_MLX_CACHE := $(CURDIR)/.cache/smolvla_mlx

TESTS ?= tests

.PHONY: goldens test bench training-audit training-goldens training-parity optimizer-goldens

goldens:
	uv run --extra reference python scripts/make_goldens.py --cache-dir $(HF_HOME) --output tests/golden

test:
	uv run --extra reference pytest $(TESTS)

bench:
	uv run python scripts/bench.py

training-audit:
	uv run python scripts/training_feasibility.py --seed 0 --output $(CURDIR)/.cache/training/t0-audit.json

training-goldens:
	uv run --extra reference python scripts/make_training_goldens.py --cache-dir $(HF_HOME) --output $(CURDIR)/.cache/training/gradient_goldens

training-parity:
	uv run python scripts/check_gradient_parity.py --goldens $(CURDIR)/.cache/training/gradient_goldens --native-cache $(CURDIR)/.cache/smolvla_mlx/policy-float32 --output $(CURDIR)/.cache/training/t1-parity.json

optimizer-goldens:
	uv run --extra reference python scripts/make_optimizer_goldens.py --cache-dir $(HF_HOME) --t1-goldens $(CURDIR)/.cache/training/gradient_goldens --output $(CURDIR)/.cache/training/optimizer_goldens
