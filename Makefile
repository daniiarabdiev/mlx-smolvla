export HF_HOME := $(CURDIR)/.cache/hf
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export SMOLVLA_MLX_CACHE := $(CURDIR)/.cache/smolvla_mlx

TESTS ?= tests

.PHONY: goldens test bench

goldens:
	uv run --extra reference python scripts/make_goldens.py

test:
	uv run --extra reference pytest $(TESTS)

bench:
	uv run python scripts/bench.py
