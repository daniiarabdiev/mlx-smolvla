export HF_HOME := $(CURDIR)/.cache/hf
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export SMOLVLA_MLX_CACHE := $(CURDIR)/.cache/smolvla_mlx

TESTS ?= tests

.PHONY: goldens test bench training-audit training-goldens training-parity optimizer-goldens optimizer-lockstep lora-benchmark lora-evaluation lora-finetune lora-finetune-resume lora-finetune-check

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

optimizer-lockstep:
	uv run python scripts/check_optimizer_lockstep.py --t1-goldens $(CURDIR)/.cache/training/gradient_goldens --optimizer-goldens $(CURDIR)/.cache/training/optimizer_goldens --native-cache $(CURDIR)/.cache/smolvla_mlx/policy-float32 --output $(CURDIR)/.cache/training/t2-lockstep.json

lora-benchmark:
	scripts/finetune_lora --benchmark-only --cache-dir $(HF_HOME) --native-cache $(CURDIR)/.cache/smolvla_mlx/policy-float32 --benchmark-output $(CURDIR)/.cache/training/t3-benchmark.json

lora-evaluation:
	uv run --extra reference python scripts/make_lora_evaluation.py --cache-dir $(HF_HOME) --native-cache $(CURDIR)/.cache/smolvla_mlx/policy-float32 --evaluation-dir $(CURDIR)/.cache/training/t3-evaluation --output $(CURDIR)/.cache/training/t3-base-evaluation.json

lora-finetune:
	scripts/finetune_lora --checkpoint-interval 100 --cache-dir $(HF_HOME) --native-cache $(CURDIR)/.cache/smolvla_mlx/policy-float32 --output $(CURDIR)/.cache/training/t3b --lora-scope expert_only --budget-mode fixed_steps --launch-config $(CURDIR)/.cache/training/t3b/launch.json --log-file $(CURDIR)/.cache/training/t3b/training.log

lora-finetune-resume:
	scripts/finetune_lora --resume --checkpoint-interval 100 --cache-dir $(HF_HOME) --native-cache $(CURDIR)/.cache/smolvla_mlx/policy-float32 --output $(CURDIR)/.cache/training/t3b --lora-scope expert_only --budget-mode fixed_steps --launch-config $(CURDIR)/.cache/training/t3b/launch.json --log-file $(CURDIR)/.cache/training/t3b/training.log

lora-finetune-check:
	uv run --extra reference python scripts/check_lora_finetune.py --cache-dir $(HF_HOME) --native-cache $(CURDIR)/.cache/smolvla_mlx/policy-float32 --run-dir $(CURDIR)/.cache/training/t3 --evaluation-dir $(CURDIR)/.cache/training/t3-evaluation --base-report $(CURDIR)/.cache/training/t3-base-evaluation.json --output $(CURDIR)/.cache/training/t3-outcome.json
