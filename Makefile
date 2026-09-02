export HF_HOME := $(CURDIR)/.cache/hf
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export SMOLVLA_MLX_CACHE := $(CURDIR)/.cache/smolvla_mlx

TESTS ?= tests

.PHONY: goldens test bench inference-comparison profile-bf16 production-evidence cache-inventory clean-cache clean-cache-dry-run training-audit training-goldens training-parity optimizer-goldens optimizer-lockstep lora-benchmark lora-evaluation lora-finetune lora-finetune-resume lora-finetune-check training-resume-lora training-resume-full training-benchmark

goldens:
	uv run --extra reference python scripts/make_goldens.py --cache-dir $(HF_HOME) --output tests/golden
	uv run --extra reference python scripts/make_stats_active_reference.py --cache-dir $(HF_HOME) --dataset-root $(HF_HOME)/datasets/svla_so101_pickplace --output reference/artifacts/stats-active-base
	uv run --extra reference python scripts/make_goldens.py --cache-dir $(HF_HOME) --checkpoint reference/artifacts/stats-active-base --checkpoint-id lerobot/smolvla_base+svla_so101_pickplace-stats --checkpoint-revision c83c3163b8ca9b7e67c509fffd9121e66cb96205+f641879e22172be7e8161d5e6c1503c2d2feb657 --output tests/golden-stats-active
	uv run --extra reference python scripts/make_public_finetune_goldens.py --cache-dir $(HF_HOME) --output tests/golden-public-finetune

test:
	uv run --extra reference pytest $(TESTS)

bench:
	uv run python scripts/bench.py

inference-comparison:
	uv run --extra reference python scripts/benchmark_inference_comparison.py --reference-cache $(HF_HOME) --native-cache $(SMOLVLA_MLX_CACHE) --output $(CURDIR)/INFERENCE_COMPARISON.json

profile-bf16:
	uv run python scripts/profile_inference_dtypes.py --native-cache $(SMOLVLA_MLX_CACHE) --output $(CURDIR)/BF16_PROFILE.json

production-evidence:
	uv run --extra reference python scripts/production_check.py --output $(CURDIR)/.cache/production-deterministic.json
	uv run --extra reference python scripts/statistical_check.py --execution-mode strict --output $(CURDIR)/.cache/statistical-strict-production-report.json --reference-cache $(HF_HOME) --native-cache $(SMOLVLA_MLX_CACHE)/strict-production-report
	uv run --extra reference python scripts/statistical_check.py --execution-mode production --output $(CURDIR)/.cache/statistical-production.json --reference-cache $(HF_HOME) --native-cache $(SMOLVLA_MLX_CACHE)/production-statistical

cache-inventory:
	uv run python -m smolvla_mlx.cache_cleanup --repository-root $(CURDIR) --cache-dir $(SMOLVLA_MLX_CACHE) --inventory

clean-cache:
	uv run python -m smolvla_mlx.cache_cleanup --repository-root $(CURDIR) --cache-dir $(SMOLVLA_MLX_CACHE)

clean-cache-dry-run:
	uv run python -m smolvla_mlx.cache_cleanup --repository-root $(CURDIR) --cache-dir $(SMOLVLA_MLX_CACHE) --dry-run

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

training-resume-lora:
	uv run --extra train python scripts/check_training_resume.py --mode lora --dataset $(HF_HOME)/datasets/svla_so101_pickplace --cache-dir $(HF_HOME) --native-cache $(SMOLVLA_MLX_CACHE)/policy-float32 --output-root $(CURDIR)/.cache/training/t4-resume-lora

training-resume-full:
	uv run --extra train python scripts/check_training_resume.py --mode full --dataset $(HF_HOME)/datasets/svla_so101_pickplace --cache-dir $(HF_HOME) --native-cache $(SMOLVLA_MLX_CACHE)/policy-float32 --output-root $(CURDIR)/.cache/training/t4-resume-full-v2

training-benchmark:
	uv run --extra train python scripts/benchmark_training.py --dataset $(HF_HOME)/datasets/svla_so101_pickplace --cache-dir $(HF_HOME) --native-cache $(SMOLVLA_MLX_CACHE)/policy-float32 --output $(CURDIR)/.cache/training/t5-benchmark.json
