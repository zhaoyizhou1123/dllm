# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dLLM** is a unified training and evaluation framework for diffusion language models (dLLMs), supporting LLaDA, Dream, A2D, BERT-Chat, EditFlow, Fast-dLLM, and GRPO reinforcement learning.

## Setup

```bash
conda activate <env>        # Activate the project conda environment
pip install -e .            # Editable install
```

Core dependencies: `transformers==4.57.0`, `accelerate==1.11.0`, `deepspeed==0.18.0`, `peft==0.17.1`, `datasets==4.2.0`.

## Commands

### Running Tests
```bash
pytest scripts/tests/ -v                        # All tests
pytest scripts/tests/test_schedulers.py -v      # Single test file
pytest scripts/tests/ -k "test_name" -v         # Single test by name
```

### Training (local multi-GPU)
```bash
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml examples/llada/sft.py \
    --model_name_or_path <model> --dataset_name <dataset> ...
```

Available accelerate configs in `scripts/accelerate_configs/`: `cpu.yaml`, `ddp.yaml`, `fsdp.yaml`, `fsdp2.yaml`, `zero1.yaml`, `zero2.yaml`, `zero3.yaml`.

### Training (Slurm HPC)
```bash
sbatch scripts/train.slurm.sh
```

### Data Preprocessing
```bash
python dllm/tools/preprocess_pt_dataset.py ...   # Pretraining data
python dllm/tools/preprocess_sft_dataset.py ...  # SFT data
python dllm/tools/merge_peft_adapter.py ...      # Merge LoRA adapters
```

### Evaluation
Evaluation integrates with `lm-evaluation-harness` (git submodule). See `examples/<pipeline>/eval.sh` for per-pipeline eval commands.

## Architecture

### Package Layout

```
dllm/
├── core/           # Reusable, pipeline-agnostic building blocks
│   ├── samplers/   # BaseSampler → MDLMSampler, BD3LMSampler
│   ├── schedulers/ # Alpha/Kappa schedulers (Linear, Cosine, Cubic)
│   ├── trainers/   # MDLMTrainer, BD3LMTrainer (extend transformers.Trainer)
│   └── eval/       # Base evaluation logic
├── pipelines/      # Model-specific implementations
│   ├── llada/      # LLaDA 1.x (models, sampler, trainer, eval)
│   ├── llada2/     # LLaDA 2.0 inference
│   ├── llada21/    # LLaDA 2.1 inference
│   ├── dream/      # Dream model
│   ├── a2d/        # AR-to-Diffusion conversion (Llama, Qwen2, Qwen3)
│   ├── bert/       # BERT-Chat fine-tuning
│   ├── editflow/   # EditFlow (insert/delete/substitute operations)
│   ├── fastdllm/   # Fast-dLLM with KV-cache + confidence decoding
│   └── rl/         # GRPO reinforcement learning training
├── utils/          # Shared utilities (configs, collators, data, models, sampling, viz)
├── data/           # Dataset loading utilities
└── tools/          # Data preprocessing and model management scripts
examples/           # Entry-point scripts mirroring pipelines/ structure
scripts/
├── tests/          # pytest test suite
├── accelerate_configs/  # Distributed training configs
└── train.slurm.sh  # Slurm job submission template
lm-evaluation-harness/  # Git submodule for evaluation
```

### Key Abstractions

**Samplers** (`dllm/core/samplers/base.py`): `BaseSampler` defines the `sample()` and `infill()` interface. Concrete samplers (MDLM, BD3LM) implement masked/block diffusion decoding. Output includes sequences and optional sampling histories for visualization.

**Schedulers** (`dllm/core/schedulers/`): `AlphaScheduler` and `KappaScheduler` control diffusion timestep weighting during training. Pass a scheduler instance when constructing a trainer.

**Trainers** (`dllm/core/trainers/`): `MDLMTrainer` and `BD3LMTrainer` extend `transformers.Trainer` with diffusion-specific loss weighting and NLL/PPL metric tracking. Each pipeline can subclass these further (e.g., `dllm/pipelines/llada/trainer.py`).

**Configs** (`dllm/utils/configs.py`): `DataArguments`, `ModelArguments`, `TrainingArguments` — all training scripts use these via HuggingFace `HfArgumentParser`.

**Utilities** (`dllm/utils/`): `get_model()` / `get_tokenizer()` handle loading; `sample_trim()` / `infill_trim()` are sampling helpers; `visualizers.py` provides terminal and video visualization of the iterative denoising process.

### Training Script Pattern

Every training entry point in `examples/` follows this structure:
1. Parse `ModelArguments`, `DataArguments`, `TrainingArguments` with `HfArgumentParser`
2. Load model + tokenizer via `get_model()` / `get_tokenizer()`
3. Prepare dataset (tokenize → group/clip → collate)
4. Instantiate trainer (e.g., `MDLMTrainer`) with a scheduler
5. Call `trainer.train()`

### GPU Execution (HPC)

When running GPU tasks on the cluster, use `srun` with explicit resource requests (see `AGENTS.md` for flags). Always activate the conda environment before launching.

## Reference Repositories

- `~/projects/mdm_correction` — **read-only** reference repo containing features and improvements to be ported into this repo. When implementing new functionality, check this repo for relevant prior work before writing from scratch.

## Development Notes

- Use absolute paths in markdown and docstrings.
- New pipeline files should include a docstring at the top with run instructions.
- Reuse `dllm/core/` components rather than duplicating logic in pipeline-specific code.
- Code style is enforced by `black` (model directories excluded per `pyproject.toml`).
