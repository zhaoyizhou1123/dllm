#!/usr/bin/env bash
#SBATCH --job-name=proseco_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/proseco_sft/job_%A.out
#SBATCH --partition=HGPU
#SBATCH --gres=gpu:h200:2

#SBATCH --chdir=/home/zhaoyiz/projects/dllm

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ---- Conda env (no apptainer on this run) --------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dllm

export HF_HOME=/home/zhaoyiz/huggingface
export PYTHONPATH=/home/zhaoyiz/projects/dllm

# ---- Paths ---------------------------------------------------------------
DATA_DIR=/home/zhaoyiz/projects/data/rstar_coder
OUTPUT_DIR=checkpoints/proseco_rstar

# ---- Cluster config ------------------------------------------------------
# HGPU H100 nodes: up to 8 GPUs per node; requesting 2 here.
# ProSeCo does 2 forward passes per step, so halve batch size vs standard SFT.
# Global batch = NGPUS × BS × GRAD_ACCUM = 2 × 4 × 16 = 128
NGPUS=2
GRAD_ACCUM=16

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

# Preflight
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: data directory not found: $DATA_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/proseco_sft

# ---- Launch ---------------------------------------------------------------
srun --nodes=1 --ntasks=1 \
  accelerate launch \
    --config_file scripts/accelerate_configs/fsdp2.yaml \
    --num_machines 1 \
    --num_processes $NGPUS \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --rdzv_backend c10d \
    examples/llada/sft_proseco.py \
      --data_dir "$DATA_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --gradient_checkpointing true \
      --num_train_epochs 6 \
      --learning_rate 2e-5 \
      --warmup_steps 1000 \
      --logging_steps 50 \
      --eval_steps 500 \
      --save_steps 500 \
      --save_total_limit 2 \
      --resume_from_checkpoint latest \
      --report_to wandb
