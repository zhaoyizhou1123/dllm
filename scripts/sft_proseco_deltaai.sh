#!/usr/bin/env bash
#SBATCH --job-name=proseco_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/proseco_sft/job_%A.out
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh

#SBATCH --chdir=/u/zzhou24/projects/dllm

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# module load default
# module load cuda/12.9.0
source ~/miniconda3/bin/activate
conda activate smdm2

# NCCL fix for deltaai cluster
unset NCCL_NET_PLUGIN
LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/sw/user/nccl/' | tr '\n' ':' | sed 's/:*$//')
export LD_LIBRARY_PATH

export HF_HOME=/projects/bgqz/zzhou24/.cache/huggingface
export PYTHONPATH=/u/zzhou24/projects/dllm

# ---- Paths ---------------------------------------------------------------
DATA_DIR=/projects/bgqz/zzhou24/data/rstar_coder
OUTPUT_DIR=/work/nvme/bgqz/zzhou24/checkpoints/proseco_rstar

# ---- Cluster config ------------------------------------------------------
# GH200 nodes: up to 4 GPUs per node (ghx4), typically 2 available.
# 2 GPUs: global batch = 2 × 8 × 8 = 128
# 4 GPUs: set NGPUS=4 and GRAD_ACCUM=4 for same global batch
NGPUS=2
# ProSeCo does 2 forward passes per step, so halve batch size vs standard SFT.
# Global batch = NGPUS × BS × GRAD_ACCUM = 2 × 4 × 16 = 128
GRAD_ACCUM=16

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

# Preflight
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: data directory not found: $DATA_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/proseco_sft

# ---- Launch ---------------------------------------------------------------
# Uses FSDP v2 (better with torch 2.8 nightly on GH200).
# Gradient checkpointing is DISABLED in the script for speed; 96GB HBM3e
# on each GH200 is enough: ~64GB (params+grads+optimizer) + ~10GB activations.
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
