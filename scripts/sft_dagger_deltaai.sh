#!/usr/bin/env bash
#SBATCH --job-name=dagger_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/dagger_sft/job_%A_%a.out
#SBATCH --partition=ghx4
#SBATCH --account=bgqz-dtai-gh
#SBATCH --array=0

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

# ---- Hyperparameter grid (3 K × 3 beta_warmup_steps = 9 jobs) -----------
K_VALUES=(16 32 64)
BETA_VALUES=(1000 10000 20000)

K_IDX=$((SLURM_ARRAY_TASK_ID / 3))
BETA_IDX=$((SLURM_ARRAY_TASK_ID % 3))

K=${K_VALUES[$K_IDX]}
BETA_WARMUP=${BETA_VALUES[$BETA_IDX]}
STRATEGY=markovian

echo "Array task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY, K=$K, beta_warmup_steps=$BETA_WARMUP"

# ---- Paths ---------------------------------------------------------------
DATA_DIR=/projects/bgqz/zzhou24/data/rstar_coder
# OUTPUT_DIR=/work/nvme/bgqz/zzhou24/checkpoints/dagger_rstar/${STRATEGY}-K${K}_beta${BETA_WARMUP}
OUTPUT_DIR=/work/hdd/bgqz/zzhou24/checkpoints/dagger_rstar/${STRATEGY}-K${K}_beta${BETA_WARMUP}

# ---- Cluster config ------------------------------------------------------
# Global batch = NGPUS × BS × GRAD_ACCUM = 4 × 8 × 4 = 128
NGPUS=4
GRAD_ACCUM=4

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29501 + SLURM_ARRAY_TASK_ID))

# Preflight
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: data directory not found: $DATA_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm/dagger_sft

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
    examples/llada/sft_dagger.py \
      --data_dir "$DATA_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --gradient_checkpointing true \
      --num_train_epochs 6 \
      --learning_rate 2e-5 \
      --warmup_steps 1000 \
      --block_size 32 \
      --K $K \
      --strategy $STRATEGY \
      --beta_warmup_steps $BETA_WARMUP \
      --logging_steps 50 \
      --eval_steps 500 \
      --save_steps 500 \
      --save_total_limit 1 \
      --report_to wandb \
      --wandb_project llada-sft \
      --wandb_name "dagger-${STRATEGY}-K${K}-beta${BETA_WARMUP}"
