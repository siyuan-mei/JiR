#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --job-name=dist_train_h100
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mail-type=NONE
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# -----------------------------
# Environment setup
module purge
module load cuda/12.6.2
module load cudnn

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_P2P_LEVEL=NVL
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-128}


# -----------------------------
# Paths and data setup
DATA_ARCHIVE=${DATA_ARCHIVE:-/path/to/mri2pet/fold1.tar}
LOCAL_DATA_DIR=$TMPDIR/mri2pet

echo "Extracting data to $LOCAL_DATA_DIR"
mkdir -p "$LOCAL_DATA_DIR"
tar -xf "$DATA_ARCHIVE" -C "$LOCAL_DATA_DIR"

export DATA_ROOT=$LOCAL_DATA_DIR/fold1
export LOG_ROOT=$TMPDIR/logs

mkdir -p $LOG_ROOT 

# -----------------------------
# Parse arguments (--config_path=..., --run_name=...)
for ARG in "$@"; do
    case $ARG in
        --config_path=*)
            CONFIG_PATH="${ARG#*=}"
            ;;
        --run_name=*)
            RUN_NAME="${ARG#*=}"
            ;;
        *)
            POSITIONAL_ARGS+=("$ARG")
            ;;
    esac
done

CONFIG_PATH=${CONFIG_PATH:-${POSITIONAL_ARGS[0]:-""}}
RUN_NAME=${RUN_NAME:-${POSITIONAL_ARGS[1]:-"default_run"}}

find_free_port() {
    while true; do
        PORT=$(( ((RANDOM<<15)|RANDOM) % 63000 + 2000 ))
        if ! lsof -i:$PORT > /dev/null 2>&1; then
            echo $PORT
            return
        fi
    done
}

MASTER_PORT=$(find_free_port)
echo "Using master port: $MASTER_PORT"

# -----------------------------
# Apptainer container path
SIF_PATH=${SIF_PATH:-/path/to/env.sif}
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}

# -----------------------------
# Run inside container
apptainer exec --nv --cleanenv \
  --env DATA_ROOT=$DATA_ROOT,LOG_ROOT=$LOG_ROOT \
  --bind "$PROJECT_ROOT":/workspace \
  --bind $TMPDIR:/tmpdir \
  $SIF_PATH \
  bash -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate synthrad
    cd /workspace
    echo 'Starting distributed training...'
    fabric run \
        --accelerator=gpu \
        --strategy=ddp_find_unused_parameters_true \
        --devices=4 \
        --num-nodes=1 \
        --node_rank=0 \
        --main-address=127.0.0.1 \
        --main-port=$MASTER_PORT \
        --precision=32-true \
        main.py --config_path='$CONFIG_PATH' --run_name='$RUN_NAME'
  "
