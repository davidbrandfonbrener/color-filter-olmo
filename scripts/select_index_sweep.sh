#!/bin/bash
#SBATCH --job-name=color-filter
#SBATCH --output=logs/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1     
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=250GB		
#SBATCH --constraint=h100

# Custom environment
source ~/.bashrc
conda deactivate
conda activate color-filter

export CONFIG=configs/base.yaml

# Accept sweep config as argument
export SWEEP_CONFIG=$1

# Accept job index as argument if there is a second argument
if [ -z "$2" ]
then
    echo $SLURM_ARRAY_TASK_ID
else
    export SLURM_ARRAY_TASK_ID=$2
fi

# Set default path for checkpoints if not set
if [ -z "$CHECKPOINTS_PATH" ]
then
    export CHECKPOINTS_PATH=ckpts
fi

# Set ntasks if not set
if [ -z "$SLURM_NTASKS_PER_NODE" ]
then
    export SLURM_NTASKS_PER_NODE=1
fi

# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python scripts/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG} select_index=True