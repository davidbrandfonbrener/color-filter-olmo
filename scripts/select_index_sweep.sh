#!/bin/bash
#SBATCH --job-name=data-olmo
#SBATCH --account=kempner_fellows
#SBATCH --output=/n/holyscratch01/sham_lab/data-olmo/logs/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1     
#SBATCH --cpus-per-task=24
#SBATCH --time=1:00:00
#SBATCH --mem=250GB		
#SBATCH --partition=kempner
#SBATCH --array=1-4
#SBATCH --exclude=holygpu8a19604

# Custom environment
source ~/.bashrc
conda deactivate
conda activate color-filter

export CONFIG=configs/base.yaml
export SWEEP_CONFIG=configs/sweeps/gen-idx-parallel.yaml
export CHECKPOINTS_PATH=/n/holyscratch01/sham_lab/data-olmo/data

# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python scripts/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG} select_index=True