#!/bin/bash
#SBATCH --partition=math
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --job-name=plate
#SBATCH --output=slurm-%A_%a.out
export JULIA_NUM_THREADS=${SLURM_CPUS_ON_NODE}

ch-run --bind=$HOME:/mnt/0 julia-1.12.5 -- \
    /usr/local/julia/bin/julia -t ${JULIA_NUM_THREADS} slurm_run_task.jl
