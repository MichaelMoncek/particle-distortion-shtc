#!/bin/bash
#SBATCH --partition=math
#SBATCH --account=math
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=plate
#SBATCH --output=slurm-%A_%a.out

apptainer exec "$HOME/julia-1.12.5.sif" julia -t auto slurm_run_task.jl
