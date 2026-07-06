#!/bin/bash
#SBATCH --array=0-1
julia -t auto slurm_run_task.jl
