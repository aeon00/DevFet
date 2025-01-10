#!/bin/bash
#SBATCH -J data_analysis_slam
#SBATCH -p batch
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 2-00:00:00
#SBATCH -N 1
#SBATCH -o ./niolon_batch_1.out
#SBATCH -e ./niolon_batch_1.err


# Load module and activate conda environment
niolon_interactive
module load all
conda activate slam # Replace with your environment name

# Export variables for Python script
export SLURM_ARRAY_TASK_COUNT=10

# Run Python script
python /home/dienye.h/DevFet/sandbox/full_info_plot.py