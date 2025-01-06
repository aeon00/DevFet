#!/bin/sh
#SBATCH -J SpangyAnalysis
#SBATCH -A b219
#SBATCH -p skylake
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2-00:00:00
#SBATCH --array=0-9%5
#SBATCH -e %x_%A_%a.err
#SBATCH -o %x_%A_%a.out
#SBATCH --mem=64G  # Requesting 64GB of memory


# Load module and activate conda environment
module load userspace/all
source ~/miniconda3/bin/activate
conda activate devfet  # Replace with your environment name

# Create necessary directories
mkdir -p /scratch/hdienye/spangy/{plots,textures,results}

# Export variables for Python script
export SLURM_ARRAY_TASK_COUNT=10

# Run Python script
python /scratch/hdienye/codes/full_info_plot.py