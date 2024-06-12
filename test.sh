#!/bin/bash
#SBATCH --job-name=act_collect    # Job name
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-06:00            # Runtime in D-HH:MM - probably too much
#SBATCH --partition=2080-galvani  # Partition to submit to
#SBATCH --mem=30G                 # Memory pool for all cores
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --cpus-per-task=8         # Request all 8 CPUs
# print info about current job
scontrol show job $SLURM_JOB_ID

# insert your commands here
srun \
singularity exec \
--nv \
--bind /mnt/qb/work/wichmann/wzz745 \
--bind /mnt/qb/work/wichmann/wzz745/torch_models:/torch_models/ \
--bind /mnt/qb/datasets/ImageNet2012 \
--bind /scratch_local/datasets/ImageNet2012 \
/mnt/qb/wichmann/tklein16/wichmann_container.sif \
/home/wichmann/wzz745/Network-Pruning-and-Interpretability/run.sh
echo DONE.
