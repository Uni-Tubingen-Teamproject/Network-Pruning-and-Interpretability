#!/bin/bash

#SBATCH --job-name=name           # Job name
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM - takes about an hour per epoch on 2 GPUs
#SBATCH --partition=2080-galvani  # Partition to submit to
#SBATCH --mem=40G                 # Memory pool for all cores
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --cpus-per-task=8        # Request all 8 CPUs per GPU
#SBATCH --mail-user=jonathan.sakouhi@gmail.com  # Email address for notifications
#SBATCH --mail-type=BEGIN,END,FAIL  # When to send email notifications
# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# run the actual command
srun \
singularity exec \
--nv \
--bind /mnt/lustre/datasets/ \
--bind /mnt/qb/work/wichmann/wzz745 \
--bind /scratch_local/ \
/mnt/qb/work/wichmann/wzz745/ffcv.sif \
./train.sh

