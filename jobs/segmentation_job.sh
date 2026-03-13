#!/bin/bash
#SBATCH --job-name=tumour_segmentation
#SBATCH -p gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=52G
#SBATCH --time=24:00:00

#SBATCH --output=jobs/output/runner_logs/%x_%j.out
#SBATCH --error=jobs/output/runner_logs/%x_%j.err

# Create log directory if it does not exist
mkdir -p jobs/output/runner_logs

echo "===== Job Information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Started at: $(date)"
echo "==========================="

module load cuda/11.7.64

# Optional but recommended for PyTorch performance
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export CUDA_VISIBLE_DEVICES=0

echo "Running tumour segmentation."
apptainer exec --nv \
  --bind /net/beegfs/groups/mmai/clonality/data:/data \
  tumour-segmentation_v2.sif \
  bash -c 'python full_scan_segmentation.py \
      --verbose \
      --overwrite \
      /data/raw/*.mrxs \
      /data/tumor_segmentation/output \
      /data/tumor_segmentation/cache \
      /data/tumor_segmentation/model/model-1000.tar'

echo "Finished at: $(date)"
