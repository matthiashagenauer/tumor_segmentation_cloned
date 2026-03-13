#!/bin/bash
#SBATCH --job-name=tumour_segmentation
#SBATCH -p gpu
#SBATCH --gres=gpu:4g.47gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:00:00

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
echo "----------------------------"
echo "Running Tiling."
echo "----------------------------"

# instead of explicit filename (e.g. R20-8433_HEII_CLO.mrxs) you can also put wildcard (e.g. *.mrxs)
apptainer exec --nv \
  --bind /net/beegfs/groups/mmai/clonality/data:/data \
  tumour-segmentation_preprocessing.sif \
  bash -c 'python full_scan_segmentation_preprocess.py \
      --verbose \
      --overwrite \
      /data/raw/R20-8433_HEII_CLO.mrxs \
      /data/tumor_segmentation/output \
      /data/tumor_segmentation/cache \
      /data/tumor_segmentation/model/model-1000.tar'

echo "----------------------------"
echo "Running Inference."
echo "----------------------------"


apptainer exec --nv \
  --bind /net/beegfs/groups/mmai/clonality/data:/data \
  tumour-segmentation_inference.sif \
  bash -c 'python full_scan_segmentation_inference.py \
      --verbose \
      --overwrite \
      /data/raw/R20-8433_HEII_CLO.mrxs \
      /data/tumor_segmentation/output \
      /data/tumor_segmentation/cache \
      /data/tumor_segmentation/model/model-1000.tar'

echo "Finished at: $(date)"
