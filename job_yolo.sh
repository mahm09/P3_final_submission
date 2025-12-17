#!/bin/bash
#SBATCH --job-name=P3_YOLO
#SBATCH --partition=l4
#SBATCH --exclude=ailab-l4-01,ailab-l4-09
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/yolo_log.txt

cd $HOME/P3
export WANDB_MODE=disabled
PASSPORT_FILE="/ceph/home/student.aau.dk/kx66ks/my_passport"

# --- THE SANDWICH FIX ---
echo "Cleaning up..."
# 1. Remove EVERYTHING related to OpenCV to start fresh
srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif pip uninstall -y opencv-python opencv-python-headless

echo "Installing Ultralytics..."
# 2. Install Ultralytics (This will annoyingly auto-install the BAD opencv-python)
srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif pip install --user ultralytics

echo "Swapping to Headless..."
# 3. Remove the BAD one again
srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif pip uninstall -y opencv-python

# 4. Install the GOOD one LAST (Force reinstall ensures the files are actually written)
srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif pip install --user --force-reinstall opencv-python-headless

# --- RUNNING ---
echo "Starting YOLO Training..."
srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif python3 yolotrain.py