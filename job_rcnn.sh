#!/bin/bash
#SBATCH --job-name=P3_RCNN
#SBATCH --partition=l4
#SBATCH --exclude=ailab-l4-01,ailab-l4-05,ailab-l4-09
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/rcnn_log.txt

cd $HOME/P3
export WANDB_MODE=disabled
PASSPORT_FILE="/ceph/home/student.aau.dk/kx66ks/my_passport"

# Create log folder if it doesn't exist
mkdir -p logs

# Run inside Singularity + Virtual Env
srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif bash -c "
    echo '--- 1. Setting up Environment ---'
    # We use a fresh environment 'env_rcnn' to avoid conflicts with the pose model
    rm -rf env_rcnn
    python3 -m venv env_rcnn
    source env_rcnn/bin/activate

    echo '--- 2. Installing Packages ---'
    pip install --upgrade pip
    # Install PyTorch, Torchvision, and Pillow (required for your RCNN script)
    pip install torch torchvision numpy pillow

    echo '--- 3. Verifying Checkpoint ---'
    if [ -f 'output/best_model.pth' ]; then
        echo '✓ Found pre-trained MobileNet backbone.'
    else
        echo '⚠ WARNING: output/best_model.pth NOT FOUND.'
        echo 'The script will crash if it cannot load the backbone.'
    fi

    echo '--- 4. Running R-CNN Training ---'
    # -u ensures we see print statements immediately in the log
    python3 -u fasterrcnntest.py
"
