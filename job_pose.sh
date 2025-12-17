#!/bin/bash
#SBATCH --job-name=P3_Pose
#SBATCH --partition=l4
#SBATCH --exclude=ailab-l4-01,ailab-l4-05,ailab-l4-09
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/pose_log.txt

cd $HOME/P3
export WANDB_MODE=disabled
PASSPORT_FILE="/ceph/home/student.aau.dk/kx66ks/my_passport"

# Run inside Singularity (for system drivers) + Virtual Env (for python packages)
srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif bash -c "
    echo '--- 1. Setting up Virtual Environment ---'
    rm -rf env
    python3 -m venv env
    source env/bin/activate

    echo '--- 2. Installing Heavy Packages ---'
    pip install --upgrade pip
    # We install everything. This WILL pull in the wrong OpenCVs. That is expected.
    pip install 'numpy<2' mediapipe seaborn matplotlib tqdm torch torchvision ultralytics scikit-learn

    echo '--- 3. THE PURGE (Fixing Conflicts) ---'
    # Force delete EVERY OpenCV version to clean the slate
    pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

    echo '--- 4. Installing The Correct OpenCV ---'
    # Install ONLY the headless version compatible with server & mediapipe
    pip install 'opencv-python-headless<4.8'

    echo '--- 5. Final Version Check ---'
    pip list | grep opencv

    echo '--- 6. Running Code ---'
    # We use -u to force logs to appear instantly
    python3 -u pose-mobilnet_v2.py
"