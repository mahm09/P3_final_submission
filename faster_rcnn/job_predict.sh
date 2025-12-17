#!/bin/bash
#SBATCH --job-name=P3_TEST
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/predict_log.txt

cd $HOME/P3
PASSPORT_FILE="/ceph/home/student.aau.dk/kx66ks/my_passport"

srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif bash -c "
    source env_rcnn/bin/activate
    python3 predict_rcnn.py
"