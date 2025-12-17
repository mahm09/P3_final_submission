#!/bin/bash
#SBATCH --job-name=GEN_GRAPHS
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/graphs_log.txt
#SBATCH --exclude=ailab-l4-05

BASE_DIR="/ceph/home/student.aau.dk/kx66ks"
PASSPORT_FILE="${BASE_DIR}/my_passport"

cd "${BASE_DIR}/P3"

srun singularity exec -B $PASSPORT_FILE:/etc/passwd --nv /ceph/container/pytorch/pytorch_25.11.sif bash -c "
    source env_rcnn/bin/activate
    pip install seaborn pandas matplotlib tqdm
    python3 generate_advanced_plots.py
"