#!/bin/bash
#SBATCH -t 03:00:00
#SBATCH -A edap30
#SBATCH -p gpua40
#SBATCH -J edap30-kurs11
#SBATCH -o edap30-kurs11%j.out -e edap30-kurs11_%j.err
#SBATCH --mail-user=er8251da-s@student.lu.se
#SBATCH --mail-type=FAIL,END
#SBATCH --no-requeue

# -----------------------------------------------------------------------------------------------
#   TOP LEVEL TRAINING SCRIPT
# -----------------------------------------------------------------------------------------------

TRAIN_FLAGS="--config "./lab3-handout/configs/your-first-net.yml" --ckpt "./checkpoint""

echo "Starting conv-training"

python "./lab3-handout/train.py" $TRAIN_FLAGS

echo "Conv-training ended"

# Make executable chmod +x run.sh