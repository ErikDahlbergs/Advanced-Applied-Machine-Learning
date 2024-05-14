#!/bin/bash

# -----------------------------------------------------------------------------------------------
#   TOP LEVEL TRAINING SCRIPT
# -----------------------------------------------------------------------------------------------

TRAIN_FLAGS="-i "./lab3-handout/configs/your-first-net.yml" --output "./checkpoint" --ckpt "./checkpoint""

echo "Starting conv-eval"

python "./lab3-handout/eval.py" $TRAIN_FLAGS

echo "Conv-eval ended"

# Make executable chmod +x eval.sh