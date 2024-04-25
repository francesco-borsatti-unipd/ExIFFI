#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

DATASETS="TEP_ACME"

DATASET_PATH="../../datasets/data/TEP/"

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "IF" \
    --interpretation "DIFFI" \
    --scenario 2 \
    --n_estimators 300 \
    --contamination 0.15 \
    --n_runs 40 \
    --pre_process

