#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

DATASETS="piade_s2_not_alarms"

DATASET_PATH="../../datasets/data/PIADE/"

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "EIF+" \
    --interpretation "EXIFFI+" \
    --scenario 1 \
    --n_estimators 300 \
    --contamination 0.15 \
    --n_runs 40

