#!/bin/bash

SCRIPT_PATH="test_metrics.py"

# List of datasets
DATASETS="piade_s2"

# Path to the datasets 
DATASET_PATH="../../datasets/data/PIADE/"

# For TEP 

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --model "EIF+" \
#     --interpretation "KernelSHAP" \
#     --n_estimators 300 \
#     --scenario 2 \
#     --pre_process 1 \
#     --compute_GFI 1 \
#     --n_runs_imp 5 \
#     --background 0.25 

# For PIADE

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "EIF+" \
    --interpretation "KernelSHAP" \
    --n_estimators 300 \
    --contamination 0.01 \
    --scenario 2 \
    --compute_GFI 1 \
    --n_runs_imp 5 \
    --background 0.5