#!/bin/bash

SCRIPT_PATH="test_feature_selection.py"

DATASETS="TEP_ACME"

# Path to the datasets 
DATASET_PATH="../../datasets/data/TEP/"

# For PIADE 

# python $SCRIPT_PATH \
#     --dataset_name $DATASETS \
#     --dataset_path $DATASET_PATH \
#     --n_estimators 300 \
#     --contamination 0.01 \
#     --model "EIF+" \
#     --model_interpretation "EIF+" \
#     --interpretation "EXIFFI+" \
#     --scenario 1 \
#     --compute_random 

# For TEP 

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --n_estimators 300 \
    --contamination 0.01 \
    --model "EIF+" \
    --model_interpretation "EIF+" \
    --interpretation "EXIFFI+" \
    --scenario 2 \
    --pre_process 

