#!/bin/bash

# Path to the Python script to execute
SCRIPT_PATH="test_global_importancies.py"

# List of datasets
DATASETS="bisect_6d"


# Path to the datasets
DATASET_PATH="../data/syn/"


python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "IF" \
    --interpretation "EXIFFI" \
    --scenario 2 

python $SCRIPT_PATH \
    --dataset_name $DATASETS \
    --dataset_path $DATASET_PATH \
    --model "IF" \
    --interpretation "EXIFFI" \
    --scenario 1 

# Use pre_process ONLY ON THE NON SYNTHETIC DATASETS

# --pre_process 
