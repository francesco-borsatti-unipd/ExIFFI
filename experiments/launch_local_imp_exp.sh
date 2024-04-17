#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

DATASETS="piade_s2"

DATASET_PATH="../../datasets/data/PIADE/"

python $SCRIPT_PATH \
        --n_estimators 300 \
        --contamination 0.01 \
        --model "EIF" \
        --dataset_path $DATASET_PATH \
        --dataset_name $DATASETS \
        --interpretation "EXIFFI" \
        --scenario 1 \
        --get_labels 

# To pre_process the data, add the following line:
# --pre_process \