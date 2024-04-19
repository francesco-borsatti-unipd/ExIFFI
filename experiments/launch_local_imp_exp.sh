#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

DATASETS="TEP"

DATASET_PATH="../../datasets/data/TEP/"

python $SCRIPT_PATH \
        --n_estimators 500 \
        --contamination 0.01 \
        --model "EIF+" \
        --dataset_path $DATASET_PATH \
        --dataset_name $DATASETS \
        --interpretation "EXIFFI+" \
        --scenario 2 \
        --n_runs 40 \
        --pre_process

# To pre_process the data, add the following line:
# --pre_process \

# To get the labels, add the following line. With the new configuration this makes sense
# only if n_runs=1
# --get_labels \