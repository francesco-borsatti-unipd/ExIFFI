#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

DATASETS="piade_s2"

DATASET_PATH="../../datasets/data/PIADE/"

python $SCRIPT_PATH \
        --pre_process \
        --contamination 0.05 \
        --n_estimators 300 \
        --model "EIF+" \
        --dataset_path $DATASET_PATH \
        --dataset_name $DATASETS \
        --interpretation "EXIFFI+" 