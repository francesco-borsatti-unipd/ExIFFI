#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

DATASETS="piade_s2_all_alarms"

DATASET_PATH="../../datasets/data/PIADE/"

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --pre_process

