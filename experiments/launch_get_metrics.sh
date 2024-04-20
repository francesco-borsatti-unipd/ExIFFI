#!/bin/bash

SCRIPT_PATH="get_metrics.py"

DATASETS="piade_s2"

DATASET_PATH="../../datasets/data/PIADE/"

python $SCRIPT_PATH \
            --dataset_name $DATASETS \
            --dataset_path $DATASET_PATH \
            --model "EIF+" \
            --interpretation "EXIFFI+" \
            --n_estimators 300 \
            --pre_process \
            --scenario 2