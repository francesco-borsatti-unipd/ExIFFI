#!/bin/bash

SCRIPT_PATH="test_local_scoremaps.py"

DATASETS="piade_s2_alarms"

DATASET_PATH="../../datasets/data/PIADE/"

# For TEP

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --n_estimators 300 \
    #     --contamination 0.1 \
    #     --model "EIF+" \
    #     --interpretation "EXIFFI+" \
    #     --scenario 2 \
    #     --feature1 "xmeas_11" \
    #     --feature2 "xmeas_41" \
    #     --downsample 1 \
    #     --pre_process 1 \

    # To pre process the data, add the following line, use for TEP   
    #--pre_process 1

# For PIADE

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --n_estimators 300 \
        --contamination 0.15 \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario 1 \
        --feature1 "#changes" \
        --feature2 "A_010" \
        --downsample 1 \
        --only_positive 1 \
        --factor 3 

