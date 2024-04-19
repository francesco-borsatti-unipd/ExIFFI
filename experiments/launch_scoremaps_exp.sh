#!/bin/bash

SCRIPT_PATH="test_local_scoremaps.py"

DATASETS="TEP"

DATASET_PATH="../../datasets/data/TEP/"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --n_estimators 500 \
        --contamination 0.1 \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --scenario 2 \
        --feature1 "xmeas_11" \
        --feature2 "xmeas_20" \
        --downsample 1 \
        --pre_process 1

    # To pre process the data, add the following line, use for TEP   
    #--pre_process 1

