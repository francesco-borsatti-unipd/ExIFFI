#!/bin/bash

SCRIPT_PATH="test_local_importances.py"

DATASETS="shuttle"

DATASET_PATH="../data/real/"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --pre_process 1 \
        --scenario 2 \
        --feats_plot "(3,0)"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --pre_process 1 \
        --scenario 1 \
        --feats_plot "(3,0)"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --pre_process 1 \
        --scenario 2 \
        --feats_plot "(3,0)"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF" \
        --interpretation "EXIFFI" \
        --pre_process 1 \
        --scenario 1 \
        --feats_plot "(3,0)"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --pre_process 1 \
        --scenario 2 \
        --feats_plot "(3,0)"

    python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "IF" \
        --interpretation "DIFFI" \
        --pre_process 1 \
        --scenario 1 \
        --feats_plot "(3,0)"

