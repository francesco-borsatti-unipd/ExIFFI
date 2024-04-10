#!/bin/bash

SCRIPT_PATH="test_global_importancies.py"

DATASETS="piade_s2"

DATASET_PATH="../../datasets/data/PIADE/"

SCENARIOS=(1 2)

python $SCRIPT_PATH \
        --dataset_name $DATASETS \
        --dataset_path $DATASET_PATH \
        --model "EIF+" \
        --interpretation "EXIFFI+" \
        --pre_process \
        --scenario 1

# for scenario in "${SCENARIOS[@]}"; do

#     python $SCRIPT_PATH \
#         --dataset_name $DATASETS \
#         --dataset_path $DATASET_PATH \
#         --model "EIF+" \
#         --interpretation "EXIFFI+" \
#         --scenario "$scenario" 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --interpretation "EXIFFI" \
    #     --scenario "$scenario" 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "IF" \
    #     --interpretation "DIFFI" \
    #     --scenario "$scenario" 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF+" \
    #     --interpretation "RandomForest" \
    #     --scenario "$scenario"

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "EIF" \
    #     --interpretation "RandomForest" \
    #     --scenario "$scenario" 

    # python $SCRIPT_PATH \
    #     --dataset_name $DATASETS \
    #     --dataset_path $DATASET_PATH \
    #     --model "IF" \
    #     --interpretation "RandomForest" \
    #     --scenario "$scenario"

#done
