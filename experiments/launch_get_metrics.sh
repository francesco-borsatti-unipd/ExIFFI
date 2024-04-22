#!/bin/bash

SCRIPT_PATH="get_metrics.py"

DATASETS="piade_s2"

DATASET_PATH="../../datasets/data/PIADE/"

# For PIADE

python $SCRIPT_PATH \
            --dataset_name $DATASETS \
            --dataset_path $DATASET_PATH \
            --model "EIF+" \
            --interpretation "KernelSHAP" \
            --contamination 0.01 \
            --n_estimators 300 \
            --scenario 2 


# For TEP 

# python $SCRIPT_PATH \
#             --dataset_name $DATASETS \
#             --dataset_path $DATASET_PATH \
#             --model "EIF+" \
#             --interpretation "KernelSHAP" \
#             --n_estimators 300 \
#             --scenario 2 \
#             --pre_process 