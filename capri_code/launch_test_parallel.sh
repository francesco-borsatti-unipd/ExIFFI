#!/bin/bash

# create a variable for the directory where to save the results                
DIR="../../container/job_c/results_c_final"
NTHREADS=12

# Set the OpenMP environment variable to NTHREADS

export OMP_NUM_THREADS=$NTHREADS

# if the dir doesn't exist, create it                                          
if [ ! -d "$DIR" ]; then                                                       
    mkdir -p "$DIR"                                                            
fi

printf "Executing test_parallel script:\n"

#DATASETS="moodify"
DATASETS="wine glass cardio pima breastw ionosphere annthyroid pendigits diabetes shuttle moodify ads"

python "test_parallel.py" \
    --wrapper \
    --savedir $DIR \
    --seed 120  \
    --dataset_names $DATASETS \
    --n_runs 1  \
    --n_runs_imps 1 \
    --n_cores $NTHREADS \
    --n_trees 600 \
    --add_bash \
    --use_C \
    #--C_fast