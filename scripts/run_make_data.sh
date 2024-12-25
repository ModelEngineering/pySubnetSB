#!/bin/bash
# $1 - number of processes
NUM_PROCESSES=$1
for (( idx=0; idx<NUM_PROCESSES; idx++ ))
do
    python make_data.py $NUM_PROCESSES $idx &
done