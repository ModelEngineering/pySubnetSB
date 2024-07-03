#!/bin/bash
# Runs separate processes to process directories
DIR="Oscillators_June_10_B_10507 Oscillators_June_10_A_11515 Oscillators_June_11_10160 Oscillators_June_11_A_2024_6877 Oscillators_June_11_B_2024_7809 Oscillators_June_9_2024_14948 Oscillators_DOE_JUNE_10_17565 Oscillators_DOE_JUNE_12_A_30917 Oscillators_DOE_JUNE_12_B_41373 Oscillators_DOE_JUNE_12_C_27662"
NUMS="100 1000 10000 100000 1000000"

for d in $DIR
do
  for n in $NUMS
    do
        python scripts/find_identity_collections.py $d $n &
    done

done
