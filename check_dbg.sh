#!/bin/bash
# Checks if there are debug codes present
for f in src/pySubnetSB_tests/*.py
  do
    echo "**$f"
    grep "IGNORE_TEST = T" $f
    grep "IS_PLOT = T" $f
    grep "pdb.set_trace()" $f
  done
  #
  for f in src/analysis_tests/*.py
  do
    echo "**$f"
    grep "IGNORE_TEST = T" $f
    grep "IS_PLOT = T" $f
    grep "pdb.set_trace()" $f
  done
