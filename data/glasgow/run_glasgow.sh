#!/bin/bash
# Runs the glasgow solver

NUM_FILE=`ls target | wc | awk '{print $1}'`

PGM="/Users/jlheller/home/Technical/repos/glasgow-subgraph-solver/build/glasgow_subgraph_solver"
${PGM} reference/0.txt target/0.txt > /tmp/t.out
${PGM} reference/0.txt target/0.txt | grep "status = true" > out.txt

for i in {1..9}
do
  ${PGM} reference/0.txt target/0.txt | grep "status = true" >> out.txt 
done
NUM_SUCCESS=`cat out.txt | wc | awk '{print $1}'`
let FRAC_SUCCESS=${NUM_SUCCESS}/${NUM_FILE}
echo ${FRAC_SUCCESS}
