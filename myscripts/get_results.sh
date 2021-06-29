#!/bin/bash
APP=blockscholes  # blockscholes bodytrack freqmine kmeans stereomatch(100!) swaptions

mkdir $APP
for i in {1..40}; do
  SUBFOLDER=${APP}_itr$i
  scp guindani@srv-ardagna2.deib.polimi.it:/home/guindani/aml/output_agora/$APP/$SUBFOLDER/results $APP/${SUBFOLDER}_results.txt
  sleep 2
done
