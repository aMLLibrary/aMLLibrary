#!/bin/bash

BASE_OUTPUT_FOLD="outputs/gpudata/drapersmith"
rm -rf $BASE_OUTPUT_FOLD
mkdir  $BASE_OUTPUT_FOLD

for conf in example_configurations/gpudata/stepwise_extrapolation_N/*/*.ini; do
  NAME=$(basename $conf)
  NAME="${NAME%.*}"
  OUTPUT_FOLD=$BASE_OUTPUT_FOLD/$NAME
  echo python3 ./run.py -c $conf -o $OUTPUT_FOLD $@
  python3 ./run.py -c $conf -o $OUTPUT_FOLD $@
done
