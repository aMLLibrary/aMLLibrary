#!/bin/bash

BASE_OUTPUT_FOLD=outputs/output_gpudata_11_extrap_P_hyperopt
mkdir $BASE_OUTPUT_FOLD

for config in example_configurations/gpudata/extrapolation_P/*.ini; do
  CONFIG_NO_EXT="${config%.*}"
  OUTPUT_FOLD=$BASE_OUTPUT_FOLD/$(basename $CONFIG_NO_EXT)
  echo python3 ./run.py -c $config -o $OUTPUT_FOLD
  python3 ./run.py -c $config -o $OUTPUT_FOLD
done
