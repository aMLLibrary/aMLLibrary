#!/bin/bash

BASE_OUTPUT_FOLD=outputs/output_gpudata_10_hyperopt_sfs
mkdir $BASE_OUTPUT_FOLD

for config in example_configurations/gpudata/*.ini; do
  CONFIG_NO_EXT="${config%.*}"
  OUTPUT_FOLD=$BASE_OUTPUT_FOLD/$(basename $CONFIG_NO_EXT)
  echo pipenv run python ./run.py -c $config -o $OUTPUT_FOLD
  pipenv run python ./run.py -c $config -o $OUTPUT_FOLD
done
