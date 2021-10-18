#!/bin/bash

CONFIG_BASE_FOLDER=example_configurations/coliva/extrapolation_vgg19
OUTPUT_ROOT=outputs/output_coliva/extrapolation_vgg19_sfs
mkdir -p $OUTPUT_ROOT

for dev in $CONFIG_BASE_FOLDER/*; do
  DEV_NAME=$(basename $dev)
  if [ "${DEV_NAME#*.}" == "ini" ]; then
    continue
  fi
  for file in $dev/*.ini; do
    OUTPUT_SUBFOLDER=$(basename "${file%.*}")
    python3 ./predict.py -m -c $file -r $OUTPUT_ROOT/sfs_training_vgg16_${DEV_NAME}_ridge/LRRidge.pickle -o $OUTPUT_ROOT/${OUTPUT_SUBFOLDER}_ridge
    python3 ./predict.py -m -c $file -r $OUTPUT_ROOT/sfs_training_vgg16_${DEV_NAME}_xgboost/XGBoost.pickle -o $OUTPUT_ROOT/${OUTPUT_SUBFOLDER}_xgboost
  done
done
