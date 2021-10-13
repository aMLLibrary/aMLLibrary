#!/bin/bash

CONFIG_BASE_FOLDER=example_configurations/coliva/extrapolation_vgg19
OUTPUT_ROOT=outputs/output_coliva/extrapolation_vgg19
mkdir -p $OUTPUT_ROOT

for dev in $CONFIG_BASE_FOLDER/*; do
  DEV_NAME=$(basename $dev)
  if [ "${DEV_NAME#*.}" == "ini" ]; then
    continue
  fi
  for file in $dev/*.ini; do
    OUTPUT_SUBFOLDER=$(basename "${file%.*}")
    #echo python3 ./predict.py -c $file -o $OUTPUT_ROOT/$OUTPUT_SUBFOLDER

    echo python3 ./predict.py -m -c $file -r $OUTPUT_ROOT/training_vgg16_${DEV_NAME}_ridge/LRRidge.pickle -o $OUTPUT_ROOT/${OUTPUT_SUBFOLDER}_ridge
    echo python3 ./predict.py -m -c $file -r $OUTPUT_ROOT/training_vgg16_${DEV_NAME}_xgbooost/XGBoost.pickle -o $OUTPUT_ROOT/${OUTPUT_SUBFOLDER}_xgboost
  done
done
