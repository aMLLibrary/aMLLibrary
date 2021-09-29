#!/bin/bash

BASE_OUTPUT_FOLD=outputs/output_coliva
mkdir $BASE_OUTPUT_FOLD

for app in example_configurations/coliva/*; do
  APP_NAME=$(basename $app)
  if [ "$APP_NAME" == "blueprint.ini" ]; then
    continue
  fi
  mkdir $BASE_OUTPUT_FOLD/$APP_NAME
  for config in $app/*; do
    CONFIG_NO_EXT="${config%.*}"
    OUTPUT_FOLD=$BASE_OUTPUT_FOLD/$APP_NAME/$(basename $CONFIG_NO_EXT)
    echo python3 ./run.py -c $config -o $OUTPUT_FOLD
    #python3 ./run.py -c $config -o $OUTPUT_FOLD
  done
done
