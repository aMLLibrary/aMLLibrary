#!/bin/bash

BASE_OUTPUT_FOLD=output_agora
mkdir $BASE_OUTPUT_FOLD

for app in example_configurations/agora/*; do
  APP_NAME=$(basename $app)
  mkdir $BASE_OUTPUT_FOLD/$APP_NAME
  for config in $app/*; do
    CONFIG_NO_EXT="${config%.*}"
    OUTPUT_FOLD=$BASE_OUTPUT_FOLD/$APP_NAME/$(basename $CONFIG_NO_EXT)
    echo pipenv run python ./run.py -c $config -o $OUTPUT_FOLD
    pipenv run python ./run.py -c $config -o $OUTPUT_FOLD
  done
done
