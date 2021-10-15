#!/bin/bash

CONFIG_BASE_FOLDER=example_configurations/coliva/extrapolation_vgg19
OUTPUT_BASE_FOLDER=outputs/output_coliva/extrapolation_vgg19_sfs
mkdir -p $OUTPUT_BASE_FOLDER

for file in $CONFIG_BASE_FOLDER/sfs_*.ini; do
    OUTPUT_SUBFOLDER=$(basename "${file%.*}")
    echo python3 ./run.py -c $file -o $OUTPUT_BASE_FOLDER/$OUTPUT_SUBFOLDER
    python3 ./run.py -c $file -o $OUTPUT_BASE_FOLDER/$OUTPUT_SUBFOLDER
done
