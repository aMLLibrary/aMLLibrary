#!/usr/bin/env python3
import os
import pandas as pd
import configparser
import numpy as np

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

SFS = True
ALL_LAYERS = False

# Initialize list of devices
layers_to_keep = (3,5,6,8,9,10,12,13,14,16,17,18)
old_to_new_name = {
    'Odroid___VGG16': 'odroid',
    'RaspberryPi3___VGG16': 'rp3',
    'TegraX2___VGG16': 'tegrax2'
}
devices = tuple(old_to_new_name.keys())

# Initialize relevant paths
datasets_folder = os.path.join('inputs', 'coliva')
configs_folder = os.path.join('example_configurations', 'coliva', 'next')
blueprint_file_name = 'blueprint_sfs.ini' if SFS else 'blueprint.ini'
blueprint_file_path = os.path.join('example_configurations', 'coliva',
                                      blueprint_file_name)

# Loop over devices
for dev in devices:
  print("\n", ">>>>>", dev)
  # Get files paths
  dataset_dev_subfolder = os.path.join(datasets_folder, dev)
  new_name = old_to_new_name[dev]
  config_dev_subfolder = os.path.join(configs_folder, new_name)

  # Create subfolder
  if not os.path.isdir(config_dev_subfolder):
    os.mkdir(config_dev_subfolder)

  # Loop over files of different iterations
  for lay in layers_to_keep:
    dataset_file_path = os.path.join(dataset_dev_subfolder,
                                     f'j{lay}_ML_input.csv')
    it00 = str(lay).zfill(2)
    config_file_name = f'sfs_{new_name}_{it00}.ini' if SFS else \
                           f'{new_name}_{it00}.ini'
    config_file_path = os.path.join(config_dev_subfolder,
                                    config_file_name)

    # Read blueprint configuration file (refreshed at each iteration)
    config = configparser.ConfigParser()
    config.read(blueprint_file_path)

    # Modify config
    config['DataPreparation']['input_path'] = f'"{dataset_file_path}"'
    config['General']['extrapolation_columns'] = str({"Layer": lay-1})

    # Save config to file
    with open(config_file_path, 'w') as f:
      config.write(f)
    print("Saved configs to", config_file_path)
