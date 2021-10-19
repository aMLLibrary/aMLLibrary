#!/usr/bin/env python3
import configparser
import numpy as np
import os
import pandas as pd
import re

"""
Blueprint script to produce .ini files, which can also be integrated to produce
.csv files.
"""

_configs_fold = 'example_configurations/coliva/all'
_datasets_fold = 'inputs/coliva'
_blueprint_config_path = 'example_configurations/coliva/blueprint.ini'
_devices = ('Odroid___VGG16', 'RaspberryPi3___VGG16', 'TegraX2___VGG16')


def produce_files(configs_fold, datasets_fold, blueprint_config_path, devices):
    # Allows running this script from both this folder and from root folder
    if os.getcwd() == os.path.dirname(__file__):
        os.chdir(os.pardir)

    # Loop over devices
    for device in devices:
        print("\n", ">>>>>", device)
        # Get and create folder paths for the device
        device_dataset_fold = os.path.join(datasets_fold, device)
        device_config_fold = os.path.join(configs_fold, device)
        if not os.path.isdir(device_config_fold):
            os.mkdir(device_config_fold)

        # Loop over files of different iterations for the device
        for dataset_name in os.listdir(device_dataset_fold):
            # Find integer in the dataset name
            integer = int(re.findall(r'\d+', dataset_name)[-1])
            int00 = str(integer).zfill(3)
            # Build paths for dataset and config file
            dataset_file_path = os.path.join(device_dataset_fold, dataset_name)
            config_file_name = int00 + '.ini'
            config_file_path = os.path.join(device_config_fold,
                                            config_file_name)
            # Read blueprint configuration file (refreshed at each iteration)
            config = configparser.ConfigParser()
            config.read(blueprint_config_path)

            # Change config
            config['DataPreparation']['input_path'] = f'"{dataset_file_path}"'
            # !!! change here other specific entries, if any !!!

            # Save config to file
            with open(config_file_path, 'w') as f:
                config.write(f)
            print("Saved to", config_file_path)


if __name__ == '__main__':
    produce_files(_configs_fold, _datasets_fold, _blueprint_config_path,
                  _devices)
