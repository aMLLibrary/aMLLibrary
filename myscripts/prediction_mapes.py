#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd
import re

_output_fold = '../../zz_old/coliva/outputs/extrapolation_vgg19'

def get_prediction_mapes(output_fold):
    # Initialize results dataframes
    dfs = {}

    # Loop over devices
    for device in os.listdir(output_fold):
        if device.startswith('training'):
            continue
        print("\n", ">>>>>", device)
        device_output_fold = os.path.join(output_fold, device)
        # Initialize dataframe for the device
        df = pd.DataFrame()

        # Loop over iterations of the device
        for experiment_name in os.listdir(device_output_fold):
            print(experiment_name)
            _, integer, technique = experiment_name.split('_')
            integer = int(integer)
            results_file_path = os.path.join(device_output_fold,
                                             experiment_name, 'mape.txt')
            # Read results from file
            if not os.path.exists(results_file_path):
                continue
            with open(results_file_path, 'r') as f:
                mape = f.read().strip('\n')
            df.loc[integer, technique] = float(mape)

        df.sort_index(inplace=True)
        dfs[device] = df

    return dfs


if __name__ == '__main__':
    get_prediction_mapes(_output_fold)
