#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd
import re

_output_fold = '../../outputs/output_coliva/all'

def get_mapes(output_fold):
    # Initialize results dataframes
    dfs = {}

    # Loop over devs
    for device in os.listdir(output_fold):
        print("\n", ">>>>>", device)
        device_output_fold = os.path.join(output_fold, device)
        # Initialize dataframe for the device
        df = pd.DataFrame()

        # Loop over iterations of the device
        for experiment_name in os.listdir(device_output_fold):
            integer = int(re.findall(r'\d+', experiment_name)[-1])
            results_file_path = os.path.join(device_output_fold,
                                             experiment_name, 'results')
            # Read results from file
            if not os.path.exists(results_file_path):
                continue
            with open(results_file_path, 'r') as f:
                for lin in f.readlines():
                    result_str = '      Best result for Technique.'
                    mape_str = 'Validation MAPE is '
                    if result_str in lin:
                        line_list = lin.replace(result_str, '').split(' - ')
                        tech = line_list[0]
                        mape = float(line_list[4].replace(mape_str, ''))
                        # print(integer, tech, mape, sep=" -- ")
                        df.loc[integer, tech] = mape
        dfs[device] = df.sort_index(inplace=True)
        print(df)

    return dfs


if __name__ == '__main__':
    get_mapes(_output_fold)
