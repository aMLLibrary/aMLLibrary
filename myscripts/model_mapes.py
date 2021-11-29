#!/usr/bin/env python3
import numpy as np
import os
import pandas as pd
import re

_output_fold = '../../zz_old/coliva/outputs/all'
_shard = 'val'

def get_model_mapes(output_fold, shard='val'):
    # Initialize dictionary that maps keywords to relevant parsing quantities
    NAME = 'name'
    IDX = 'idx'
    parsing_dict = {'tr':  {NAME: 'Training',     IDX: 2 },
                    'hp':  {NAME: 'HP Selection', IDX: 3 },
                    'val': {NAME: 'Validation',   IDX: 4 } }
    # Find matching shard
    found = False
    for key in parsing_dict:
        if shard.lower().startswith(key):
            shard_dict = parsing_dict[key]
            found = True
            break
    if not found:
        raise ValueError(f"{shard} does not begin with one of "
                         f"{list(parsing_dict.keys())}")
    print(f"Extracting {shard_dict[NAME]} MAPEs...")

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
            print(experiment_name)
            integer = int(re.findall(r'\d+', experiment_name)[-1])
            results_file_path = os.path.join(device_output_fold,
                                             experiment_name, 'results.txt')
            # Read results from file
            if not os.path.exists(results_file_path):
                continue
            with open(results_file_path, 'r') as f:
                # Loop over file lines
                for lin in f.readlines():
                    result_str = 'Best result for Technique.'
                    if result_str in lin:
                        # Line with MAPEs was found: now parse it
                        mape_idx = shard_dict[IDX]
                        mape_str = f'{shard_dict[NAME]} MAPE is '
                        line_list = lin.replace(result_str, '').split(' - ')
                        technique = line_list[0].strip()
                        mape = line_list[mape_idx]
                        strings_to_remove = (mape_str, ' ', '\t', '(', ')')
                        for ch in strings_to_remove:
                            mape = mape.replace(ch, '')
                        mape = float(mape)
                        #print(integer, technique, mape, sep=" -- ")
                        df.loc[integer, technique] = mape
        df.sort_index(inplace=True)
        dfs[device] = df

    return dfs


if __name__ == '__main__':
    get_model_mapes(_output_fold, _shard)
