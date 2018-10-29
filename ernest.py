"""
Copyright 2018 Elif Sahin
Copyright 2018 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import argparse
import configparser as cp
import pandas as pd
import numpy as np

class Ernest(object):
    def __init__(self):
        self.conf = cp.ConfigParser()
        self.conf.optionxform = str
        self.parameters = {}
    
    def compute_ernest_df(self, input_path):
        self.input_name = os.path.splitext(os.path.basename(input_path))[0]
        self.df = pd.read_csv(input_path)
        self.ernest_df = self.df.copy()
        self.ernest_df['C1'] = self.ernest_df['dataSize'] * (1 / self.ernest_df['nContainers'])
        self.ernest_df['C2'] = np.log(self.ernest_df['nContainers'])
        self.ernest_df['C3'] = np.sqrt(self.ernest_df['dataSize']) * (1 / self.ernest_df['nContainers'])
        # use this column only for nnls (positive = True in config file)
        self.ernest_df['C4'] = np.power(self.ernest_df['dataSize'], 2) * (1 / self.ernest_df['nContainers'])
        self.ernest_df = self.ernest_df[['run', 'applicationCompletionTime', 'users', 'nContainers', 'C1', 'C2', 'C3', 'C4']]
        print(self.ernest_df)
        self.ernest_df.to_csv(os.path.join('inputs', self.input_name + '_ernest.csv'), index = False)

if __name__ == '__main__':
    ernest = Ernest()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', metavar = 'input_file', type = str, help = 'input file path', required = True)

    args = parser.parse_args()
    ernest.compute_ernest_df(args.input)

