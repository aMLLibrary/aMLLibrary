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
import pandas as pd
import numpy as np
import configparser as cp

class AnalyticalModelTool(object):
    def __init__(self):
        self.conf = cp.ConfigParser()
        self.conf.optionxform = str
    
    def compute_analytical_csv(self, config_path, input_path, output_type):
        self.input_name = os.path.splitext(os.path.basename(input_path))[0]
        self.conf.read(config_path)
        self.df = pd.read_csv(input_path)

        if self.input_name == 'Azure_rf':
            x = self.df[self.df['nContainers'] == 20]
            num = x.filter(regex='nTask')
            num.rename(columns=lambda x: x[6:], inplace=True)
            avg = x.filter(regex='avgTask')
            avg.rename(columns=lambda x: x[8:], inplace=True)
            final = num.multiply(avg)
            final['sum'] = final[list(final.columns)].sum(axis=1)
            final['nContainers'] = self.df['nContainers']
            y = pd.DataFrame()
            for i in range(6, 50, 2):
                final[['result']] = final[['sum']].div(i, axis = 0)
                y = y.append(final[['result']], ignore_index = True)

            cores = list(range(6, 50, 2)) * 5
            cores.sort()
            y['nContainers'] = pd.DataFrame({'nContainers': cores})
            maxTask = x.filter(regex='maxTask')
            sHmax = x.filter(regex='SHmax')
            sHavg = x.filter(regex='SHavg')
            bmax = x.filter(regex='Bmax')
            bavg = x.filter(regex='Bavg')
            nTask = x.filter(regex='nTask')
            avgTask = x.filter(regex='avgTask')
            maxTask = maxTask.append([maxTask] * int((self.df.shape[0] / 5) - 1), ignore_index=True)
            sHmax = sHmax.append([sHmax] * int((self.df.shape[0] / 5) - 1), ignore_index=True)
            sHavg = sHavg.append([sHavg] * int((self.df.shape[0] / 5) - 1), ignore_index=True)
            bmax = bmax.append([bmax] * int((self.df.shape[0] / 5) - 1), ignore_index=True)
            bavg = bavg.append([bavg] * int((self.df.shape[0] / 5) - 1), ignore_index=True)
            nTask = nTask.append([nTask] * int((self.df.shape[0] / 5) - 1), ignore_index=True)
            avgTask = avgTask.append([avgTask] * int((self.df.shape[0] / 5) - 1), ignore_index=True)
            result = pd.concat([y, self.df['run'], maxTask, sHmax, sHavg, bmax, bavg, avgTask, nTask, self.df['users'], self.df['dataSize']], axis = 1)
            result.rename(columns={'result': 'applicationCompletionTime'}, inplace = True)
            result = result[self.df.columns]
            self.analytical_df = result.copy()

        else:
            self.analytical_df = self.df.copy()
            num = self.df.filter(regex='nTask')
            num.rename(columns=lambda x: x[6:], inplace=True)
            avg = self.df.filter(regex='avgTask')
            avg.rename(columns=lambda x: x[8:], inplace=True)
            final = num.multiply(avg)
            final['sum'] = final[list(final.columns)].sum(axis=1)
            final[['result']] = final[['sum']].div(self.df['nContainers'].values,axis=0)
            self.analytical_df['applicationCompletionTime'] = final[['result']]
        
        group = self.conf.get('DataPreparation', 'analytical_model')

        if group == 'groupby':
            self.analytical_df = self.analytical_df.groupby(['nContainers', 'dataSize']).mean().reset_index()
            cols = list(self.analytical_df)
            cols = cols[1:] + cols[:1]
            cols = cols[1:] + cols[:1]
            cols[len(cols) - 1], cols[len(cols) - 2] = cols[len(cols) - 2], cols[len(cols) - 1]
            self.analytical_df = self.analytical_df[cols]

        self.analytical_df['inverse_nContainers'] = 1 / self.analytical_df['nContainers']
        self.analytical_df['weight'] = 1

        if output_type == 'normal':
            self.analytical_df.to_csv(os.path.join('inputs', self.input_name + '_analytical.csv'), index = False)
        else:
            self.analytical_df['applicationCompletionTime'] *= 1.28
            self.analytical_df.to_csv(os.path.join('inputs', self.input_name + '_analytical_noisy.csv'), index = False)

        print(self.analytical_df)
        
    def calculate_error(self, output_type):
        data_sizes = []
        if 'query' in self.input_name:
            data_sizes = [250, 750, 1000]
            cores = list(range(6, 46, 2))
        else:
            cores = list(range(6, 50, 2))
            if 'kmeans' in self.input_name:
                data_sizes = [5, 10, 15, 20]
            elif 'logistic_30' in self.input_name:
                data_sizes = [30]
            elif 'logistic_45' in self.input_name:
                data_sizes = [45]
            elif 'logistic_50' in self.input_name:
                data_sizes = [50]
            elif 'logistic_55' in self.input_name:
                data_sizes = [55]
            elif 'Azure_rf' in self.input_name:
                data_sizes = [60]
        
        result = pd.DataFrame()
        mult = 1 if output_type == 'normal' else 1.28
        for i in cores:
            for j in data_sizes:
                x = self.df[self.df['dataSize'] == j][self.df['nContainers'] == i]['applicationCompletionTime']
                if not x.empty:
                    y = self.analytical_df[self.analytical_df['dataSize'] == j][self.analytical_df['nContainers'] == i]['applicationCompletionTime'].values[0] * mult 
                    data = pd.DataFrame({"err": ((x - y) / x).abs()})
                    result = result.append(data)

        print("Errors")
        #print(result.sort_index())
        avg_error = result.mean() * 100
        print(avg_error['err'])


if __name__ == '__main__':
    amt = AnalyticalModelTool()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', metavar = 'config_file', type = str, help = 'config file path', required = True)
    parser.add_argument('--input', '-i', metavar = 'input_file', type = str, help = 'input file path', required = True)
    parser.add_argument('--type', '-t', metavar = 'type', type = str, help = 'type - normal or noisy', required = True)

    args = parser.parse_args()
    amt.compute_analytical_csv(args.config, args.input, args.type)
    amt.calculate_error(args.type)

