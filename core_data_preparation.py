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

import ast
from data_preparation import DataPreparation
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pylab as plt
import matplotlib
import pickle

class CoreDataPreparation(DataPreparation):
  def __init__(self):
    super(CoreDataPreparation, self).__init__()
    plt.rcParams.update({'font.size': 28})
    matplotlib.pyplot.switch_backend('agg') 
    plt.rc('grid', linestyle="-", color='black')
    self.out_dir = 'plots'
    if not os.path.exists(self.out_dir):
      os.makedirs(self.out_dir)

  def read_inputs(self, config_path, input_path, analytical_path):
    super(CoreDataPreparation, self).read_inputs(config_path, input_path, analytical_path)
    self.input_name = os.path.splitext(os.path.basename(input_path))[0]
    self.scenario = self.conf.get('DataPreparation', 'scenario')
    self.evaluate = self.conf.get('DataPreparation', 'evaluation_without_apriori')

    self.df['inverse_nContainers'] = 1 / self.df['nContainers']

    if self.test_data_size and self.train_data_size:
      self.logger.debug("Train (%s) and Test (%s) Data Size are given.", self.train_data_size, self.test_data_size)
      self.test_rows_size = self.df[self.df['dataSize'].isin(self.test_data_size)].index.values.astype(int)
      self.train_rows_size = self.df[self.df['dataSize'].isin(self.train_data_size)].index.values.astype(int)
      
    elif self.test_data_size:
      self.logger.debug("No train data size is given")
      self.test_rows_size = self.df[self.df['dataSize'] == self.test_data_size].index.values.astype(int)
      self.train_rows_size = self.df[self.df['dataSize'] != self.test_data_size].index.values.astype(int)
    
    elif self.train_data_size:
      self.logger.debug("No test data size is given")
      self.train_rows_size = self.df[self.df['dataSize'] == self.train_data_size].index.values.astype(int)
      self.test_rows_size = self.df[~self.df['dataSize'].isin(self.train_data_size)].index.values.astype(int)

    else:
      self.test_rows_size = self.df.index.values.astype(int)
      self.train_rows_size = self.df.index.values.astype(int)
    
    if self.hybrid_ml == 'on':
      self.df['weight'] = int(self.conf.get('DataPreparation', 'op_analytical_ratio'))
      self.analytical_df = pd.read_csv(analytical_path)
      
      print("all analytical data")
      print(self.analytical_df)
      if self.train_data_size == self.test_data_size:
        self.analytical_df = self.analytical_df[self.analytical_df['dataSize'].isin(self.train_data_size)]
      print("new analytical data")
      print(self.analytical_df)

      self.analytical_rows = [i for i in range(self.df.shape[0], self.df.shape[0] + self.analytical_df.shape[0])]
      self.df = pd.concat([self.df, self.analytical_df], ignore_index = True)

    self.test_rows_core = self.check_cores('test')
    self.train_rows_core = self.check_cores('train')
    
    if self.evaluate == 'on' and self.test_data_size == self.train_data_size:
      self.change_test_set()

    self.plot_original_data()
 
  def check_cores(self, set_type):
    if ast.literal_eval(self.conf.get('DataPreparation', 'split_test_set')):
      if set_type == 'test':
        if self.conf.get('DataPreparation', 'test_cores') != "":
          test_cores = ast.literal_eval(self.conf.get('DataPreparation', 'test_cores'))
          return self.df[self.df['nContainers'].isin(test_cores)].index.values.astype(int)
      else:
        if self.conf.get('DataPreparation', 'train_cores') != "":
          train_cores = ast.literal_eval(self.conf.get('DataPreparation', 'train_cores'))
          return self.df[self.df['nContainers'].isin(train_cores)].index.values.astype(int)
    return self.df.index.values.astype(int)


  def change_test_set(self):
    test_filter = list(set(self.test_rows_size) & set(self.test_rows_core))
    train_filter = list(set(self.train_rows_size) & set(self.train_rows_core))    
    
    n_task = self.df[self.df.index.isin(train_filter)].filter(regex='nTask')
    avg_task = self.df[self.df.index.isin(train_filter)].filter(regex='avgTask')
    max_task = self.df[self.df.index.isin(train_filter)].filter(regex='maxTask')
    sh_max = self.df[self.df.index.isin(train_filter)].filter(regex='SHmax')
    sh_avg = self.df[self.df.index.isin(train_filter)].filter(regex='SHavg')
    b_max = self.df[self.df.index.isin(train_filter)].filter(regex='Bmax')
    b_avg = self.df[self.df.index.isin(train_filter)].filter(regex='Bavg')
    
    avg_n_task = n_task.mean(axis = 0)
    avg_avg_task = avg_task.mean(axis = 0)
    avg_max_task = max_task.mean(axis = 0)
    avg_sh_max = sh_max.mean(axis = 0)
    avg_sh_avg = sh_avg.mean(axis = 0)
    avg_b_max = b_max.mean(axis = 0)
    avg_b_avg = b_avg.mean(axis = 0)
    
    print(self.df)    
    self.df.loc[self.df[self.df.index.isin(test_filter)].filter(regex='nTask').index, n_task.columns] = avg_n_task.values
    self.df.loc[self.df[self.df.index.isin(test_filter)].filter(regex='avgTask').index, avg_task.columns] = avg_avg_task.values
    self.df.loc[self.df[self.df.index.isin(test_filter)].filter(regex='maxTask').index, max_task.columns] = avg_max_task.values
    self.df.loc[self.df[self.df.index.isin(test_filter)].filter(regex='SHmax').index, sh_max.columns] = avg_sh_max.values
    self.df.loc[self.df[self.df.index.isin(test_filter)].filter(regex='SHavg').index, sh_avg.columns] = avg_sh_avg.values
    self.df.loc[self.df[self.df.index.isin(test_filter)].filter(regex='Bmax').index, b_max.columns] = avg_b_max.values
    self.df.loc[self.df[self.df.index.isin(test_filter)].filter(regex='Bavg').index, b_avg.columns] = avg_b_avg.values
    
    if 'runbest' in self.input_name:
      print("also do gaps")
      gap_column_names = [i for i in self.df.columns.values if "gap" in i]
      gap_column_names.append("nCoresTensorflow")
      train_gap_values = self.df.ix[train_filter]
      unique_train_cores = train_gap_values["nCoresTensorflow"].unique()
      train_gap_values = train_gap_values[gap_column_names]
      gap_value_1_avg = train_gap_values.groupby("nCoresTensorflow", as_index=False)["gap_value_1"].mean()
      gap_value_2_avg = train_gap_values.groupby("nCoresTensorflow", as_index=False)["gap_value_2"].mean()
      gap_value_3_avg = train_gap_values.groupby("nCoresTensorflow", as_index=False)["gap_value_3"].mean()
      for test_index in test_filter:
        core = self.df["nCoresTensorflow"][test_index]
        closest_core = min(unique_train_cores, key=lambda x:abs(x-core))
        #self.df["gap_value_1"][test_index] = float(gap_value_1_avg.loc[gap_value_1_avg['nCoresTensorflow'] == closest_core]["gap_value_1"])
        #self.df["gap_value_2"][test_index] = float(gap_value_2_avg.loc[gap_value_2_avg['nCoresTensorflow'] == closest_core]["gap_value_2"])
        #self.df["gap_value_3"][test_index] = float(gap_value_3_avg.loc[gap_value_3_avg['nCoresTensorflow'] == closest_core]["gap_value_3"])
        self.df["gap_value_1"][test_index] = closest_core * float(gap_value_1_avg.loc[gap_value_1_avg["nCoresTensorflow"] == closest_core]["gap_value_1"]) / core
        self.df["gap_value_2"][test_index] = closest_core * float(gap_value_2_avg.loc[gap_value_2_avg["nCoresTensorflow"] == closest_core]["gap_value_2"]) / core
        self.df["gap_value_3"][test_index] = closest_core * float(gap_value_3_avg.loc[gap_value_3_avg["nCoresTensorflow"] == closest_core]["gap_value_3"]) / core

  def get_original_sets(self, train_predicted, cv_predicted, test_predicted):
    org_train_array, org_train_predicted = self.scale_back_arrays('train', train_predicted.reshape(-1, 1))
    org_cv_array, org_cv_predicted = self.scale_back_arrays('cv', cv_predicted.reshape(-1, 1))
    org_test_array, org_test_predicted = self.scale_back_arrays('test', test_predicted.reshape(-1, 1))
    whole_array, _ = self.scale_back_arrays('whole', None)

    self.org_train_df = pd.DataFrame(org_train_array, index = self.train_df.index, columns = self.train_df.columns)
    self.org_train_predicted = pd.DataFrame(org_train_predicted, index = self.train_df.index, columns = self.train_df.columns)

    self.org_cv_df = pd.DataFrame(org_cv_array, index = self.cv_df.index, columns = self.cv_df.columns)
    self.org_cv_predicted = pd.DataFrame(org_cv_predicted, index = self.cv_df.index, columns = self.cv_df.columns)

    self.org_test_df = pd.DataFrame(org_test_array, index = self.test_df.index, columns = self.test_df.columns)
    self.org_test_predicted = pd.DataFrame(org_test_predicted, index = self.test_df.index, columns = self.test_df.columns)

    self.org_df = pd.DataFrame(whole_array, index = self.df.index, columns = self.df.columns)

  def plot(self, algorithm, best_parameter, train_predicted, cv_predicted, test_predicted):
    self.get_original_sets(train_predicted, cv_predicted, test_predicted)
    self.plot_cores_runtime(algorithm, best_parameter)
    self.plot_predicted_true(algorithm, best_parameter)
    self.plot_predicted_residual(algorithm, best_parameter)
    #self.plot_sample_weights()
    self.plot_analytical(algorithm)

  def plot_cores_runtime(self, algorithm, best_parameter):
    plt.figure(figsize=(15, 12))
    ax = plt.axes()
    major_ticks = np.arange(0, 50, 10)
    minor_ticks = np.arange(0, 50, 2)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(b = True, which = 'both', color = 'black', linestyle=':')
    
    plt.xlabel("Number of cores", labelpad = 30)
    plt.ylabel(self.target_column_name + " (s)", labelpad = 30)
    
    plt.scatter(self.org_train_df['nContainers'], self.org_train_predicted[self.target_column_name] / 1000, s=200, edgecolor="darkorange", facecolor="none", label="Train Predicted Values", alpha=1, marker="^")
    plt.scatter(self.org_train_df['nContainers'], self.org_train_df[self.target_column_name] / 1000, s=200, edgecolor="black", facecolor="none", label="Train True Values", alpha=1, marker="^")
    
    plt.scatter(self.org_cv_df['nContainers'], self.org_cv_predicted[self.target_column_name] / 1000, s=200, edgecolor="red", facecolor="none", label="Cv Predicted Values", alpha=1, marker="s")
    plt.scatter(self.org_cv_df['nContainers'], self.org_cv_df[self.target_column_name] / 1000, s=200, edgecolor="black", facecolor="none", label="Cv True Values", alpha=1, marker="s")
    
    plt.scatter(self.org_test_df['nContainers'], self.org_test_predicted[self.target_column_name] / 1000, s=200, edgecolor="green", facecolor="none", label="Test Predicted Values", alpha=1, marker="o")  
    plt.scatter(self.org_test_df['nContainers'], self.org_test_df[self.target_column_name] / 1000, s=200, edgecolor="black", facecolor="none", label="Test True Values", alpha=1, marker="o")
    
    plot = plt.legend()
    pickle.dump(plot, open(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_cores_runtime.pickle'), 'wb'))
    #plt.title("Best Model for " + algorithm + " with all sets \n" + self.input_name + " " + self.scenario)
    #plt.annotate("Parameters : " + str(best_parameter), (0,0), (0, -150), xycoords='axes fraction', textcoords='offset points', va='bottom')
    
    plt.savefig(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_cores_runtime.pdf'))
    plt.close()

  def plot_predicted_true(self, algorithm, best_parameter):
    plt.figure(figsize=(15, 12))
    ax = plt.axes()
    ax.grid(b = True, which = 'major', color = 'black', linestyle=':')
    
    plt.xlabel("Predicted Valus of " + self.target_column_name + " (s)", labelpad = 30)
    plt.ylabel("True Values of " + self.target_column_name + " (s)", labelpad = 30)

    plt.scatter(self.org_train_predicted[self.target_column_name] / 1000, self.org_train_df[self.target_column_name] / 1000, s=200, edgecolor="darkorange", facecolor="none", label="Train Set", alpha=1, marker="^")
    plt.scatter(self.org_cv_predicted[self.target_column_name] / 1000, self.org_cv_df[self.target_column_name] / 1000, s=200, edgecolor="red", facecolor="none", label="Cv Set", alpha=1, marker="s")
    plt.scatter(self.org_test_predicted[self.target_column_name] / 1000, self.org_test_df[self.target_column_name] / 1000, s=200, edgecolor="green", facecolor="none", label="Test Set", alpha=1, marker="o")
    
    lims = [ np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()]) ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0, color='blue')
    
    plot = plt.legend()
    pickle.dump(plot, open(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_predicted_true.pickle'), 'wb'))
    #plt.title("Predicted vs True Values for " + algorithm + " with all sets \n" + self.input_name + " " + self.scenario)
    #plt.annotate("Parameters : " + str(best_parameter), (0,0), (0, -150), xycoords='axes fraction', textcoords='offset points', va='bottom')

    plt.savefig(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_predicted_true.pdf'))
    plt.close()

  def plot_predicted_residual(self, algorithm, best_parameter): 
    plt.figure(figsize=(15, 12))
    ax = plt.axes()
    ax.grid(b = True, which = 'major', color = 'black', linestyle=':')
    
    plt.xlabel("Predicted Valus of " + self.target_column_name + " (s)", labelpad = 30)
    plt.ylabel("Residual Values of " + self.target_column_name + " (s)", labelpad = 30)
    
    plt.scatter(self.org_train_predicted[self.target_column_name] / 1000, (self.org_train_df[self.target_column_name] - self.org_train_predicted[self.target_column_name]) / 1000, s=200, edgecolor="darkorange", facecolor="none", label="Train Set", alpha=1, marker="^")
    plt.scatter(self.org_cv_predicted[self.target_column_name] / 1000, (self.org_cv_df[self.target_column_name] - self.org_cv_predicted[self.target_column_name]) / 1000, s=200, edgecolor="red", facecolor="none", label="Cv Set", alpha=1, marker="s")
    plt.scatter(self.org_test_predicted[self.target_column_name] / 1000, (self.org_test_df[self.target_column_name] - self.org_test_predicted[self.target_column_name]) / 1000, s=200, edgecolor="green", facecolor="none", label="Test Set", alpha=1, marker="o")
    
    plt.axhline(y=0, color='black', linestyle='-')
    
    plot = plt.legend()
    pickle.dump(plot, open(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_predicted_residual.pickle'), 'wb'))
    #plt.title("Predicted vs Residual Values for " + algorithm + " with all sets \n" + self.input_name + " " + self.scenario)
    #plt.annotate("Parameters : " + str(best_parameter), (0,0), (0, -150), xycoords='axes fraction', textcoords='offset points', va='bottom')

    plt.savefig(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_predicted_residual.pdf'))
    plt.close()

  def plot_original_data(self):
    plt.figure(figsize=(16, 13))
    ax = plt.axes()
    ax.grid(b = True, which = 'both', color = 'black', linestyle=':')
    major_ticks = np.arange(0, 50, 10)
    minor_ticks = np.arange(0, 50, 2)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    plt.xlabel("Number of cores", labelpad = 30)
    plt.ylabel("Original " + self.target_column_name + " (s)", labelpad = 30)
    
    if self.input_name == 'P8_kmeans':
      plt.scatter(self.df[self.df['dataSize'] == 5]['nContainers'], self.df[self.df['dataSize'] == 5][self.target_column_name] / 1000, s=200, edgecolor="darkorange", facecolor="none", label="dataSize = 5", alpha=1, marker="^")
      plt.scatter(self.df[self.df['dataSize'] == 10]['nContainers'], self.df[self.df['dataSize'] == 10][self.target_column_name] / 1000, s=200, edgecolor="green", facecolor="none", label="dataSize = 10", alpha=1, marker="^")
      plt.scatter(self.df[self.df['dataSize'] == 15]['nContainers'], self.df[self.df['dataSize'] == 15][self.target_column_name] / 1000, s=200, edgecolor="red", facecolor="none", label="dataSize = 15", alpha=1, marker="^")
      plt.scatter(self.df[self.df['dataSize'] == 20]['nContainers'], self.df[self.df['dataSize'] == 20][self.target_column_name] / 1000, s=200, edgecolor="blue", facecolor="none", label="dataSize = 20", alpha=1, marker="^")

    elif self.input_name == 'query26' or self.input_name == 'query40':
      plt.scatter(self.df[self.df['dataSize'] == 250]['nContainers'], self.df[self.df['dataSize'] == 250][self.target_column_name] / 1000, s=200, edgecolor="darkorange", facecolor="none", label="dataSize = 250", alpha=1, marker="^")
      plt.scatter(self.df[self.df['dataSize'] == 750]['nContainers'], self.df[self.df['dataSize'] == 750][self.target_column_name] / 1000, s=200, edgecolor="green", facecolor="none", label="dataSize = 750", alpha=1, marker="^")
      plt.scatter(self.df[self.df['dataSize'] == 1000]['nContainers'], self.df[self.df['dataSize'] == 1000][self.target_column_name] / 1000, s=200, edgecolor="red", facecolor="none", label="dataSize = 1000", alpha=1, marker="^")

    else:
      plt.scatter(self.df['nContainers'], self.df[self.target_column_name] / 1000, s=200, edgecolor="darkorange", facecolor="none", label="all dataSizes", alpha=1, marker="^")

    plot = plt.legend()
    pickle.dump(plot, open(os.path.join(self.out_dir, 'original_data_' + self.input_name + '.pickle'), 'wb'))
    #plt.title("Original Data of " + self.input_name)
    plt.savefig(os.path.join(self.out_dir, 'original_data_' + self.input_name + '.pdf'))
    plt.close()
  
  def plot_analytical(self, algorithm):
    if self.hybrid_ml == 'on':
      plt.figure(figsize=(15, 12))
      ax = plt.axes()
      major_ticks = np.arange(0, 50, 10)
      minor_ticks = np.arange(0, 50, 2)
      ax.set_xticks(major_ticks)
      ax.set_xticks(minor_ticks, minor=True)
      ax.grid(b = True, which = 'both', color = 'black', linestyle=':')
    
      plt.xlabel("Number of cores", labelpad = 30)
      plt.ylabel(self.target_column_name + " (s)", labelpad = 30)
    
      cores_predicted = self.org_test_predicted[self.org_test_predicted['dataSize'].isin(self.test_data_size)]['nContainers']
      time_predicted = self.org_test_predicted[self.org_test_predicted['dataSize'].isin(self.test_data_size)][self.target_column_name]

      cores_true = self.org_test_df[self.org_test_df['dataSize'].isin(self.test_data_size)]['nContainers']
      time_true = self.org_test_df[self.org_test_df['dataSize'].isin(self.test_data_size)][self.target_column_name]
    
      cores_analytical = self.analytical_df[self.analytical_df['dataSize'].isin(self.test_data_size)]['nContainers']
      time_analytical = self.analytical_df[self.analytical_df['dataSize'].isin(self.test_data_size)][self.target_column_name]
    
      plt.scatter(cores_predicted, time_predicted / 1000, s=300, edgecolor="green", facecolor="none", label="Test Predicted Values", alpha=1, marker="o")
      plt.scatter(cores_true, time_true / 1000, s=300, edgecolor="black", facecolor="none", label="Test True Values", alpha=1, marker="o")
      plt.scatter(cores_analytical, time_analytical / 1000, s=300, edgecolor="red", facecolor="none", label="Analytical Values", alpha=1, marker="o")
      
      plot = plt.legend()
      pickle.dump(plot, open(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_analytical_cores_runtime.pickle'), 'wb'))
      plt.savefig(os.path.join(self.out_dir, self.input_name + '_' + self.scenario + '_' + algorithm + '_analytical_cores_runtime.pdf'))
      plt.close()
    
    
