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

import pandas as pd
import numpy as np
import ast
import os
import logging
import configparser as cp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreparation(object):
  def __init__(self):
    self.conf = cp.ConfigParser()
    self.conf.optionxform = str
    
    self.scaler = StandardScaler()
    self.logger = logging.getLogger(__name__)

  def read_inputs(self, config_path, input_path, analytical_path):
    self.conf.read(config_path)

    debug = ast.literal_eval(self.conf.get('DebugLevel', 'debug'))
    logging.basicConfig(level = logging.DEBUG) if debug else logging.basicConfig(level = logging.INFO)

    self.test_size = float(self.conf.get('DataPreparation', 'test_size'))
    self.cv_size = float(self.conf.get('DataPreparation', 'cv_size'))
    target_column = int(self.conf.get('DataPreparation', 'target_column'))
    self.hybrid_ml = self.conf.get('DataPreparation', 'hybrid_ml')

    if self.conf.get('DataPreparation', 'test_data_size') == "":
      self.test_data_size = False
    else:
      self.test_data_size = ast.literal_eval(self.conf.get('DataPreparation', 'test_data_size'))
     
    if self.conf.get('DataPreparation', 'train_data_size') == "":
      self.train_data_size = False
    else:
      self.train_data_size = ast.literal_eval(self.conf.get('DataPreparation', 'train_data_size'))
    
    self.df = pd.read_csv(input_path)
     
    if target_column != 1:
      column_names = list(self.df)
      column_names[1], column_names[target_column] = column_names[target_column], column_names[1]
      self.df = self.df.reindex(columns = column_names)
    
    self.target_column_name = self.df.columns[1]
    
    self.test_rows_size = self.df.index.values.astype(int)
    self.train_rows_size = self.df.index.values.astype(int)
    self.test_rows_core = self.df.index.values.astype(int)
    self.train_rows_core = self.df.index.values.astype(int)

    if 'run' in self.df.columns:
      self.df = self.df.drop(['run'], axis = 1)

  def split_data(self, seed):
    if 'weight' in self.df.columns:
      sample_weight = self.df['weight']
    
    self.df = self.df.loc[:, (self.df != self.df.iloc[0]).any()]
    
    if 'weight' not in self.df.columns and self.hybrid_ml == 'on':
      self.df['weight'] = sample_weight
    
    self.df = shuffle(self.df, random_state = seed)
    self.df = self.scale_data(self.df)

    if self.hybrid_ml == 'on':
      self.df['weight'] = sample_weight 
      self.scaled_analytical_df = self.df[self.df.index.isin(self.analytical_rows)]
      self.df = self.df.drop(self.analytical_rows)

    train_filter = list(set(self.train_rows_size) & set(self.train_rows_core))
    test_filter = list(set(self.test_rows_size) & set(self.test_rows_core))
    
    self.train_df = self.df[self.df.index.isin(train_filter)]
    self.test_df = self.df[self.df.index.isin(test_filter)]
    
    if self.train_df.shape == self.df.shape:
      self.logger.info('Test set not given - split the input file to train and test sets')
      self.train_df, self.test_df = train_test_split(self.df, test_size = self.test_size, random_state = seed)

    self.train_labels = self.train_df.iloc[:,0]
    self.train_features = self.train_df.iloc[:,1:]
     
    self.test_labels = self.test_df.iloc[:,0]
    self.test_features = self.test_df.iloc[:,1:]
      
    self.train_features, self.cv_features, self.train_labels, self.cv_labels = train_test_split(self.train_features, \
                                                                                                self.train_labels, \
                                                                                                test_size = self.cv_size, \
                                                                                                random_state = seed)
     
    if self.hybrid_ml == 'on':
      other_columns = self.scaled_analytical_df.loc[:, self.scaled_analytical_df.columns != self.target_column_name]
      self.train_features = pd.concat([self.train_features, other_columns])
      self.train_labels = pd.concat([self.train_labels, self.scaled_analytical_df[self.target_column_name]])
    
    self.train_df = pd.concat([self.train_labels, self.train_features], axis = 1)
    self.cv_df = pd.concat([self.cv_labels, self.cv_features], axis = 1)
    self.test_df = pd.concat([self.test_labels, self.test_features], axis = 1)
    
    return self.train_features, self.test_features, self.cv_features, self.train_labels, self.test_labels, self.cv_labels

  def scale_data(self, df):
    scaled_array = self.scaler.fit_transform(df.values)
    return pd.DataFrame(scaled_array, index = df.index, columns = df.columns)

  def get_nn_weights(self):
    self.train_df_original = self.train_df.copy()
    self.train_df = pd.DataFrame(np.repeat(self.train_df.values, self.train_df['weight'].tolist(), axis=0), columns = self.train_df.columns)
    return self.train_df.iloc[:,0], self.train_df.iloc[:,1:]

  def reset_train_set(self):
    self.train_df = self.train_df_original.copy()
    #print("CHANGE THE TRAIN DF - SHOULD BE ORIGINAL SHAPE")
    #print(self.train_df.shape)
    return self.train_df.iloc[:,0], self.train_df.iloc[:,1:]

  def mean_absolute_percentage_error(self, set_type, predicted_labels, algorithm, scenario, input_name):
    true_labels, predicted_labels = self.scale_back_arrays(set_type, predicted_labels) 
    difference = true_labels - predicted_labels
    
    mape_df = pd.DataFrame(np.abs(np.divide(difference, true_labels)), columns = self.df.columns)[[self.target_column_name]]
    mape = (np.mean(mape_df) * 100)[self.target_column_name]
    fraction_err = pd.DataFrame(np.abs(np.divide(difference, true_labels)) * 100, columns = self.df.columns)[[self.target_column_name]]
    fraction_err_gt_30 = len(fraction_err[fraction_err[self.target_column_name] >= 30]) / fraction_err.shape[0]
    
    if set_type == 'train':
      np.save(open(os.path.join('outputs', scenario + '_' + input_name + '_' + algorithm + '_train_predicted.npy'), 'wb'), predicted_labels)
      np.save(open(os.path.join('outputs', scenario + '_' + input_name + '_' + algorithm + '_train_true.npy'), 'wb'), true_labels) 
    elif set_type == 'cv':
      np.save(open(os.path.join('outputs', scenario + '_' + input_name + '_' + algorithm + '_cv_predicted.npy'), 'wb'), predicted_labels)
      np.save(open(os.path.join('outputs', scenario + '_' + input_name + '_' + algorithm + '_cv_true.npy'), 'wb'), true_labels)
    else:
      np.save(open(os.path.join('outputs', scenario + '_' + input_name + '_' + algorithm + '_test_predicted.npy'), 'wb'), predicted_labels)
      np.save(open(os.path.join('outputs', scenario + '_' + input_name + '_' + algorithm + '_test_true.npy'), 'wb'), true_labels)
      
    return mape, np.float64(fraction_err_gt_30)
  
  def scale_back_arrays(self, set_type, predicted_labels):
    if set_type == 'train':
      true_labels = self.scaler.inverse_transform(self.train_df)
      temp_train_df = self.train_df.copy()
      temp_train_df.iloc[:,0] = predicted_labels
      predicted_labels = self.scaler.inverse_transform(temp_train_df)

    elif set_type == 'cv':
      true_labels = self.scaler.inverse_transform(self.cv_df)
      temp_cv_df = self.cv_df.copy()
      temp_cv_df.iloc[:,0] = predicted_labels
      predicted_labels = self.scaler.inverse_transform(temp_cv_df)

    elif set_type == 'test':
      true_labels = self.scaler.inverse_transform(self.test_df)
      temp_test_df = self.test_df.copy()
      temp_test_df.iloc[:,0] = predicted_labels
      predicted_labels = self.scaler.inverse_transform(temp_test_df)

    else:
      true_labels = self.scaler.inverse_transform(self.df)
      predicted_labels = None

    return true_labels, predicted_labels
  
  def plot(self, algorithm, best_parameter, train_predicted, cv_predicted, test_predicted):
    return 
