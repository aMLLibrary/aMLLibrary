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

import logging
import ast
import csv
import os
import time
import datetime
import numpy as np
import configparser as cp
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from data_preparation import DataPreparation
from core_data_preparation import CoreDataPreparation
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from neural_network import Net
from multiprocessing import Manager, Process
from multiprocessing.pool import ThreadPool
import threading
import pickle
import torch

import warnings
warnings.filterwarnings(action = "ignore", module = "scipy", message = "^internal gelsd")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action = 'ignore', category = DataConversionWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)

class ML(object):
  def __init__(self, seed, config_path, input_path, test_path):
    self.all_hyper_parameters = {}
    self.seed = seed
    self.input_name = os.path.splitext(os.path.basename(input_path))[0]
    self.read_config_file(config_path)
		
    self.data_preparation.read_inputs(config_path, input_path, test_path)
    self.train_features, self.test_features, self.cv_features, self.train_labels, self.test_labels, self.cv_labels = self.data_preparation.split_data(seed)

    self.logger = logging.getLogger(__name__)

  def read_config_file(self, path):
    conf = cp.ConfigParser()
    conf.optionxform = str
    conf.read(path)
    self.debug = ast.literal_eval(conf.get('DebugLevel', 'debug'))
    
    self.parameters = {}
    self.algorithms = ast.literal_eval(conf.get('ML', 'algorithms'))
    self.scenario = conf.get('DataPreparation', 'scenario')
    self.hybrid_ml = conf.get('DataPreparation', 'hybrid_ml')
    self.num_cores = int(conf.get('DataPreparation', 'num_cores'))
    self.nn_loss_plots = conf.get('DataPreparation', 'nn_loss_plots')

    log_file = os.path.join('outputs', self.scenario + '_' + self.input_name + '_model_selection.log')
    if not os.path.exists('outputs'):
      os.makedirs('outputs')
    logging.basicConfig(filename = log_file, level = logging.DEBUG) if self.debug else logging.basicConfig(filename = log_file, level = logging.INFO)
    
    os.system("cat " + path + " > " + log_file)
    class_dp = conf.get('ML', 'data_preparation_class')

    if class_dp == 'Core_DP':
      self.data_preparation = CoreDataPreparation()
    elif class_dp == 'DP':
      self.data_preparation = DataPreparation()
    
    if conf.has_section('SVR'):
      self.parameters['SVR'] = self.config_section_map(conf, 'SVR')

    if conf.has_section('DecisionTree'):
      self.parameters['DecisionTree'] = self.config_section_map(conf, 'DecisionTree')
	
    if conf.has_section('RandomForest'):
      self.parameters['RandomForest'] = self.config_section_map(conf, 'RandomForest')
	
    if conf.has_section('LinearRegression'):
      temp = self.config_section_map(conf, 'LinearRegression')

      if temp['lasso'] == [True]:
        del temp['lasso']
        self.parameters['LinearRegressionWithLasso'] = temp
	
      else:
        del temp['lasso']
        del temp['alpha']
        self.parameters['LinearRegression'] = temp
	
    if conf.has_section('NeuralNetwork'):
      self.parameters['NeuralNetwork'] = self.config_section_map(conf, 'NeuralNetwork')
      len_activation_functions = len(self.parameters['NeuralNetwork']['activation_functions'][0])

      assert (all(i <= len_activation_functions for i in self.parameters['NeuralNetwork']['num_layers'])), \
	    "Number of layers cannot be greater than the number of activation functions"
  
  def get_grid(self): 
    result = []
    if 'poly' in self.parameters['SVR']['kernel']:
      poly_dict = dict(self.parameters['SVR'])
      poly_dict['kernel'] = ['poly']
      result.append(poly_dict)

    if 'rbf' in self.parameters['SVR']['kernel']:
      gamma_rbf_dict = dict(self.parameters['SVR'])
      gamma_rbf_dict['kernel'] = ['rbf']
      del gamma_rbf_dict['degree']
      result.append(gamma_rbf_dict)

    if 'sigmoid' in self.parameters['SVR']['kernel']:
      gamma_sigmoid_dict = dict(self.parameters['SVR'])
      gamma_sigmoid_dict['kernel'] = ['sigmoid']
      del gamma_sigmoid_dict['degree']
      result.append(gamma_sigmoid_dict)
    
    if 'linear' in self.parameters['SVR']['kernel']:
      linear_dict = dict(self.parameters['SVR'])
      linear_dict['kernel'] = ['linear']
      del linear_dict['gamma']
      del linear_dict['degree']
      result.append(linear_dict)
      
    return result

  def config_section_map(self, conf, section):
    dict1 = {}
    options = conf.options(section)
    for option in options:
      dict1[option] = ast.literal_eval(conf.get(section, option))
    return dict1

  def select_regressor(self, algorithm):
    if algorithm == 'SVR':
      return svm.SVR()
    elif algorithm == 'DecisionTree':
      return DecisionTreeRegressor()
    elif algorithm == 'RandomForest':
      return RandomForestRegressor()
    elif algorithm == 'LinearRegression':
      return LinearRegression()
    elif algorithm == 'LinearRegressionWithLasso':
      return Lasso()
    elif algorithm == 'NeuralNetwork':
      if self.hybrid_ml == 'on':
        self.train_labels, self.train_features = self.data_preparation.get_nn_weights()
      regressor = Net(self.train_features, self.train_labels, self.cv_features, self.cv_labels, self.test_features, self.test_labels, self.debug, self.scenario, self.input_name, self.nn_loss_plots)
      self.parameters[algorithm] = regressor.prepare(self.parameters[algorithm])
      return regressor

  def set_sample_weight(self, algorithm):
    if self.hybrid_ml == 'on':
      if algorithm == 'LinearRegressionWithLasso':
        self.sample_weight = self.train_features['weight'].values.all()
      elif algorithm != 'NeuralNetwork':
        self.sample_weight = np.asarray(self.train_features['weight'])
      else:
        self.sample_weight = None
    else:
      self.sample_weight = None

  def try_hyperparameter(self, algorithm, regressor, g, scores, hyperparam):
    start_time = time.time()
    self.logger.debug('Parameters to consider : %s', g)

    regressor.set_params(**g) 
    regressor.fit(self.train_features, self.train_labels, self.sample_weight) 
    
    cv_predicted = regressor.predict(self.cv_features)
    train_predicted = regressor.predict(self.train_features)

    mse = mean_squared_error(self.cv_labels, cv_predicted)
    scores.append(mse)
     
    train_mape, train_fraction = self.data_preparation.mean_absolute_percentage_error('train', train_predicted.reshape(-1, 1), algorithm, self.scenario, self.input_name)
    cv_mape, cv_fraction = self.data_preparation.mean_absolute_percentage_error('cv', cv_predicted.reshape(-1,1), algorithm, self.scenario, self.input_name)

    hyperparam.append(g)
    end_time = time.time()
    
    self.logger.debug('MSE of the algorithm : %s', mse)
    self.logger.debug('MAPE on train set : %s', train_mape)
    self.logger.debug('MAPE on cv set : %s', cv_mape)
    self.logger.debug('Time to train the parameters: %s', (end_time - start_time) / 60)
    self.logger.debug('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

  def train_model_with_HP(self):
    self.logger.info('Start training the models')
    self.start_time = time.time()

    for algorithm in self.algorithms:
      self.logger.info('Algorithm : %s', algorithm)
      
      print(algorithm)
 
      self.set_sample_weight(algorithm)
      regressor = self.select_regressor(algorithm)
      
      if algorithm == 'SVR':
        grid = ParameterGrid(self.get_grid())
      else:
        grid = ParameterGrid(self.parameters[algorithm])
      
      manager = Manager()
      scores = manager.list()
      hyperparam = manager.list()
      
      pool = ThreadPool(processes = self.num_cores)
      
      self.algorithm_start_time = time.time()
      for g in grid:
        pool.apply(self.try_hyperparameter, args = (algorithm, regressor, g, scores, hyperparam))

      pool.close()
      pool.join()
			
      self.logger.debug('- - - - - - - - - BEST PARAMETER SELECTED - - - - - - - - - - - -')
      self.algorithm_end_time = time.time()

      min_score_index = scores.index(min(scores))
      best_parameter = hyperparam[min_score_index]

      chosen_model = regressor.set_params(**best_parameter)
      #chosen_model = pickle.load(open('./results/ml/runbest/results_without_apriori/conf_1/' + self.scenario + '_' + self.input_name + '_' + algorithm + '_chosen_model.pickle', 'rb'))
      #best_parameter = ""
      self.set_sample_weight(algorithm)
      chosen_model.fit(self.train_features, self.train_labels, self.sample_weight)
      if algorithm == 'NeuralNetwork':
        torch.save(chosen_model.state_dict(), os.path.join('outputs', self.scenario + '_' + self.input_name + '_' + algorithm + '_chosen_model.pth'))
      else:
        pickle.dump(chosen_model, open(os.path.join('outputs', self.scenario + '_' + self.input_name + '_' + algorithm + '_chosen_model.pickle'), 'wb'))

      train_predicted = chosen_model.predict(self.train_features)
      cv_predicted = chosen_model.predict(self.cv_features)
      test_predicted = chosen_model.predict(self.test_features)
      
      train_error, train_fraction = self.data_preparation.mean_absolute_percentage_error('train', train_predicted.reshape(-1, 1), algorithm, self.scenario, self.input_name)
      cv_error, cv_fraction = self.data_preparation.mean_absolute_percentage_error('cv', cv_predicted.reshape(-1, 1), algorithm, self.scenario, self.input_name)
      test_error, test_fraction = self.data_preparation.mean_absolute_percentage_error('test', test_predicted.reshape(-1, 1), algorithm, self.scenario, self.input_name)
       
      self.logger.debug('MAPE on test set : %s', test_error)
      self.logger.info('Best Parameter : %s', best_parameter)
      self.logger.debug('- - - - - - - - - - - - - - - - - - - - -\n\n')
        
      self.all_hyper_parameters[algorithm + '_model'] = chosen_model
      self.all_hyper_parameters[algorithm + '_train_mape_error'] = train_error
      self.all_hyper_parameters[algorithm + '_test_mape_error'] = test_error
      self.all_hyper_parameters[algorithm + '_validation_mape_error'] = cv_error
      self.all_hyper_parameters[algorithm + '_train_fraction_error'] = train_fraction
      self.all_hyper_parameters[algorithm + '_test_fraction_error'] = test_fraction
      self.all_hyper_parameters[algorithm + '_validation_fraction_error'] = cv_fraction
      self.all_hyper_parameters[algorithm + '_best_parameter'] = best_parameter
      self.all_hyper_parameters[algorithm + '_time'] = (self.algorithm_end_time - self.algorithm_start_time) / 60

      self.data_preparation.plot(algorithm, best_parameter, train_predicted, cv_predicted, test_predicted)
      if self.hybrid_ml == 'on' and algorithm == 'NeuralNetwork':
        self.train_labels, self.train_features = self.data_preparation.reset_train_set()
        
    self.logger.info('Finished training the models')
    self.end_time = time.time()
    self.execution_time = (self.end_time - self.start_time) / 60
    self.logger.info('Found the best parameter for all given algorithms in : %s min', self.execution_time)
    
    csvfile = open(os.path.join('outputs', self.scenario + '_' + self.input_name + '_output.csv'), 'w')
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerow(['Algorithm', 'Train Mape Error (%)', 'Cross-Validation Mape Error (%)', 'Test Mape Error (%)', 'Train Fraction Error (%)', 'Cross-Validation Fraction Error (%)', 'Test Fraction Error (%)', 'Best Parameters', 'Found in (min)'])
    for algorithm in self.algorithms:
      writer.writerow([algorithm, self.all_hyper_parameters[algorithm + '_train_mape_error'],
				  self.all_hyper_parameters[algorithm + '_validation_mape_error'],
				  self.all_hyper_parameters[algorithm + '_test_mape_error'],
                                  self.all_hyper_parameters[algorithm + '_train_fraction_error'],
                                  self.all_hyper_parameters[algorithm + '_validation_fraction_error'],
                                  self.all_hyper_parameters[algorithm + '_test_fraction_error'],
				  self.all_hyper_parameters[algorithm + '_best_parameter'],
                                  self.all_hyper_parameters[algorithm + '_time']])
    
