#!/usr/bin/env python3
"""
Copyright 2019 Marco Lattuada
Copyright 2022 Nahuel Coliva

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
import argparse
import os
import subprocess
import sys

import time
import shutil

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
  
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
  
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

import sequence_data_processing
from model_building.predictor import Predictor

def generate_tests():
    """
    Generates a dictionary with several models and configurations.

    The tests generated are:
    -ernest: tests the ernest feature;
    -faas_test: prepares output for faas_predict while testing DecisionTree, KFold and HoldOut;
    -faas_predict: tests the prediction module;
    -faas_test_sfs: tests SFS;
    -faas_test_hyperopt: tests the hyperopt integration;
    -faas_test_hyperopt_sfs: tests the integration between hyperopt and SFS;
    -faas_test_xgboost_fs: tests feature selection with XGBoost.
    """
    tests = [
        {
            'Name': 'ernest',
            'General':{
                'run_num': 1,
                'techniques':['LRRidge'],
                'validation': 'All',
                'hp_selection': 'All',
                'y': 'y'
            },
            'DataPreparation':{
                'input_path': parent+'/inputs/simplest.csv',
                'ernest': True,
                'rename_columns': {"x1": "cores", "x2": "datasize"}
            },
            'LRRidge':{
                'alpha': [0.1]
            }
        },
        {
            'Name': 'faas_test',
            'General':{
                'run_num': 1,
                'techniques': ['LRRidge', 'DecisionTree'],
                'hp_selection': 'KFold',
                'validation': 'HoldOut',
                'folds': 4,
                'hold_out_ratio': 0.2,
                'y': 'ave_response_time'
            },
            'DataPreparation':{
                'input_path': parent+'/inputs/faas_test.csv',
                'inverse': ['Lambda'],
                'product_max_degree': 2,
                'product_interactions_only': True
            },
            'LRRidge':{
                'alpha': [0.02, 0.1, 1.0]
            },
            'DecisionTree':{
                'criterion': ['mse'],
                'max_depth': [2,5],
                'max_features': ['auto'],
                'min_samples_split': [0.01],
                'min_samples_leaf': [0.01]
            }
        },
        {
            'Name': 'faas_predict',
            'General':{
                'y': 'ave_response_time'
            },
            'DataPreparation':{
                'input_path': parent+'/inputs/faas_predict.csv'
            }
        },
        {
            'Name': 'faas_test_sfs',
            'General':{
                'run_num': 1,
                'techniques': ['LRRidge'],
                'hp_selection': 'KFold',
                'validation': 'HoldOut',
                'folds': 4,
                'hold_out_ratio': 0.2,
                'y': 'ave_response_time'
            },
            'DataPreparation':{
                'input_path': parent+'/inputs/faas_test.csv',
                'inverse': ['Lambda']
            },
            'FeatureSelection':{
                'method': 'SFS',
                'max_features': 3,
                'folds': 3
            },
            'LRRidge':{
                'alpha': [0.02, 0.1, 1]
            }
        },
        {
            'Name': 'faas_test_hyperopt',
            'General':{
                'run_num': 1,
                'techniques': ['LRRidge'],
                'hp_selection': 'KFold',
                'validation': 'HoldOut',
                'folds': 4,
                'hold_out_ratio': 0.2,
                'y': 'ave_response_time',
                'hyperparameter_tuning': 'Hyperopt',
                'hyperopt_max_evals': 10,
                'hyperopt_save_interval': 5
            },
            'DataPreparation':{
                'input_path': parent+'/inputs/faas_test.csv',
                'inverse': ['Lambda']
            },
            'LRRidge':{
                'alpha': ['loguniform(0.01,1)']
            }
        },
        {
            'Name': 'faas_test_hyperopt_sfs',
            'General':{
                'run_num': 1,
                'techniques': ['LRRidge'],
                'hp_selection': 'KFold',
                'validation': 'HoldOut',
                'folds': 4,
                'hold_out_ratio': 0.2,
                'y': 'ave_response_time',
                'hyperparameter_tuning': 'Hyperopt',
                'hyperopt_max_evals': 10,
                'hyperopt_save_interval': 5
            },
            'DataPreparation':{
                'input_path': parent+'/inputs/faas_test.csv',
                'inverse': ['Lambda']
            },
            'FeatureSelection':{
                'method': 'SFS',
                'max_features': 3,
                'folds': 3
            },
            'LRRidge':{
                'alpha': ['loguniform(0.01,1)']
            }
        },
        {
            'Name': 'faas_test_xgboost_fs',
            'General':{
                'run_num': 1,
                'techniques': ['LRRidge'],
                'hp_selection': 'KFold',
                'validation': 'HoldOut',
                'folds': 2,
                'hold_out_ratio': 0.2,
                'y': 'ave_response_time'
            },
            'DataPreparation':{
                'input_path': parent+'/inputs/faas_test.csv'
            },
            'FeatureSelection':{
                'method': 'XGBoost',
                'max_features': 2,
                'XGBoost_tolerance': 0.4
            },
            'LRRidge':{
                'alpha': [0.1,0.2]
            }
        }
    ]
    return tests

def main():
    """
    This script is used to self check the library.Quality of the results is not analyzed nor compared with any reference
    
    It generates several possible configurations and runs them
    """
    parser = argparse.ArgumentParser(description="Performs regression tests")
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-o', "--output", help="output folder where all the models will be stored", default=parent+"/output_test_configurations")
    args = parser.parse_args()

    done_file_flag = os.path.join(args.output, 'done')

    try:
        os.mkdir(args.output)
    except FileExistsError:
        if os.path.exists(done_file_flag):
            print(args.output+" already exists with all tests performed. Deleting and starting anew...")
            time.sleep(3)
            shutil.rmtree(args.output)
            os.mkdir(args.output)
        else:
            print(args.output+" already exists. Restarting from where we left...")
            time.sleep(3)

    tests = generate_tests()

    #Perform tests
    outcomes = []
    for configuration in tests:
        test_name = configuration.pop('Name')
        output_path = os.path.join(args.output,test_name)

        try:
            #Check if the test was already performed in a previous incomplete run
            test_done_file_flag = os.path.join(output_path, 'done')
            if os.path.exists(test_done_file_flag):
                print(test_name,"already performed",sep=" ",end="\n\n\n")
                outcomes.append(str(test_name)+" already performed")
                continue
            print("Starting",test_name,sep=" ")

            if test_name == 'faas_predict':
                # Build object
                predictor_obj = Predictor(regressor_file=os.path.join(args.output,'faas_test/LRRidge.pickle'), output_folder=output_path, debug=args.debug)

                # Perform prediction reading from a config file
                predictor_obj.predict(config_file=configuration, mape_to_file=True)
            else:
                sequence_data_processor = sequence_data_processing.SequenceDataProcessing(configuration, debug=args.debug, output=output_path)
                sequence_data_processor.process()
        except Exception as e:
            print("Exception",e,"raised", sep=' ')
            outcomes.append(str(test_name)+" failed with exception "+str(e))
        else:
            outcomes.append(str(test_name)+" successfully run")
        print('\n\n\n')

    #Print results
    print('\n\n\n\n*************Test Results*************')
    i = 0
    for outcome in outcomes:
        i += 1
        print(str(i)+')',outcome, sep=' ')

    # Create success flag file
    with open(done_file_flag, 'wb') as f:
        pass

if __name__ == '__main__':
    main()
