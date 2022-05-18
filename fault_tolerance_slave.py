#!/usr/bin/env python3
"""
Copyright 2019 Marco Lattuada

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
import sys

import sequence_data_processing


def generate_test():
    test = {
        'General':{
            'run_num': 1,
            'techniques': ['LRRidge','DecisionTree'],#['LRRidge','DecisionTree','XGBoost'],
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
            'input_path': 'inputs/faas_test.csv',
            'inverse': ['Lambda']
        },
        'FeatureSelection':{
            'method': 'SFS',
            'max_features': 3,
            'folds': 3
        },
        'LRRidge':{
            'alpha': ['loguniform(0.01,1)']
        },
        'DecisionTree':{
            'criterion': ['mse'],
            'max_depth': [2,3,5],
            'max_features': ['auto'],
            'min_samples_split': [0.01,0.1,0.5],
            'min_samples_leaf': [0.01,0.1,0.5]
        },
        'XGBoost':{
            'min_child_weight': [1],
            'gamma': ['loguniform(0.1,10)'],
            'n_estimators': [1000],
            'learning_rate': ['loguniform(0.01,1)'],
            'max_depth': [100]
        }
    }
    return test


def main():
    """
    Script used to stress test the fault tolerance of the library

    Checks that interrupting the tests performed by test.py is fault tolerant
    """
    parser = argparse.ArgumentParser(description="Performs regression tests")
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-o', "--output", help="output folder where all the models will be stored", default="output_fault_tolerance")
    args = parser.parse_args()

    done_file_flag = os.path.join(args.output, 'done')

    try:
        os.mkdir(args.output)
    except FileExistsError:
        if os.path.exists(done_file_flag):
            print(args.output+" already exists. Terminating the program...")
            sys.exit(1)

    test = generate_test()
    
    try:
        sequence_data_processor = sequence_data_processing.SequenceDataProcessing(test, debug=args.debug, output=args.output)
        sequence_data_processor.process()

    except Exception as e:
        print("Exception",e,"raised","\nFault tolerance failed", sep=' ')
        sys.exit(1)

    # Create success flag file
    with open(done_file_flag, 'wb') as f:
        pass



if __name__ == '__main__':
    main()
