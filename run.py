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

import argparse
from model_selection import ML
from data_preparation import DataPreparation

parser = argparse.ArgumentParser()
parser.add_argument('seed', metavar = 'seed', type = int, help = 'seed for ML')
parser.add_argument('--config', '-c', metavar = 'config_file', type = str, help = 'config file path', required = True)
parser.add_argument('--input', '-i', metavar = 'input_file', type = str, help = 'input file path', required = True)
parser.add_argument('--analytical', '-a', metavar = 'analytical_model_file', type = str, help = 'analytical model file path')

args = parser.parse_args()

learner = ML(args.seed, args.config, args.input, args.analytical)
learner.train_model_with_HP()
