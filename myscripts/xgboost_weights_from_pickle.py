#!/usr/bin/env python3
import os, sys

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)
sys.path.append('.')

import pickle
import regressor  # local file

def main():
  # Get folder from first arg
  output_folder = sys.argv[1]
  # Check if folder exists
  if not os.path.isdir(output_folder):
    exit(f"{output_folder} folder does not exist")
  # Check if XGBoost object exists
  xgboost_pickle_path = os.path.join(output_folder, 'XGBoost.pickle')
  if not os.path.exists(xgboost_pickle_path):
    exit(f"{output_folder} folder does not contain 'XGBoost.pickle'")
  with open(xgboost_pickle_path, 'rb') as f:
    regressor = pickle.load(f)
  weights = regressor.get_regressor().get_booster().get_fscore()
  weights_sum = sum(weights.values())
  for key in weights:
    weights[key] /= weights_sum
  print(f"Weights from '{xgboost_pickle_path}':\n  ", weights)



if __name__ == '__main__':
  main()
