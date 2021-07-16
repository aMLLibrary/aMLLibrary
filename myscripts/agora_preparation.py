#!/usr/bin/env python3
import os
import pandas as pd
import configparser
import numpy as np

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

# Initialize list of apps
apps = 'blockscholes bodytrack freqmine kmeans stereomatch swaptions'.split()

# Initialize relevant paths
orig_datasets_folder = os.path.join('inputs', 'agora_orig')
final_datasets_folder = os.path.join('inputs', 'agora')
final_configs_folder = os.path.join('example_configurations', 'agora')
configs_blueprint_path = os.path.join('example_configurations',
                                      'agora_blueprint.ini')
if not os.path.isdir(final_datasets_folder):
  os.mkdir(final_datasets_folder)
if not os.path.isdir(final_configs_folder):
  os.mkdir(final_configs_folder)


# Loop over apps
for app in apps:
  print("\n", ">>>>>", app)
  # Get files paths
  orig_dataset_app_subfolder = os.path.join(orig_datasets_folder, app)
  final_dataset_app_subfolder = os.path.join(final_datasets_folder, app)
  config_files_app_subfolder = os.path.join(final_configs_folder, app)

  # Create subfolders
  if not os.path.isdir(final_dataset_app_subfolder):
    os.mkdir(final_dataset_app_subfolder)
  if not os.path.isdir(config_files_app_subfolder):
    os.mkdir(config_files_app_subfolder)

  # Set maximum number of iterations used for this app
  maxiter = 100 if app == 'stereomatch' else 40
  listdir = os.listdir(orig_dataset_app_subfolder)

  # Loop over files of different iterations
  for it in range(1, maxiter+1):
    covariate_file = f'data_itr_{it}.csv'
    target_file = f'target_exec_time_ms_itr_{it}.csv'

    # Check whether the two files actually exist
    if not (covariate_file in listdir and target_file in listdir):
      exit(f"Error: {covariate_file} or {target_file} does not exist")
    dataset_file_path = os.path.join(orig_dataset_app_subfolder,
                                     covariate_file)
    target_file_path = os.path.join(orig_dataset_app_subfolder, target_file)

    # Join covariates and target into a single, full dataset
    df = pd.read_csv(dataset_file_path, encoding='utf-8')
    df_tar = pd.read_csv(target_file_path, encoding='utf-8')
    df['exec_time_ms'] = df_tar['exec_time_ms']

    # Change some column names to a standardized one
    thr_name = 'nThreads'
    df = df.rename(columns={'threads':       thr_name,
                            'workingthread': thr_name,
                            'num_threads':   thr_name})

    # Add threads column if nonexistent
    if not thr_name in df:
      df.loc[:,thr_name] = 1

    # Save new dataset to file
    it00 = str(it).zfill(3)
    df_path = os.path.join(final_dataset_app_subfolder,
                           f'{app}_itr_{it00}.csv')
    df.to_csv(df_path, index=False)
    print("Saved dataset to", df_path)

    # Read blueprint configuration file (refreshed at each iteration)
    config = configparser.ConfigParser()
    config.read(configs_blueprint_path)

    # Modify config
    config['DataPreparation']['input_path'] = f'"{df_path}"'
    if app == 'freqmine':
      config['DataPreparation']['inverse'] = f"['{thr_name}', 'THRESHOLD']"
    elif app in ['stereomatch', 'bodytrack']:
      config['DataPreparation']['product_max_degree'] = '2'

    # Set min_child_weight according to data size
    ndata = df.shape[0]
    if ndata <= 11:
      weights = str( [1] )
    elif ndata <= 20:
      weights = str( [1,2] )
    elif ndata <= 30:
      weights = str( [1,2,3] )
    elif ndata <= 100:
      linsp = np.linspace(1, int(0.1 * ndata), 3)
      weights = str( [int(_) for _ in linsp] )
    else:
      linsp = np.linspace(int(0.01 * ndata), int(0.05 * ndata), 4)
      weights = str( [int(_) for _ in linsp] )
    print('n =', ndata, '-> min_child_weight =', weights)
    config['XGBoost']['min_child_weight'] = weights

    # Add Sequential Feature Selection
    maxfeatures = {'bodytrack': '5', 'freqmine': '2',
                  'kmeans': '3', 'stereomatch': '5'
                  }
    if app in maxfeatures:
      config['FeatureSelection'] = {}
      config['FeatureSelection']['method'] = '"SFS"'
      config['FeatureSelection']['max_features'] = maxfeatures[app]
      config['FeatureSelection']['folds'] = '5'

    # Save config to file
    config_file_path = os.path.join(config_files_app_subfolder,
                                    f'{app}_itr_{it00}.ini')
    with open(config_file_path, 'w') as f:
      config.write(f)
    print("Saved configs to", config_file_path)
