#!/usr/bin/env python3
import os
import pandas as pd
import configparser

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

# Initialize list of apps
apps = 'blockscholes bodytrack freqmine kmeans stereomatch swaptions'.split()
#apps = ['blockscholes']

# Initialize relevant paths
base_datasets_folder = os.path.join('inputs', 'agora')
base_configs_folder = 'example_configurations'
final_configs_folder = os.path.join(base_configs_folder, 'agora')
blueprint_path = os.path.join(base_configs_folder, 'agora_blueprint.ini')
if not os.path.isdir(final_configs_folder):
  os.mkdir(final_configs_folder)

# Loop over apps
for app in apps:
  print("\n", ">>>>>", app)
  # Get files paths
  app_folder = os.path.join(base_datasets_folder, app)
  full_dataset_app_folder = os.path.join(app_folder, 'full')

  # Create folders and subfolders
  if not os.path.isdir(full_dataset_app_folder):
    os.mkdir(full_dataset_app_folder)
  config_files_subfolder = os.path.join(final_configs_folder, app)
  if not os.path.isdir(config_files_subfolder):
    os.mkdir(config_files_subfolder)

  # Set maximum number of iterations used for this app
  maxiter = 100 if app == 'stereomatch' else 40
  listdir = os.listdir(app_folder)

  # Loop over files of different iterations
  for it in range(1, maxiter+1):
    covariate_file = f'data_itr_{it}.csv'
    target_file = f'target_exec_time_ms_itr_{it}.csv'

    # Check whether the two files actually exist
    if not (covariate_file in listdir and target_file in listdir):
      exit(f"Error: {covariate_file} or {target_file} does not exist")
    dataset_file_path = os.path.join(app_folder, covariate_file)
    target_file_path = os.path.join(app_folder, target_file)

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
    df_path = os.path.join(full_dataset_app_folder, f'itr{it}.csv')
    df.to_csv(df_path, index=False)
    print("Saved dataset to", df_path)

    # Read blueprint configuration file
    config = configparser.ConfigParser()
    config.read(blueprint_path)

    # Modify config
    config['DataPreparation']['input_path'] = f'"{df_path}"'
    if app == 'freqmine':
      config['DataPreparation']['inverse'] = f"['{thr_name}', 'THRESHOLD']"
    elif app in ['stereomatch', 'bodytrack']:
      config['DataPreparation']['product_max_degree'] = '2'

    # Save config to file
    config_file_path = os.path.join(config_files_subfolder,
                                    f'{app}_itr{it}.ini')
    with open(config_file_path, 'w') as f:
      config.write(f)
    print("Saved configs to", config_file_path)
