#!/usr/bin/env python3
import os
import pandas as pd
import configparser

# Go to parent directory
os.chdir(os.pardir)

# Initialize list of apps
apps = 'blockscholes bodytrack freqmine kmeans stereomatch swaptions'.split()
#apps = ['blockscholes']
apps = []



##### WRITE FULL DATASETS #####
base_folder = os.path.join('inputs', 'agora')
for app in apps:
  print("\n", ">>>>>", app)
  # Get files paths
  app_folder = os.path.join(base_folder, app)
  full_dataset_app_folder = os.path.join(app_folder, 'full')
  if not os.path.isdir(full_dataset_app_folder):
    os.mkdir(full_dataset_app_folder)

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
    covariate_file_path = os.path.join(app_folder, covariate_file)
    target_file_path = os.path.join(app_folder, target_file)

    # Join covariates and target into a single, full dataset
    df = pd.read_csv(covariate_file_path, encoding='utf-8')
    df_tar = pd.read_csv(target_file_path, encoding='utf-8')
    df['exec_time_ms'] = df_tar['exec_time_ms']

    # Change some column names to a standardized one
    thr_name = 'nThreads'
    df.rename({'threads':       thr_name,
               'workingthread': thr_name,
               'num_threads':   thr_name})

    # Add threads column if nonexistent
    if not thr_name in df:
      df.loc[:,thr_name] = 1

    # Save new dataset to file
    df_path = os.path.join(full_dataset_app_folder, f'itr{it}.csv')
    df.to_csv(df_path, index=False)
    print(f"Successfully saved to {df_path}")



##### WRITE CONFIGURATION FILES #####
base_configs_folder = 'example_configurations'
final_configs_folder = os.path.join(base_configs_folder, 'agora')
blueprint_path = os.path.join(base_configs_folder, 'agora_blueprint.ini')
if not os.path.isdir(final_configs_folder):
  os.mkdir(final_configs_folder)

# Read blueprint configuration file
config = configparser.ConfigParser()
config.read(blueprint_path)
print(config.sections())
