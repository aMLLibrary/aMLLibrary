#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

# Initialize list of apps
apps = 'blockscholes bodytrack freqmine kmeans stereomatch swaptions'.split()
#apps = ['blockscholes']

# Initialize relevant paths
base_datasets_folder = os.path.join('inputs', 'agora')

# Initialize plot utilities
pdf = PdfPages('plots.pdf')

# Loop over apps
for app in apps:
  print("\n", ">>>>>", app)
  # Get files paths
  app_folder = os.path.join(base_datasets_folder, app)
  full_dataset_app_folder = os.path.join(app_folder, 'full')

  # Set maximum number of iterations used for this app
  maxiter = 100 if app == 'stereomatch' else 40

  # Read full dataset
  df_path = os.path.join(full_dataset_app_folder, f'itr{maxiter}.csv')
  df = pd.read_csv(df_path)
  print("Opened", df_path)

  # Get covariates and target
  y_name = 'exec_time_ms'
  covariates = set(df) - set([y_name])
  thr_name = 'nThreads'
  
  # Plot all features
  fig = plt.figure(figsize=(16,10))
  fig.suptitle(f"{app}, iteration {maxiter}")
  idx = 1
  for col in covariates:
    fig.add_subplot(2, 3, idx)
    idx += 1
    plt.scatter(df[col], df[y_name], marker='.')
    plt.xlabel(col)
    if col == thr_name:
      fig.add_subplot(2, 3, idx)
      idx += 1
      plt.scatter(1/df[col], df[y_name], marker='.')
      plt.xlabel(f"1/{col}")
    plt.ylabel(y_name)
  pdf.savefig(fig)
  plt.clf()

pdf.close()
