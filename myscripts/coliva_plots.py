#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

SFS = False

# Initialize relevant paths
base_datasets_folder = os.path.join('/home/bruno/DEIB_Dropbox/aml', 'aml_outputs', 'output_coliva')

# Initialize list of used techniques
techniques = ['XGBOOST', 'LR_RIDGE']
layers_to_keep = (3,5,6,8,9,10,12,13,14,16,17,18)

# Initialize results dataframes
dfs = {}

# Loop over devs
for dev in os.listdir(base_datasets_folder):
  print("\n", ">>>>>", dev)
  dev_folder = os.path.join(base_datasets_folder, dev)
  # Initialize empty dataframe for this device
  df = pd.DataFrame(columns=techniques)
  # Loop over iterations of device
  for expp in os.listdir(dev_folder):
    check = ('sfs' not in expp) if SFS else ('sfs' in expp)
    if check or '20' in expp:
        continue
    rm = ('sfs_'+dev+'_') if SFS else (dev+'_')
    print(rm)
    exp_num = int(expp.replace(rm, ''))
    results_file_path = os.path.join(dev_folder, expp, 'results')
    # Read results from file
    if not os.path.exists(results_file_path):
        continue
    with open(results_file_path, 'r') as f:
      for lin in f.readlines():
        result_str = '      Best result for Technique.'
        mape_str = 'Validation MAPE is '
        if result_str in lin:
          line_list = lin.replace(result_str, '').split(' - ')
          tech = line_list[0]
          mape = float(line_list[4].replace(mape_str, ''))
          print(exp_num, tech, mape, sep=" -- ")
          df.loc[exp_num, tech] = mape
  dfs[dev] = df



# Plot results
fig = plt.figure(figsize=(10,15))
suptit = 'Hyperopt + SFS' if SFS else 'Hyperopt'
fig.suptitle(suptit, y=0.92, size=14)
for idx, dev in enumerate(dfs):
  ax = fig.add_subplot(3,1,idx+1)
  ax.set_title(dev)
  df = dfs[dev]
  maxx = np.max(df[techniques])[0]
  for tech in df.columns:
    if tech == 'XGBOOST':
        ax.scatter(df.index, df[tech], s=40, label=tech)
    else:
        ax.scatter(df.index, df[tech], s=10, label=tech)
  ax.set_xlabel("Iteration")
  ax.set_xticks(layers_to_keep)
  ax.set_ylim((0.0, 1.0))
  ax.set_yticks(np.arange(0.0, 1.0, 0.5), minor=False)
  ax.set_yticks(np.arange(0.0, 1.0, 0.25), minor=True)
  ax.set_ylabel("MAPE")
  ax.grid(axis='y', which='major', alpha=1.0)
  ax.grid(axis='y', which='minor', alpha=0.25)
  ax.legend()

filename = "coliva_plots_sfs.png" if SFS else "coliva_plots.png"
fig.savefig(filename)
print("Saved to", filename)
