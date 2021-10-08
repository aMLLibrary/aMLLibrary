#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

# Initialize relevant paths
base_datasets_folder = os.path.join('/home/bruno/DEIB_Dropbox/aml', 'aml_outputs', 'output_coliva')

# Initialize list of used techniques
techniques = ('XGBOOST', 'LR_RIDGE')

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
    if 'sfs' in expp or '20' in expp:  # or 'sfs' not in expp ...
        continue
    exp_num = int(expp.replace(dev+'_', ''))  # or 'sfs_'+dev...
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
for idx, dev in enumerate(dfs):
  ax = fig.add_subplot(3,1,idx+1)
  ax.set_title(dev)
  df = dfs[dev]
  for tech in df.columns:
    if tech == 'XGBOOST':
        ax.scatter(df.index, df[tech], s=40, label=tech)
    else:
        ax.scatter(df.index, df[tech], s=10, label=tech)
  ax.set_xlabel("Iteration")
  ax.set_xticks(tuple(range(1,19)))
  ax.set_ylabel("MAPE")
  ax.grid(axis='y')
  ax.legend()

#fig.savefig("coliva_plots.pdf")
fig.savefig("coliva_plots.png")  # or ..._sfs.png
