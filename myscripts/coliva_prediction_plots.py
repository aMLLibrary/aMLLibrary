#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

# Initialize relevant paths
root_folder_path = os.path.join('/home/bruno/DEIB_Dropbox/aml',
    'aml_outputs', 'output_coliva', 'extrapolation_vgg19')

# Initialize list of used techniques
devices = ('rp3', 'tegrax2')
layers = (2,3,5,6,8,9,10,11,13,14,15,16,18,19,20,21)
techniques = {'ridge': 'LR_RIDGE', 'xgboost': 'XGBOOST'}

dfs = {}
for dev in devices:
    df = pd.DataFrame(index=layers, columns=techniques.values())
    for lay in layers:
        lay00 = str(lay).zfill(2)
        for tech in techniques:
            # Get subfolder path for specific experiment
            subfolder_name = '_'.join((dev, lay00, tech))
            subfolder_path = os.path.join(root_folder_path, subfolder_name)
            # Read MAPE of experiment from file
            mape_file_path = os.path.join(subfolder_path, 'mape.txt')
            with open(mape_file_path, 'r') as f:
                mape = f.read().strip('\n')
            df.loc[lay, techniques[tech]] = float(mape)
    dfs[dev] = df
    print(f">> {dev}:")
    print(df)

# Plot results
fig = plt.figure(figsize=(10,15))
fig.suptitle('Extrapolation from VGG16 to VGG19', y=0.92, size=14)
for idx, dev in enumerate(devices):
    df = dfs[dev]
    ax = fig.add_subplot(2,1,idx+1)
    maxx = round(df.max().max(), 3)
    ax.set_title(f"{dev} (max MAPE = {maxx})")
    for tech in techniques.values():
        if tech == 'XGBOOST':
            ax.scatter(df.index, df[tech], label=tech, marker='x')
        else:
            ax.scatter(df.index, df[tech], label=tech)
    ax.set_xlabel("Iteration")
    ax.set_xticks(layers)
    ax.set_ylim((0.0, 1.0))
    ax.set_yticks(np.arange(0.0, 1.01, 0.5), minor=False)
    ax.set_yticks(np.arange(0.0, 1.01, 0.25), minor=True)
    ax.set_ylabel("MAPE")
    ax.grid(axis='y', which='major', alpha=1.0)
    ax.grid(axis='y', which='minor', alpha=0.25)
    ax.legend()

filename = "coliva_plots_extrapolation.png"
fig.savefig(filename)
print("Saved to", filename)
