#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd

import generic_get_mapes

# User parameters
output_fold_ = '../outputs/output_coliva/next'
plot_title_ = 'Hyperopt + SFS'
fig_size_ = (10,15)
subplots_layout_ = (3,1)
max_mape_ = 1.0
plot_file_name_ = 'next_aml.png'


def plot(output_fold, plot_title, fig_size, subplots_layout, max_mape,
         plot_file_name):
    # Allows running this script from both this folder and from root folder
    if os.getcwd() == os.path.dirname(__file__):
        os.chdir(os.pardir)

    # Get MAPE dataframes
    dfs = generic_get_mapes.get_mapes(output_fold)

    # Initialize plot
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(plot_title, y=0.92, size=14)
    # Loop over devices
    for idx, device in enumerate(dfs):
        df = dfs[device]
        ax = fig.add_subplot(*subplots_layout, idx+1)
        maxx = round(df.max().max(), 3)
        ax.set_title(f"{device} (max MAPE = {maxx})")
        # Loop over regression techniques
        for tech in df.columns:
            if tech == 'XGBOOST':
                ax.scatter(df.index, df[tech], label=tech, marker='x')
            else:
                ax.scatter(df.index, df[tech], label=tech)
        # Set axes utilities
        ax.set_xticks(df.index)
        ax.set_ylim((0.0, max_mape))
        ax.set_yticks(np.arange(0.0, max_mape+0.01, 0.5), minor=False)
        ax.set_yticks(np.arange(0.0, max_mape+0.01, 0.1), minor=True)
        ax.set_ylabel("MAPE")
        ax.grid(axis='y', which='major', alpha=1.0)
        ax.grid(axis='y', which='minor', alpha=0.25)
        ax.legend()
    # Save figure to file
    fig.savefig(plot_file_name, bbox_inches='tight')
    print("\nSaved to", plot_file_name)


if __name__ == '__main__':
    plot(output_fold_, plot_title_, fig_size_, subplots_layout_, max_mape_,
         plot_file_name_)
