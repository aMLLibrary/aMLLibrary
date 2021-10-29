#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd

import generic_get_mapes

# User parameters
_output_fold = '../outputs/output_coliva/next'
_shard = 'val'
_plot_title = 'Hyperopt + SFS'
_fig_size = (10,15)
_subplots_layout = (3,1)
_max_mape = 1.0
_plot_filename = 'next_aml.png'


def plot(output_fold, shard, plot_title, fig_size, subplots_layout, max_mape,
         plot_filename):
    # Allows running this script from both this folder and from root folder
    if os.getcwd() == os.path.dirname(__file__):
        os.chdir(os.pardir)

    # Get MAPE dataframes
    dfs = generic_get_mapes.get_mapes(output_fold, shard)

    # Initialize plot
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(plot_title, y=0.92, size=14)
    # Loop over devices
    for idx, device in enumerate(dfs):
        df = dfs[device]
        ax = fig.add_subplot(*subplots_layout, idx+1)
        ax.set_title(device)
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
    fig.savefig(plot_filename, bbox_inches='tight')
    print("\nSaved to", plot_filename)


if __name__ == '__main__':
    plot(_output_fold, _shard, _plot_title, _fig_size, _subplots_layout,
         _max_mape, _plot_filename)
