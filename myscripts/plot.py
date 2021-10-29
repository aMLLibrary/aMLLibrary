#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd

import model_mapes
import prediction_mapes

# User parameters
_prediction = False
_output_fold = '../zz_old/coliva/outputs/next_sfs'
_shard = 'val'
_plot_title = 'IDK'
_fig_size = (10,15)
_subplots_layout = (3,1)
_plots_colors = 'rainbow'
_max_mape = 1.0
_plot_filename = 'idk.png'


def plot(prediction, output_fold, shard, plot_title, fig_size, subplots_layout,
         plots_colors, max_mape, plot_filename):
    # Allows running this script from both this folder and from root folder
    if os.getcwd() == os.path.dirname(__file__):
        os.chdir(os.pardir)

    # Get MAPE dataframes
    if prediction:
        dfs = prediction_mapes.get_prediction_mapes(output_fold)
    else:
        dfs = model_mapes.get_model_mapes(output_fold, shard)

    print("\n>>> Datasets:")
    for key, df in dfs.items():
        print(key, ":\n", df, "\n", sep="")

    # Initialize plot
    fig = plt.figure(figsize=fig_size)
    fig.suptitle(plot_title, y=0.92, size=14)

    # Loop over devices
    for idx, device in enumerate(dfs):
        df = dfs[device]
        techniques = tuple(df.keys())
        num_techniques = len(techniques)

        # Initialize colormap
        if isinstance(plots_colors, str):
            colormap = plt.cm.get_cmap(name=plots_colors, lut=num_techniques)
        elif isinstance(plots_colors, list) or isinstance(plots_colors, tuple):
            colormap = lambda x: plots_colors[x]
        else:
            raise ValueError("Error with colormap")

        ax = fig.add_subplot(*subplots_layout, idx+1)
        ax.set_title(device)
        bests = df.apply(min, axis=1)
        out_of_range = False
        # Loop over regression techniques
        for tech in df.columns:
            for idx, mape in df[tech].iteritems():
                color = colormap(techniques.index(tech))
                if mape > max_mape:
                    out_of_range = True
                    ax.scatter(idx, max_mape-0.02, color=color, label=tech,
                               marker='^')
                else:
                    ax.scatter(idx, mape, color=color, label=tech, marker='o',
                               s=20)
                    if mape == bests[idx]:
                        ax.scatter(idx, mape, color=color, label=tech,
                                   marker='x', s=60)
        # Set axes utilities
        ax.set_xticks(df.index)
        ax.set_ylim((0.0, max_mape))
        ax.set_yticks(np.arange(0.0, max_mape+0.01, 0.5), minor=False)
        ax.set_yticks(np.arange(0.0, max_mape+0.01, 0.1), minor=True)
        ax.set_ylabel("MAPE")
        ax.grid(axis='y', which='major', alpha=1.0)
        ax.grid(axis='y', which='minor', alpha=0.25)

        # Create and set legend
        handles = []
        labels = []
        for idx, tech in enumerate(techniques):
            handles.append(Line2D([0],[0], marker='o', lw=0,
                                  color=colormap(idx)))
            labels.append(tech)
        if out_of_range:
            oor_handle = Line2D([0],[0],marker='^', ms=8, lw=0, color='silver')
            handles.append(oor_handle)
            labels.append(f"> {max_mape}")
        best_handle = (Line2D([0],[0], marker='o', ms=8, lw=0, color='silver'),
                       Line2D([0],[0], marker='x', mew=2, ms=12, lw=0,
                              color='silver'))
        handles.append(best_handle)
        labels.append("best")
        ax.legend(handles=handles, labels=labels)

    # Save figure to file
    fig.savefig(plot_filename, bbox_inches='tight')
    print("\nSaved to", plot_filename)


if __name__ == '__main__':
    plot(_prediction, _output_fold, _shard, _plot_title, _fig_size,
         _subplots_layout, _plots_colors, _max_mape, _plot_filename)
