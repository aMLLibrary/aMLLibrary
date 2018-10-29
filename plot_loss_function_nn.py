"""
Copyright 2018 Elif Sahin
Copyright 2018 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 26})
    matplotlib.pyplot.switch_backend('agg')
    plt.rc('grid', linestyle="-", color='black')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', metavar = 'input_file', type = str, help = 'input file path to plot', required = True)

    args = parser.parse_args()
    train_df = pd.read_csv(args.input)
    cv_df = pd.read_csv(args.input.replace('Train', 'Cv'))
    test_df = pd.read_csv(args.input.replace('Train', 'Test'))
    
    epochs = [i for i in range(0, 20001, 1000)]
    train_df = train_df[train_df['Epoch'].isin(epochs)]
    cv_df = cv_df[cv_df['Epoch'].isin(epochs)]
    test_df = test_df[test_df['Epoch'].isin(epochs)]
    plt.figure(figsize=(30, 15))
    ax = plt.axes()
    major_ticks = np.arange(0, 20000, 1000)
    minor_ticks = np.arange(0, 20000, 500)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.grid(b = True, which = 'both', color = 'black', linestyle=':')
    
    # xticks(np.arange(0, 20000, step=100))
    plt.xlabel("Number of iterations", labelpad = 30)
    plt.ylabel("Loss", labelpad = 30)

    plt.scatter(train_df['Epoch'], train_df['Loss'], s=200, edgecolor="darkorange", facecolor="none", label="Train Loss", alpha=1, marker="^")
    plt.scatter(test_df['Epoch'], test_df['Loss'], s=200, edgecolor="red", facecolor="none", label="Test Loss", alpha=1, marker="^")
    plt.scatter(cv_df['Epoch'], cv_df['Loss'], s=200, edgecolor="green", facecolor="none", label="Cv Loss", alpha=1, marker="^")
    
    plt.legend()
    plt.title("Loss vs Iteration")
    plt.savefig(os.path.join('loss_plots', os.path.splitext(os.path.basename(args.input))[0]))
    plt.close()
    
