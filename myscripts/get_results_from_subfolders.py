#!/usr/bin/env python3
import os
import pandas as pd
import configparser
import numpy as np

# Allows running this script from both this folder and from root folder
if os.getcwd() == os.path.dirname(__file__):
  os.chdir(os.pardir)

print(os.getcwd())
RESULTS_ROOT_FOLDER = os.path.join('..', 'aml_outputs', 'output_coliva')

for subb in os.listdir(RESULTS_ROOT_FOLDER):
    subfolder = os.path.join(RESULTS_ROOT_FOLDER, subb)
    print("\n\n>>>", subfolder)
    for expp in os.listdir(subfolder):
        experiment = os.path.join(subfolder, expp)
        print("\n>", experiment)
        results_file_path = os.path.join(experiment, 'results')
        with open(results_file_path, 'r') as f:
            for ll in f.readlines():
                if 'Validation MAPE is ' in ll:
                    line = ll.split(" - ")
                    print(line[0], "-", line[4], end="")
                elif 'Overall best result is ' in ll:
                    line = ll.split('[')[0].replace(' is Technique.', ': ')
                    print(line)
