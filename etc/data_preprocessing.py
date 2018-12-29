
import pandas as pd
import numpy as np
import math
import ast
import os
import logging
import configparser as cp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action = 'ignore', category = DataConversionWarning)

# Change str to float
def is_column_line(str):
    try:
        float(str)
        return False
    except ValueError:
        return True

    def read_csv_data(self):
        data = pd.read_csv(file_path)

    def add_invers_features(data):
        data_matrix = pd.DataFrame.as_matrix(data)
        features_names = list(data.columns.values)
        features_dict = dict(data)
        features_num = data_matrix.shape[1]
        new_features_names = []
        new_features_dict = {}
        for i in range(features_num):
            feature_name = features_names[i]
            print(feature_name)
            new_features_names.append(feature_name)
            new_features_dict[feature_name] = list(features_dict.get(feature_name))
            invers_feature_name = 'inverse_'+feature_name
            new_features_names.append(invers_feature_name)
            try:
                new_features_dict[invers_feature_name] = list(1/np.array(new_features_dict[feature_name]))
            except ValueError:
                new_features_dict[invers_feature_name] = list(np.zeros(data_matrix.shape[0]))
        return new_features_dict
