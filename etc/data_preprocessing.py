
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


