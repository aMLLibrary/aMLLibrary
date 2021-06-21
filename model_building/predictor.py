from configparser import ConfigParser
import pandas as pd
import pickle
import numpy as np

import data_preparation.data_loading
import data_preparation.onehot_encoding

class Predictor:
  def __init__(self, config_file, regressor_file, output_file):
    # Initialize strings
    self._config_file = config_file
    self._regressor_file = regressor_file
    self._output_file = output_file
    print(f"Parameters: {self._config_file}, {self._regressor_file}, and "
          f"{self._output_file}")

    # Read data
    config = ConfigParser()
    config.read(config_file)
    self._data = pd.read_csv(config['DataPreparation']['input_path'])
    self._xx = self._data.drop(columns=[config['General']['y']])
    self._yy = self._data[config['General']['y']]

    # Read regressor
    with open(regressor_file, "rb") as f:
      self._regressor = pickle.load(f)

  def predict(self):
    self._yy_predicted = self._regressor.predict(self._xx)
    yy_both = pd.DataFrame()
    yy_both['real'] = self._yy
    yy_both['pred'] = self._yy_predicted
    print(yy_both)
    mape = np.mean(np.abs(np.divide(self._yy-self._yy_predicted, self._yy)))
    print("MAPE =", str(mape))
