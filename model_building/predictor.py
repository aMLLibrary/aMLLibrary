"""
Copyright 2021 Bruno Guindani

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

import configparser as cp
import logging
import os
import pickle
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

import custom_logger
import sequence_data_processing
import data_preparation.data_loading
import data_preparation.onehot_encoding
import model_building.model_building


class Predictor(sequence_data_processing.SequenceDataProcessing):
    """
    Class that uses Pickle objects to make predictions on new datasets
    """
    def __init__(self, regressor_file, output_folder, debug):
        """
        Constructor of the class

        Parameters
        ----------
        regressor_file: str
            Pickle binary file that stores the model to be used for prediction

        output_folder: str
            The directory where all the outputs will be written; it is created by this library and cannot exist before using this module

        debug: bool
            True if debug messages should be printed
        """
        # Set verbosity level and initialize logger
        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        self._logger = custom_logger.getLogger(__name__)

        # Initialize flags
        self._output_folder = output_folder

        # Read regressor
        with open(regressor_file, "rb") as f:
            self._regressor = pickle.load(f)

    def predict(self, config_file, mape_to_file):
        """
        Performs prediction and computes MAPE

        Parameters
        ----------
        config_file: str
            The configuration file describing the experimental campaign to be performed

        mape_to_file: bool
            True if computed MAPE should be written to a text file (file name is mape.txt)
        """
        # Read configuration from the file indicated by the argument
        if not os.path.exists(config_file):
            self._logger.error("%s does not exist", config_file)
            sys.exit(-1)
        # Check if output path already exist
        if os.path.exists(self._output_folder):
            self._logger.error("%s already exists. Terminating the program...", self._output_folder)
            sys.exit(1)
        os.mkdir(self._output_folder)

        # Read config file
        self.load_campaign_configuration(config_file)

        # Read data
        self._logger.info("-->Executing data load")
        data_loader = data_preparation.data_loading.DataLoading(self._campaign_configuration)
        self.data = data_loader.process(None)
        self.data = self.data.data
        self._logger.debug("Current data frame is:\n%s", str(self.data))
        self._logger.info("<--")

        # Start prediction
        self._logger.info("-->Performing prediction")
        yy = self.data[self._campaign_configuration['General']['y']]
        xx = self.data.drop(columns=[self._campaign_configuration['General']['y']])
        yy_pred = self._regressor.predict(xx)

        # Write predictions to file
        yy_both = pd.DataFrame()
        yy_both['real'] = yy
        yy_both['pred'] = yy_pred
        self._logger.debug("Parameters configuration is:")
        self._logger.debug("-->")
        self._logger.debug("Current data frame is:\n%s", str(yy_both))
        self._logger.debug("<--")
        yy_file = os.path.join(self._output_folder, 'prediction.csv')
        with open(yy_file, 'w') as f:
            yy_both.to_csv(f, index=False)
        self._logger.info("Saved to %s", str(yy_file))

        # Compute and output MAPE
        mape = mean_absolute_percentage_error(yy, yy_pred)
        self._logger.info("---MAPE = %s", str(mape))
        if mape_to_file:
          mape_file = os.path.join(self._output_folder, 'mape.txt')
          with open(mape_file, 'w') as f:
            f.write(str(mape))
            f.write('\n')
          self._logger.info("Saved MAPE to %s", str(mape_file))

        self._logger.info("<--Performed prediction")


    def predict_from_df(self, xx):
        """
        Performs prediction on a dataframe

        Parameters
        ----------
        xx: pandas.DataFrame
            The covariate matrix to be used for prediction

        Returns
        -------
        yy_pred
            The predicted values for the dependent variable
        """
        self._logger.info("-->Performing prediction on dataframe")
        yy_pred = self._regressor.predict(xx)
        self._logger.info("Predicted values are: %s", str(yy_pred))
        self._logger.info("<--Performed prediction")
        return yy_pred
