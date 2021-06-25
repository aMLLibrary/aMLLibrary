"""
Copyright 2021 Bruno Guindani
"""
import ast
import configparser as cp
import logging
import os
import pickle
import pprint
import sys
import numpy as np
import pandas as pd

import custom_logger
import data_preparation.data_loading
import data_preparation.onehot_encoding
import model_building.model_building


class Predictor(SequenceDataProcessing):
    """
    Class that uses Pickle objects to make predictions on new datasets
    """
    def __init__(self, config_file, regressor_file, output_folder, debug, mape_to_text):
        """
        Constructor of the class

        Parameters
        ----------
        config_file: str
            The configuration file describing the experimental campaign to be performed

        regressor_file: str
            Pickle binary file that stores the model to be used for prediction

        output_folder: str
            The directory where all the outputs will be written; it is created by this library and cannot exist before using this module

        debug: bool
            True if debug messages should be printed

        mape_to_text: bool
            True if computed MAPEs should be written to a text file (file name is mape.txt)
        """
        # Set verbosity level and initialize logger
        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        self._logger = custom_logger.getLogger(__name__)

        self._output_folder = output_folder
        self._regressor_file = regressor_file
        self._mape_to_text = mape_to_text

        # Read config file
        self.conf = cp.ConfigParser()
        self.conf.optionxform = str
        self.conf.read(config_file)
        self._campaign_configuration = {}
        self.load_campaign_configuration()

        # Check if output path already exist
        if os.path.exists(output_folder):
            self._logger.error("%s already exists", output_folder)
            sys.exit(1)
        os.mkdir(output_folder)

        # Read data
        self._logger.info("-->Executing data load")
        data_loader = data_preparation.data_loading.DataLoading(self._campaign_configuration)
        self.data = data_loader.process(None)
        self.data = self.data.data
        self._logger.debug("Current data frame is:\n%s", str(self.data))
        self._logger.info("<--")


    def predict(self):
        """
        Performs prediction and computes MAPE on it
        """
        self._logger.info("-->Performing prediction")
        yy = self.data[self._campaign_configuration['General']['y']]
        xx = self.data.drop(columns=[self._campaign_configuration['General']['y']])
        with open(self._regressor_file, "rb") as f:
            regressor = pickle.load(f)
        yy_pred = regressor.predict(xx)

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
        mape = self.compute_mape(self, yy, yy_pred)
        self._logger.info("---MAPE = %s", str(mape))
        if self._mape_to_text:
          mape_file = os.path.join(self._output_folder, 'mape.txt')
          with open(mape_file, 'w') as f:
            f.write(str(mape))
            f.write('\n')
          self._logger.info("Saved MAPE to %s", str(mape_file))

        self._logger.info("<--Performed prediction")
