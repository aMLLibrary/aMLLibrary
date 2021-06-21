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


class Predictor:
    def __init__(self, config_file, regressor_file, output_folder):
        logging.basicConfig(level=logging.INFO)
        self._logger = custom_logger.getLogger(__name__)
        self._output_folder = output_folder
        self._regressor_file = regressor_file

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
        self._datb = data_loader.process(None)
        self._logger.info("Current data frame is:\n%s", str(self._datb))
        self._logger.info("<--")
        self._datb = self._datb.data
        self._logger.info("Current data frame is:\n%s", str(self._datb))
        self._logger.info("<--")


    def load_campaign_configuration(self):
        self._campaign_configuration = {}

        for section in self.conf.sections():
            self._campaign_configuration[section] = {}
            for item in self.conf.items(section):
                try:
                    self._campaign_configuration[section][item[0]] = ast.literal_eval(item[1])
                except (ValueError, SyntaxError):
                    self._campaign_configuration[section][item[0]] = item[1]

        self._logger.info("Parameters configuration is:")
        self._logger.info("-->")
        self._logger.info(pprint.pformat(self._campaign_configuration, width=1))
        self._logger.info("<--")


    def predict(self):
        self._logger.info("-->Executing prediction")
        yy = self._datb[self._campaign_configuration['General']['y']]
        self._datb = self._datb.drop(columns=[self._campaign_configuration['General']['y']])
        with open(self._regressor_file, "rb") as f:
            regressor = pickle.load(f)
        yy_pred = regressor.predict(self._datb)

        # Write predictions to file
        yy_both = pd.DataFrame()
        yy_both['real'] = yy
        yy_both['pred'] = yy_pred
        yy_file = os.path.join(self._output_folder, 'prediction.csv')
        with open(yy_file, 'w') as f:
            yy_both.to_csv(f, index=False)
        self._logger.info("Saved to %s", str(yy_file))

        # Compute MAPE
        difference = yy - yy_pred
        mape = np.mean(np.abs(np.divide(difference, yy)))
        self._logger.info("---MAPE = %s", str(mape))
        mape_file = os.path.join(self._output_folder, 'mape.txt')
        with open(mape_file, 'w') as f:
          f.write(str(mape))
          f.write('\n')

        self._logger.info("<--Performed prediction")
