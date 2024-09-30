"""
Main module of the library

Copyright 2019 Marjan Hosseini
Copyright 2019 Marco Lattuada
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

This module defines the SequenceDataProcessing class which is the only class that has to be accessed to generate regressors
"""
import ast
import configparser as cp
import logging
import os
import pandas as pd
import pprint
import random
import shutil
import sys
import time

import custom_logger
import data_preparation.column_selection
import data_preparation.data_check
import data_preparation.data_loading
import data_preparation.ernest
import data_preparation.extrapolation
import data_preparation.interpolation
import data_preparation.inversion
import data_preparation.logarithm
import data_preparation.onehot_encoding
import data_preparation.product
import data_preparation.rename_columns
import data_preparation.row_selection
import data_preparation.xgboost_feature_selection

import model_building.model_building


class SequenceDataProcessing:
    """
    Main class which performs the whole design space exploration and builds the regressors

    Its main method is process which performs three main steps:
    1. generate the set of points (i.e., combination of training data, technique, hyper-parameters) to be evaluated
    2. build the regressor corresponding to each point
    3. evaluate the results of all the regressors to identify the best one

    Attributes
    ----------
    _data_preprocessing_list: list of DataPreparation
        The list of steps to be executed for data preparation

    _model_building: ModelBuilding
        The object which performs the actual model building

    _random_generator: RandomGenerator
        The random generator used in the whole application both to generate random numbers and to initialize other random generators
    """

    def __init__(self, input_configuration, debug=False, output="output", j=1, details=False, keep_temp=False):
        """
        Constructor of the class

        - Copy the parameters to member variables
        - Initialize the logger
        - Build the data preparation flow adding or not data preparation steps on the basis of the content of the loaded configuration file

        Parameters
        ----------
        input_configuration: str or dict
            The configuration file describing the experimental campaign to be performed, or a dictionary with the same structure

        debug: bool
            True if debug messages should be printed

        output: str
            The directory where all the outputs will be written; it is created by this library and cannot exist before using this module

        j: integer
            The number of processes to be used in the grid search

        details: bool
            True if the results of the single experiments should be added

        keep_temp: bool
            True if temporary files should not be deleted at the end of a successful run
        """
        self._done_file_flag = os.path.join(output, 'done')

        self._data_preprocessing_list = []
        self.debug = debug
        self._keep_temp = keep_temp

        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        self._logger = custom_logger.getLogger(__name__)

        # Read campaign configuration
        if isinstance(input_configuration, str):
            # Read configuration from the file indicated by the argument
            self.input_configuration_file = input_configuration
            if not os.path.exists(input_configuration):
                self._logger.error("%s does not exist", input_configuration)
                sys.exit(-1)
            general_args = {'configuration_file': input_configuration, 'output': output,
                            'j': str(j), 'debug': str(debug), 'details': str(details)
                           }
            self.load_campaign_configuration(input_configuration, general_args)
        elif isinstance(input_configuration, dict):
            # Read configuration from the argument dict
            self.input_configuration_file = None
            self._campaign_configuration = input_configuration
            general_args = {'output': output, 'j': j, 'debug': debug, 'details': details}
            self._campaign_configuration['General'].update(general_args)
            self.complete_campaign_configuration()
        else:
            self._logger.error("input_configuration must be a path string to a configuration file or a dictionary")
            sys.exit(1)

        self._logger.debug("Parameters configuration is:")
        self._logger.debug("-->")
        self._logger.debug(pprint.pformat(self._campaign_configuration, width=1))
        self._logger.debug("<--")

        # Initialize random number generator
        self.random_generator = random.Random(self._campaign_configuration['General']['seed'])

        # Check if output path already exist
        if os.path.exists(output) and os.path.exists(self._done_file_flag):
            self._logger.error("%s already exists. Terminating the program...", output)
            sys.exit(1)
        if not os.path.exists(output):
            os.mkdir(self._campaign_configuration['General']['output'])
        if isinstance(input_configuration, str):
            shutil.copyfile(input_configuration, os.path.join(output, 'configuration.ini'))
        confpars = cp.ConfigParser()
        confpars.read_dict(self._campaign_configuration)
        with open(os.path.join(output, 'configuration_enriched.ini'), 'w') as conf_enriched:
            confpars.write(conf_enriched)


        # Check that validation method has been specified
        if 'validation' not in self._campaign_configuration['General']:
            self._logger.error("Validation not specified")
            sys.exit(1)

        # Check that if HoldOut is selected, hold_out_ratio is specified
        if self._campaign_configuration['General']['validation'] == "HoldOut" or self._campaign_configuration['General']['hp_selection'] == "HoldOut":
            if "hold_out_ratio" not in self._campaign_configuration['General']:
                self._logger.error("hold_out_ratio not set")
                sys.exit(1)

        # Check that if Extrapolation is selected, extrapolation_columns is specified
        if self._campaign_configuration['General']['validation'] == "Extrapolation":
            if "extrapolation_columns" not in self._campaign_configuration['General']:
                self._logger.error("extrapolation_columns not set")
                sys.exit(1)

        # Check that if Interpolation is selected, interpolation_columns is specified
        if self._campaign_configuration['General']['validation'] == "Interpolation":
            if "interpolation_columns" not in self._campaign_configuration['General']:
                self._logger.error("interpolation_columns not set")
                sys.exit(1)

        # Check that if XGBoost is used for feature selection tolerance is specified
        if 'FeatureSelection' in self._campaign_configuration and self._campaign_configuration['FeatureSelection']['method'] == "XGBoost":
            if "XGBoost_tolerance" not in self._campaign_configuration['FeatureSelection']:
                self._logger.error("XGBoost tolerance not set")
                sys.exit(1)

        # Check that if ernest is used, normalization, product, column_selection, and inversion are disabled
        if 'ernest' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['ernest']:
            if 'use_columns' in self._campaign_configuration['DataPreparation'] or "skip_columns" in self._campaign_configuration['DataPreparation']:
                self._logger.error("use_columns and skip_columns cannot be used with ernest")
                sys.exit(1)
            if 'inverse' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['inverse']:
                self._logger.error("inverse cannot be used with ernest")
                sys.exit(1)
            if 'product_max_degree' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['product_max_degree']:
                self._logger.error("product cannot be used with ernest")
                sys.exit(1)
            if 'normalization' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['normalization']:
                self._logger.error("normalization cannot be used with ernest")
                sys.exit(1)

        # Adding read on input to data preprocessing step
        self._data_preprocessing_list.append(data_preparation.data_loading.DataLoading(self._campaign_configuration))

        # Adding column renaming if required
        if 'rename_columns' in self._campaign_configuration['DataPreparation']:
            self._data_preprocessing_list.append(data_preparation.rename_columns.RenameColumns(self._campaign_configuration))

        # Adding row selection if required
        if 'skip_rows' in self._campaign_configuration['DataPreparation']:
            self._data_preprocessing_list.append(data_preparation.row_selection.RowSelection(self._campaign_configuration))

        # Adding column selection if required
        if 'use_columns' in self._campaign_configuration['DataPreparation'] or 'skip_columns' in self._campaign_configuration['DataPreparation']:
            self._data_preprocessing_list.append(data_preparation.column_selection.ColumnSelection(self._campaign_configuration))

        # Transform categorical features in onehot encoding
        self._data_preprocessing_list.append(data_preparation.onehot_encoding.OnehotEncoding(self._campaign_configuration))

        # Split according to extrapolation values if required
        if self._campaign_configuration['General']['validation'] == "Extrapolation":
            self._data_preprocessing_list.append(data_preparation.extrapolation.Extrapolation(self._campaign_configuration))

        # Split according to interpolation values if required
        if self._campaign_configuration['General']['validation'] == "Interpolation":
            self._data_preprocessing_list.append(data_preparation.interpolation.Interpolation(self._campaign_configuration))

        # Adding inverted features if required
        if 'inverse' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['inverse']:
            self._data_preprocessing_list.append(data_preparation.inversion.Inversion(self._campaign_configuration))

        # Adding logarithm computation if required
        if 'log' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['log']:
            self._data_preprocessing_list.append(data_preparation.logarithm.Logarithm(self._campaign_configuration))

        # Adding product features if required
        if (('product_max_degree' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['product_max_degree'])
            or 'selected_products' in self._campaign_configuration['DataPreparation']):
            self._data_preprocessing_list.append(data_preparation.product.Product(self._campaign_configuration))

        # Create ernest features if required
        if 'ernest' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['ernest']:
            self._data_preprocessing_list.append(data_preparation.ernest.Ernest(self._campaign_configuration))

        # Adding data check
        self._data_preprocessing_list.append(data_preparation.data_check.DataCheck(self._campaign_configuration))

        self._model_building = model_building.model_building.ModelBuilding(self.random_generator.random())

    def load_campaign_configuration(self, configuration_file, general_args={}):
        """
        Load the campaign configuration from configuration_file to the member dictionary, self._campaign_configuration

        Parameters
        ----------
        configuration_file: str
            The configuration file describing the experimental campaign to be performed
        general_args: dict of str: str
            Arguments to add to the "General" section of the campaign configuration
        """

        confpars = cp.ConfigParser()
        confpars.optionxform = str
        confpars.read(configuration_file)

        for key, val in general_args.items():
            confpars['General'][key] = val

        self._campaign_configuration = {}
        for section in confpars.sections():
            self._campaign_configuration[section] = {}
            for item in confpars.items(section):
                try:
                    self._campaign_configuration[section][item[0]] = ast.literal_eval(item[1])
                except (ValueError, SyntaxError):
                    self._campaign_configuration[section][item[0]] = item[1]

        self.complete_campaign_configuration()

    def complete_campaign_configuration(self):
        """
        Update the campaign configuration with missing features, if any
        """
        if 'run_num' not in self._campaign_configuration['General']:
            self._campaign_configuration['General']['run_num'] = 1
        if 'seed' not in self._campaign_configuration['General']:
            self._campaign_configuration['General']['seed'] = 0

    def process(self):
        """
        the main code which actually performs the design space exploration of models

        Only a single regressor is returned: the best model of the best technique.

        These are the main steps:
        - data are preprocessed and dumped to data_preprocessed.csv
        - design space exploration of the required models (i.e., the models specified in the configuration file) is performed
        - eventually, best model is used to predict all the data
        - best model is returned

        Returns
        -------
        Regressor
            The regressor containing the overall best model and the preprocessing steps used to preprocess the input data
        """

        os.environ["OMP_NUM_THREADS"] = "1"

        start = time.time()

        self._logger.info("-->Starting experimental campaign")
        # performs reading data, drops irrelevant columns
        # initial_df = self.preliminary_data_processing.process(self._campaign_configuration)
        # logging.info("Loaded and cleaned data")

        # performs inverting of the columns and adds combinatorial terms to the df
        # ext_df = self.data_preprocessing.process(initial_df, self._campaign_configuration)
        # logging.info("Preprocessed data")

        data_processing = None

        for data_preprocessing_step in self._data_preprocessing_list:
            self._logger.info("-->Executing %s", data_preprocessing_step.get_name())
            data_processing = data_preprocessing_step.process(data_processing)
            self._logger.debug("Current data frame is:\n%s", str(data_processing))
            self._logger.info("<--")

        data = self._campaign_configuration['DataPreparation']['input_path']
        if isinstance(data, str):
            shutil.copyfile(data, os.path.join(self._campaign_configuration['General']['output'], 'data.csv'))
        elif isinstance(data, pd.DataFrame):
            data.to_csv(os.path.join(self._campaign_configuration['General']['output'], 'data.csv'))
        else:
            self._logger.error("input_path must be a path string to a dataset or a pandas.DataFrame")
            sys.exit(1)

        data_processing.data.to_csv(os.path.join(self._campaign_configuration['General']['output'], 'data_preprocessed.csv'))

        regressor = self._model_building.process(self._campaign_configuration, data_processing, int(self._campaign_configuration['General']['j']))

        end = time.time()
        execution_time = str(end - start)
        file_handler = logging.FileHandler(os.path.join(self._campaign_configuration['General']['output'], 'results.txt'), 'a+')
        self._logger.addHandler(file_handler)
        self._logger.info("<--Execution Time : %s", execution_time)

        # Create success flag file
        with open(self._done_file_flag, 'wb') as f:
            pass

        # Delete temporary files if required
        if not self._keep_temp:
            run_num = self._campaign_configuration['General']['run_num']
            for i in range(run_num):
                folder_name = os.path.join(self._campaign_configuration['General']['output'], 'run_' + str(i))
                self._logger.debug("Removing temporary folder %s...", folder_name)
                shutil.rmtree(folder_name)
                self._logger.debug("Done")

        #Close logging
        self._logger.removeHandler(file_handler)
        file_handler.close()

        return regressor
