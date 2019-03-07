#!/usr/bin/env python3
"""
Copyright 2019 Marco Lattuada

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

import abc
import logging
import os
import pandas as pd

class RegressionInputs:
    """
    Data structure storing inputs information for a regression problem

    Attributes
    ----------
    data: dataframe
        The whole dataframe

    training_idx: list of integers
        The indices of the rows of the data frame to be used to train the model

    x_columns: list of strings
        The labels of the columns of the data frame to be used to train the model

    y_column: string
        The label of the y column
    """
    data = pd.DataFrame()

    training_idx = []

    x_columns = []

    y_column = ""

    def __init__(self, data, training_idx, x_columns, y_column):
        """
        Parameters
        data: dataframe
            The whole dataframe

        training_idx: list of integers
            The indices of the rows of the data frame to be used to train the model

        x_columns: list of strings
            The labels of the columns of the data frame to be used to train the model

        y_column: string
            The label of the y column
        """
        self.data = data
        self.training_idx = training_idx
        self.x_columns = x_columns
        self.y_column = y_column

class ExperimentConfiguration(abc.ABC):
    """
    Abstract class representing a single experiment configuration to be performed

    Attributes
    ----------
    _campaign_configuration: dict of dict
        The set of options specified by the user though command line and campaign configuration files

    _hyperparameters: dictionary
        The set of hyperparameters of this experiment configuration

    _regression_inputs: RegressionInputs
        The input of the regression problem to be solved

    _local_folder: Path
        The path where all the results related to this experiment configuration will be stored

    _logger: Logger
        The logger associated with this class and its descendents

    Methods
    -------
    _prepare_data()
        Generates the two pandas data frame with x_columns and y
    """

    _campaign_configuration = {}

    _hyperparameters = None

    _regression_inputs = None

    _local_folder = ""

    _logger = None

    def __init__(self, campaign_configuration, hyperparameters, regression_inputs):
        """
        campaign_configuration: dict of dict:
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dictionary
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved
        """

        self._campaign_configuration = campaign_configuration
        self._hyperparameters = hyperparameters
        self._regression_inputs = regression_inputs
        self._logger = logging.getLogger(__name__)

        signature = self.compute_signature()
        os.mkdir(os.path.join(self._campaign_configuration['General']['output'], signature))

    @abc.abstractmethod
    def train(self):
        """
        Build the model with the experiment configuration represented by this object
        """

    @abc.abstractmethod
    def compute_signature(self):
        """
        Compute the signature associated with this experiment configuration
        """

    def _prepare_data(self):
        """
        Generate the x and y pandas dataframes containing only the necessary information

        Returns
        -------
        df,df
            The data frame containing the x_columns column and the data frame containing the y column
        """
        xdata = self._regression_inputs.data.loc[self._regression_inputs.training_idx, self._regression_inputs.x_columns]
        ydata = self._regression_inputs.data.loc[self._regression_inputs.training_idx, self._regression_inputs.y_column]
        return xdata, ydata
