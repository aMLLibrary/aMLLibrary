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
from enum import Enum


import numpy as np

class Technique(Enum):
    """
    Enum class listing the different regression techniques"
    """
    NONE = 0
    LR_RIDGE = 1
    XGBOOST = 2
    DT = 3
    #TODO: add extra techniques such as  SVR, etc.

enum_to_configuration_label = {Technique.LR_RIDGE: 'LRRidge', Technique.XGBOOST: 'XGBoost', Technique.DT: 'DecisionTree'}


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

    _signature: str
        The signature associated with this experiment configuration

    validation_mape: float
        The MAPE obtained on the validation data

    Methods
    -------
    train()
        Build the model starting from training data

    validate()
        Compute the MAPE on the validation set

    _compute_signature()
        Compute the string identifier of this experiment

    compute_estimations()
        Compute the estimated values for a give set of DataAnalysis

    get_signature()
        Return the signature of this experiment
    """

    _campaign_configuration = {}

    _hyperparameters = None

    _regression_inputs = None

    _local_folder = ""

    _logger = None

    _signature = None

    validation_mape = 0.0

    technique = Technique.NONE

    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix):
        """
        campaign_configuration: dict of dict:
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dictionary
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved
        """

        #Initialized attributes
        self._campaign_configuration = campaign_configuration
        self._hyperparameters = hyperparameters
        self._regression_inputs = regression_inputs
        self._signature = self._compute_signature(prefix)
        self._logger = logging.getLogger("_".join(self._signature))

        #Create experiment directory
        experiment_directory = self._campaign_configuration['General']['output']
        for token in self._signature:
            experiment_directory = os.path.join(experiment_directory, token)
        assert not os.path.exists(experiment_directory)
        os.makedirs(experiment_directory)

        #Logger writes to stdout and file
        file_handler = logging.FileHandler(os.path.join(experiment_directory, 'log'))
        self._logger.addHandler(file_handler)

    @abc.abstractmethod
    def train(self):
        """
        Build the model with the experiment configuration represented by this object
        """

    def validate(self):
        """
        Validate the model, i.e., compute the MAPE on the validation set
        """
        validation_rows = self._regression_inputs.validation_idx
        self._logger.debug("Validating model")
        predicted_y = self.compute_estimations(validation_rows)
        real_y = self._regression_inputs.data.loc[validation_rows, self._regression_inputs.y_column].values.astype(np.float64)
        if self._regression_inputs.y_column in self._regression_inputs.scalers:
            y_scaler = self._regression_inputs.scalers[self._regression_inputs.y_column]
            predicted_y = y_scaler.inverse_transform(predicted_y)
            real_y = y_scaler.inverse_transform(real_y)
        difference = real_y - predicted_y
        self.validation_mape = np.mean(np.abs(np.divide(difference, real_y)))
        self._logger.debug("Validated model. MAPE is %f", self.validation_mape)

    @abc.abstractmethod
    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration
        """

    @abc.abstractmethod
    def compute_estimations(self, rows):
        """
        Compute the estimations and the MAPE for runs in rows

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered
        """

    def get_signature(self):
        """
        Return the signature of this experiment
        """
        return "_".join(self._signature)
