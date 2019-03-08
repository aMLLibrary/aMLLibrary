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
    train()
        Build the model starting from training data

    validate()
        Compute the MAPE on the validation set

    compute_signature()
        Compute the string identifier of this experiment

    compute_estimations()
        Compute the estimated values for a give set of data
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

        #Initialized attributes
        self._campaign_configuration = campaign_configuration
        self._hyperparameters = hyperparameters
        self._regression_inputs = regression_inputs
        self._logger = logging.getLogger(__name__)

        #Create experiment directory
        signature = self.compute_signature()
        experiment_directory = os.path.join(self._campaign_configuration['General']['output'], signature)
        assert not os.path.exists(experiment_directory)
        os.mkdir(experiment_directory)

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
        self._logger.debug("Validating model")
        self.compute_estimations(self._regression_inputs.validation_idx)
        self._logger.debug("Validated model")

    @abc.abstractmethod
    def compute_signature(self):
        """
        Compute the signature associated with this experiment configuration
        """

    @abc.abstractmethod
    def compute_estimations(self, rows):
        """
        Compute the estimations and the MAPE for runs in rows
        """
