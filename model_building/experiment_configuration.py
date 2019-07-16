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

import matplotlib
matplotlib.use('Agg')
#pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

import numpy as np

class Technique(Enum):
    """
    Enum class listing the different regression techniques"
    """
    NONE = 0
    LR_RIDGE = 1
    XGBOOST = 2
    DT = 3
    RF = 4
    SVR = 5
    NNLS = 6

enum_to_configuration_label = {Technique.LR_RIDGE: 'LRRidge', Technique.XGBOOST: 'XGBoost', Technique.DT: 'DecisionTree',
                               Technique.RF: 'RandomForest', Technique.SVR: 'SVR', Technique.NNLS: 'NNLS'}


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

    _experiment_directory: str
        The directory where output of this experiment has to be stored

    _regressor
        The actual object performing the regression

    Methods
    -------
    train()
        Build the model starting from training data

    _train()
        Actual implementation of train

    validate()
        Compute the MAPE on the validation set

    generate_plots()
        Generate plots about real vs. predicted

    _compute_signature()
        Compute the string identifier of this experiment

    compute_estimations()
        Compute the estimated values for a give set of DataAnalysis

    get_signature()
        Return the signature of this experiment

    get_signature_string()
        Return the signature of this experiment as string

    _start_file_logger()
        Start to log also to output file

    stop_file_logger()
        Stop to log also to output file

    get_regressor()
        Return the regressor associated with this experiment configuration

    get_technique()
        Return the technique associated with this experiment configuration
    """

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
        self._logger = logging.getLogger(self.get_signature_string())
        self._logger.debug("---")
        self.validation_mape = None
        self._regressor = None

        #Create experiment directory
        self._experiment_directory = self._campaign_configuration['General']['output']
        for token in self._signature:
            self._experiment_directory = os.path.join(self._experiment_directory, token)
        #Import here to avoid problems with circular dependencies
        import model_building.sfs_experiment_configuration
        if isinstance(self, model_building.sfs_experiment_configuration.SFSExperimentConfiguration) or 'FeatureSelection' not in self._campaign_configuration:
            assert not os.path.exists(self._experiment_directory)
            os.makedirs(self._experiment_directory)


    def train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._start_file_logger()
        self._train()
        self._stop_file_logger()

    @abc.abstractmethod
    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """

    def validate(self):
        """
        Validate the model, i.e., compute the MAPE on the validation set
        """
        self._start_file_logger()
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
        self._logger.debug("Real vs. predicted: %s %s", str(real_y), str(predicted_y))
        self._logger.debug("Validated model. MAPE is %f", self.validation_mape)
        self._stop_file_logger()

    def generate_plots(self):
        self._start_file_logger()
        if self._campaign_configuration['General']['validation'] == "HoldOut":
            training_rows = self._regression_inputs.training_idx
            predicted_y = self.compute_estimations(training_rows)
            real_y = self._regression_inputs.data.loc[training_rows, self._regression_inputs.y_column].values.astype(np.float64)
            if self._regression_inputs.y_column in self._regression_inputs.scalers:
                y_scaler = self._regression_inputs.scalers[self._regression_inputs.y_column]
                predicted_y = y_scaler.inverse_transform(predicted_y)
                real_y = y_scaler.inverse_transform(real_y)
            plt.scatter(real_y, predicted_y, linestyle='None', s=10, marker="*", linewidth=0.5, label="Training", c="green")
        validation_rows = self._regression_inputs.validation_idx
        predicted_y = self.compute_estimations(validation_rows)
        real_y = self._regression_inputs.data.loc[validation_rows, self._regression_inputs.y_column].values.astype(np.float64)
        if self._regression_inputs.y_column in self._regression_inputs.scalers:
            y_scaler = self._regression_inputs.scalers[self._regression_inputs.y_column]
            predicted_y = y_scaler.inverse_transform(predicted_y)
            real_y = y_scaler.inverse_transform(real_y)
        plt.scatter(real_y, predicted_y, linestyle='None', s=10, marker="+", linewidth=0.5, label="Validation", c="blue")
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.plot(plt.xlim(), plt.ylim(), "r--", linewidth=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("Real execution times [s]")
        plt.ylabel("Predicted execution times [s]")
        plt.legend()
        plt.savefig(os.path.join(self._experiment_directory, "real_vs_predicted.pdf"))
        self._stop_file_logger()

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
        return self._signature

    def get_signature_string(self):
        """
        Return the signature of this experiment as string
        """
        return "_".join(self._signature)

    def _start_file_logger(self):
        """
        Add the file handler to the logger
        """
        #Logger writes to stdout and file
        file_handler = logging.FileHandler(os.path.join(self._experiment_directory, 'log'), 'a+')
        self._logger.addHandler(file_handler)

    def _stop_file_logger(self):
        """
        Remove the file handler from the logger
        """
        handlers = self._logger.handlers[:]
        for handler in handlers:
            handler.close()
            self._logger.removeHandler(handler)

    def __getstate__(self):
        """
        Auxilixiary function used by pickle. Ovverriden to avoid problems with logger lock
        """
        temp_d = self.__dict__.copy()
        if '_logger' in temp_d:
            temp_d['_logger'] = temp_d['_logger'].name
        return temp_d

    def __setstate__(self, temp_d):
        """
        Auxilixiary function used by pickle. Ovverriden to avoid problems with logger lock
        """
        if '_logger' in temp_d:
            temp_d['_logger'] = logging.getLogger(temp_d['_logger'])
        self.__dict__.update(temp_d)

    def get_regressor(self):
        """
        Return the regressor wrapped in this experiment configuration
        """
        return self._regressor
