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
import copy
import logging
import os
from enum import Enum

import numpy as np
import matplotlib

matplotlib.use('Agg')
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # noqa: E402

# pylint: disable=wrong-import-position
import custom_logger  # noqa: E402


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
    STEPWISE = 7


enum_to_configuration_label = {Technique.LR_RIDGE: 'LRRidge', Technique.XGBOOST: 'XGBoost',
                               Technique.DT: 'DecisionTree', Technique.RF: 'RandomForest',
                               Technique.SVR: 'SVR', Technique.NNLS: 'NNLS',
                               Technique.STEPWISE: 'Stepwise'}


class ExperimentConfiguration(abc.ABC):
    """
    Abstract class representing a single experiment configuration to be performed

    Each experiment configuration describes the building of a single regressor and it is characterized by:
        - technique
        - combination of hyperparameter
        - training set
        - hp selection set
        - validation set

    Each experiment configuration is described univocally by its signature which consists of a list of strings

    Attributes
    ----------
    _campaign_configuration: dict of str: dict of str: str
        The set of options specified by the user though command line and campaign configuration files

    hyperparameters: dict of str: object
        The combination of hyperparameters of this experiment configuration (key is the name of the hyperparameter, value is the value in this configuration)

    _regression_inputs: RegressionInputs
        The input of the regression problem to be solved

    _logger: Logger
        The logger associated with this instance

    _signature: str
        The string signature associated with this experiment configuration

    mapes: dict of str: float
        The MAPE obtained on the different sets

    _experiment_directory: str
        The directory where output of this experiment has to be stored

    _regressor
        The actual object performing the regression; it is intialized by the subclasses and its type depends on the particular technique

    Methods
    -------
    train()
        Build the model starting from training data

    _train()
        Actual internal implementation of the training process

    evaluate()
        Compute the MAPE on the different input dataset

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

    set_training_data()
        Set the training data overwritting current ones

    get_regressor()
        Return the regressor associated with this experiment configuration

    get_technique()
        Return the technique associated with this experiment configuration

    get_hyperparameters()
        Return the values of the hyperparameters associated with this experiment configuration

    get_x_columns()
        Return the columns used in the regression

    print_model()
        Prints in readable form the trained model; at the moment is not pure virtual since not all the subclasses implement it
    """

    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dict of str: object
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved

        prefix: list of str
            The prefix to be used in the computation of the signature of this experiment configuration
        """

        # Initialized attributes
        self._campaign_configuration = campaign_configuration
        self._hyperparameters = hyperparameters
        self._regression_inputs = regression_inputs
        self._signature = self._compute_signature(prefix)
        self._logger = custom_logger.getLogger(self.get_signature_string())
        self.mapes = {}
        self._regressor = None

        # Create experiment directory
        self._experiment_directory = self._campaign_configuration['General']['output']
        for token in self._signature:
            self._experiment_directory = os.path.join(self._experiment_directory, token)
        # Import here to avoid problems with circular dependencies
        # pylint: disable=import-outside-toplevel
        import model_building.hyperopt_sfs_experiment_configuration as hsec
        if ((isinstance(self, hsec.SFSExperimentConfiguration) or isinstance(self, hsec.HyperoptSFSExperimentConfiguration) or 'FeatureSelection' not in self._campaign_configuration or 'method' not in self._campaign_configuration['FeatureSelection'] or self._campaign_configuration['FeatureSelection']['method'] != "SFS")
            and not isinstance(self, hsec.HyperoptExperimentConfiguration)):
            assert not os.path.exists(self._experiment_directory), self._experiment_directory
            os.makedirs(self._experiment_directory)

    def train(self):
        """
        Builid the model with the experiment configuration represented by this object

        This public method wraps the private method which performs the actual work. In doing this it controls the start/stop of logging on file
        """
        self._start_file_logger()
        self._train()
        self._stop_file_logger()

    @abc.abstractmethod
    def _train(self):
        """
        Build the model with the experiment configuration represented by this object

        The actual implementation is demanded to the subclasses
        """

    @abc.abstractmethod
    def initialize_regressor(self):
        """
        The actual implementation is demanded to the subclasses
        """

    def evaluate(self):
        """
        Validate the model, i.e., compute the MAPE on the validation set, hp selection, training
        """
        self._start_file_logger()

        for set_name in ["validation", "hp_selection", "training"]:
            rows = self._regression_inputs.inputs_split[set_name]
            self._logger.debug("Computing MAPE on %s set", set_name)
            predicted_y = self.compute_estimations(rows)
            real_y = self._regression_inputs.data.loc[rows, self._regression_inputs.y_column].values.astype(np.float64)
            if self._regression_inputs.y_column in self._regression_inputs.scalers:
                y_scaler = self._regression_inputs.scalers[self._regression_inputs.y_column]
                predicted_y = y_scaler.inverse_transform(predicted_y)
                real_y = y_scaler.inverse_transform(real_y)
            difference = real_y - predicted_y
            self.mapes[set_name] = np.mean(np.abs(np.divide(difference, real_y)))
            self._logger.debug("Real vs. predicted: %s %s", str(real_y), str(predicted_y))
            self._logger.debug("MAPE on %s set is %f", set_name, self.mapes[set_name])

        self._stop_file_logger()

    def generate_plots(self):
        """
        Generate the plots in the experiment output directory which compare predicted values vs. real ones
        """
        self._start_file_logger()
        if self._campaign_configuration['General']['validation'] in {"Extrapolation", "HoldOut"}:
            training_rows = self._regression_inputs.inputs_split["training"]
            predicted_training_y = self.compute_estimations(training_rows)
            real_training_y = self._regression_inputs.data.loc[training_rows, self._regression_inputs.y_column].values.astype(np.float64)
            if self._regression_inputs.y_column in self._regression_inputs.scalers:
                y_scaler = self._regression_inputs.scalers[self._regression_inputs.y_column]
                predicted_training_y = y_scaler.inverse_transform(predicted_training_y)
                real_training_y = y_scaler.inverse_transform(real_training_y)
            plt.scatter(real_training_y, predicted_training_y, linestyle='None', s=10, marker="*", linewidth=0.5, label="Training", c="green")
        validation_rows = self._regression_inputs.inputs_split["validation"]
        predicted_validation_y = self.compute_estimations(validation_rows)
        real_validation_y = self._regression_inputs.data.loc[validation_rows, self._regression_inputs.y_column].values.astype(np.float64)
        if self._regression_inputs.y_column in self._regression_inputs.scalers:
            y_scaler = self._regression_inputs.scalers[self._regression_inputs.y_column]
            predicted_validation_y = y_scaler.inverse_transform(predicted_validation_y)
            real_validation_y = y_scaler.inverse_transform(real_validation_y)
        plt.scatter(real_validation_y, predicted_validation_y, linestyle='None', s=10, marker="+", linewidth=0.5, label="Validation", c="blue")
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.plot(plt.xlim(), plt.ylim(), "r--", linewidth=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("Real execution times [s]")
        plt.ylabel("Predicted execution times [s]")
        plt.legend()
        plt.savefig(os.path.join(self._experiment_directory, "real_vs_predicted.pdf"))
        plt.savefig(os.path.join(self._experiment_directory, "real_vs_predicted.png"))
        if self._campaign_configuration['General']['validation'] == "Extrapolation" and len(self._campaign_configuration['General']['extrapolation_columns']) == 1:
            plt.figure()
            extrapolation_column = next(iter(self._campaign_configuration['General']['extrapolation_columns']))
            normalization = 'normalization' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['normalization']
            column_label = "original_" + extrapolation_column if normalization else extrapolation_column

            x_training_values = self._regression_inputs.data.loc[self._regression_inputs.inputs_split["training"], column_label]
            x_validation_values = self._regression_inputs.data.loc[self._regression_inputs.inputs_split["validation"], column_label]

            training_error = np.multiply(np.divide(real_training_y - predicted_training_y, real_training_y), 100)
            validation_error = np.multiply(np.divide(real_validation_y - predicted_validation_y, real_validation_y), 100)
            mape = np.mean(np.abs(validation_error))

            plt.scatter(x_training_values, training_error, linestyle='None', s=10, marker="*", linewidth=0.5, label="Training", c="green")
            plt.scatter(x_validation_values, validation_error, linestyle='None', s=10, marker="+", linewidth=0.5, label="Validation", c="blue")
            xlim = plt.xlim()
            ylim = plt.ylim()
            plt.plot(xlim, [0, 0], "r--", linewidth=0.5)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel(extrapolation_column)
            plt.ylabel("Error [%]")
            plt.legend()
            plt.title("Extrapolation on " + extrapolation_column + " - MAPE " + "{0:.2f}".format(mape) + "%")
            plt.savefig(os.path.join(self._experiment_directory, "error_vs_extrapolation.pdf"))
            plt.savefig(os.path.join(self._experiment_directory, "error_vs_extrapolation.png"))

        self._stop_file_logger()

    @abc.abstractmethod
    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration

        Parameters
        ----------
        prefix: list of str
            The prefix which has to be added at the beginning of the signature.
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
        Return
        ------
        list of str
            The signature of this experiment
        """
        return self._signature

    def get_signature_string(self):
        """
        Return
        ------
        str
            The signature of this experiment as string
        """
        return "_".join(self._signature)

    def _start_file_logger(self):
        """
        Add the file handler to the logger to save the log of this experiment on file
        """
        # Logger writes to stdout and file
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
            temp_d['_logger'] = custom_logger.getLogger(temp_d['_logger'])
        self.__dict__.update(temp_d)

    def get_regressor(self):
        """
        Return
        ------
        The regressor wrapped in this experiment configuration
        """
        if not getattr(self, '_regressor', None) or not self._regressor:
          self.initialize_regressor()
        return self._regressor

    def get_hyperparameters(self):
        """
        Return
        ------
        dict of str: object
            The hyperparameters associated with this experiment
        """
        return copy.deepcopy(self._hyperparameters)

    def get_x_columns(self):
        """
        Return
        ------
        list of str:
            the columns used in the regression
        """
        return copy.deepcopy(self._regression_inputs.x_columns)

    def print_model(self):
        """
        Method which prints the representation of the generated model as an empty string when the subclass does not override this method
        """
        return ""

    def set_training_data(self, new_training_data):
        """
        Set the training set of this experiment configuration
        """
        self._regression_inputs = new_training_data
