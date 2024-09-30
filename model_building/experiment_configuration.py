"""
Copyright 2019 Marco Lattuada
Copyright 2021 Bruno Guindani
Copyright 2022 Nahuel Coliva

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
import pickle
import os
from enum import Enum
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import regressor

# pylint: disable=wrong-import-position
import custom_logger  # noqa: E402


class Technique(Enum):
    """
    Enum class listing the different regression techniques
    """
    NONE = 0
    LR_RIDGE = 1
    XGBOOST = 2
    DT = 3
    RF = 4
    SVR = 5
    NNLS = 6
    STEPWISE = 7
    DUMMY = 8
    NEURAL_NETWORK = 9


enum_to_configuration_label = {Technique.LR_RIDGE: 'LRRidge', Technique.XGBOOST: 'XGBoost',
                               Technique.DT: 'DecisionTree', Technique.RF: 'RandomForest',
                               Technique.SVR: 'SVR', Technique.NNLS: 'NNLS',
                               Technique.STEPWISE: 'Stepwise', Technique.DUMMY: 'Dummy',
                               Technique.NEURAL_NETWORK: 'NeuralNetwork'}


def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = np.finfo(np.float64).eps
    if len(y_true.shape) == 1:
        y_true = y_true.values.reshape(y_true.shape[0],1)
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(mape, axis=0)


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

    rmses: dict of str: float
        The Root Mean Squared Errors (RMSE) obtained on the different sets

    r2s: dict of str: float
        The R^2 scores obtained on the different sets

    _experiment_directory: str
        The directory where output of this experiment has to be stored

    _regressor
        The actual object performing the regression; it is intialized by the subclasses and its type depends on the particular technique

    _hyperparameters: dict of str: object
        The hyperparameter values for the technique regressor

    _disable_model_parallelism: bool
        Signals whether each XGBoost model parallelism is disabled (True, thus disabled, when parallel training of models is enabled from
        the configuration file, False otherwise)

    Methods
    -------
    train()
        Build the model starting from training data

    _train()
        Actual internal implementation of the training process

    initialize_regressor()
        Initializes the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values

    evaluate()
        Compute the MAPE on the different input dataset

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

    set_regressor()
        Set the regressor associated with this experiment configuration

    get_technique()
        Return the technique associated with this experiment configuration

    get_hyperparameters()
        Return the values of the hyperparameters associated with this experiment configuration

    repair_hyperparameters()
        Repair hyperparameter values which cause the regressor to raise errors

    get_x_columns()
        Return the columns used in the regression

    set_x_columns()
        Set the columns to be used in the regression

    print_model()
        Print the representation of the generated model

    set_training_data()
        Set the training data overwriting current ones

    is_wrapper()
        Return whether this object has a wrapped experiment configuration
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
        self._regression_inputs = regression_inputs.copy()
        self._signature = self._compute_signature(prefix)
        self._logger = custom_logger.getLogger(self.get_signature_string())
        self.mapes = {}
        self.rmses = {}
        self.r2s = {}
        self._regressor = None
        self.trained = False

        # Create experiment directory
        self._experiment_directory = self._campaign_configuration['General']['output']
        for token in self._signature:
            self._experiment_directory = os.path.join(self._experiment_directory, token)
        # Import here to avoid problems with circular dependencies
        # pylint: disable=import-outside-toplevel
        import model_building.wrapper_experiment_configuration as wec
        if (    not isinstance(self, wec.SFSExperimentConfiguration)
            and not isinstance(self, wec.HyperoptExperimentConfiguration)
            and not isinstance(self, wec.HyperoptSFSExperimentConfiguration)
           ):
            # This is not a wrapper of another experiment: create experiment directory
            assert self._experiment_directory
            if not os.path.exists(self._experiment_directory):
                os.makedirs(self._experiment_directory)

    def train(self, force=False, disable_model_parallelism=False):
        """
        Build the model with the experiment configuration represented by this object

        This public method wraps the private method which performs the actual work. In doing this it controls the start/stop of logging on file

        Parameters
        ----------
        force: bool
            Force training even if Pickle regressor file is present

        disable_model_parallelism: bool
            Signals whether each XGBoost model parallelism is disabled (True, thus disabled, when parallel training of models is enabled from
            the configuration file, False otherwise)
        """
        self._disable_model_parallelism = disable_model_parallelism
        if self.is_wrapper():
            self._wrapped_experiment_configuration._disable_model_parallelism = disable_model_parallelism
        
        regressor_path = os.path.join(self._experiment_directory, 'regressor.pickle')

        # Fault tolerance mechanism for interrupted runs
        if os.path.exists(regressor_path):
            try:
                with open(regressor_path, 'rb') as f:
                    regressor_obj = pickle.load(f)
                if force: #re-training the model requires keeping the same hyperparameters previously found
                    self._hyperparameters = regressor_obj.get_hypers()
                    if self.is_wrapper():
                        self._wrapped_experiment_configuration._hyperparameters = self._hyperparameters
                else:
                    self.set_regressor(regressor_obj.get_regressor())
                    self.set_x_columns(regressor_obj.get_x_columns())
                    self._hyperparameters = regressor_obj.get_hypers()
                    if self.is_wrapper():
                        self._wrapped_experiment_configuration._hyperparameters = self._hyperparameters
                    self.trained = True
                    if hasattr(self, '_sfs_trained'):
                        self._sfs_trained = True
                    if hasattr(self, '_hyperopt_trained'):
                        self._hyperopt_trained = True
                    return
            except EOFError:
                # Run was interrupted in the middle of writing the regressor to file: we restart the experiment
                pass
        self._start_file_logger()
        self.initialize_regressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._train()
        self.trained = True
        self._stop_file_logger()

        trained_regressor = regressor.Regressor(self._campaign_configuration,self.get_regressor(),self.get_x_columns(),None,self.get_hyperparameters())
        with open(regressor_path, 'wb') as f:
            pickle.dump(trained_regressor, f)

    @abc.abstractmethod
    def _train(self):
        """
        Build the model with the experiment configuration represented by this object

        The actual implementation is demanded to the subclasses
        """

    @abc.abstractmethod
    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments

        The actual implementation is demanded to the subclasses
        """

    @abc.abstractmethod
    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values

        The actual implementation is demanded to the subclasses
        """

    def evaluate(self):
        """
        Validate the model, i.e., compute the MAPE and other metrics on the validation set, hp selection, training

        Values are stored in the appropriate class members
        """
        self._start_file_logger()

        self._logger.debug("Computing metrics for %s", str(self.get_signature()))
        for set_name in ["validation", "hp_selection", "training"]:
            rows = self._regression_inputs.inputs_split[set_name]
            self._logger.debug("On %s set:", set_name)
            self._logger.debug("-->")
            predicted_y = self.compute_estimations(rows).reshape(-1,1)
            real_y = self._regression_inputs.data.loc[rows, self._regression_inputs.y_column].values.astype(np.float64).reshape(-1,1)
            if self._regression_inputs.y_column in self._regression_inputs.scalers:
                y_scaler = self._regression_inputs.scalers[self._regression_inputs.y_column]
                predicted_y = y_scaler.inverse_transform(predicted_y)
                real_y = y_scaler.inverse_transform(real_y)
            # self._logger.debug("Real vs. predicted: %s %s", str(real_y), str(predicted_y))
            # Mean Absolute Percentage Error
            self.mapes[set_name] = mean_absolute_percentage_error(real_y, predicted_y)
            self._logger.debug("MAPE is %f", self.mapes[set_name])
            # Root Mean Squared Error
            self.rmses[set_name] = mean_squared_error(real_y, predicted_y, squared=False)
            self._logger.debug("RMSE is %f", self.rmses[set_name])
            # R-squared metric
            self.r2s[set_name] = r2_score(real_y, predicted_y)
            self._logger.debug("R^2  is %f", self.r2s[set_name])
            self._logger.debug("<--")

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

    def compute_estimations(self, rows):
        """
        Compute the predictions for data points indicated in rows estimated by the regressor

        The actual implementation is demanded to the subclasses

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        return self._regressor.predict(xdata)

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
        if self._regressor is None:
            self.initialize_regressor()
        return self._regressor

    def set_regressor(self, reg):
        """
        Set the regressor associated with this experiment configuration

        Parameters
        ----------
        reg: regressor object
            the regressor to be set in this experiment configuration
        """
        self._regressor = reg

    def get_hyperparameters(self):
        """
        Return
        ------
        dict of str: object
            The hyperparameters associated with this experiment
        """
        return copy.deepcopy(self._hyperparameters)

    def repair_hyperparameters(self, hypers):
        """
        Repair and return hyperparameter values which cause the regressor to raise errors

        Parameters
        ----------
        hypers: dict of str: object
            the hyperparameters to be repaired

        Return
        ------
        dict of str: object
            the repaired hyperparameters
        """
        return copy.deepcopy(hypers)

    def get_x_columns(self):
        """
        Return the columns used in the regression

        Return
        ------
        list of str:
            the columns used in the regression
        """
        return copy.deepcopy(self._regression_inputs.x_columns)

    def set_x_columns(self, x_cols):
        """
        Set the columns to be used in the regression

        Parameters
        ----------
        x_cols: list of str
            the columns to be used in the regression
        """
        self._regression_inputs.x_columns = x_cols

    def print_model(self):
        """
        Print the representation of the generated model, or just the model name if not overridden by the subclass

        This is not a pure virtual method since not all the subclasses implement it
        """
        return "(" + enum_to_configuration_label[self.technique] + ")"

    def set_training_data(self, new_training_data):
        """
        Set the training set of this experiment configuration
        """
        self._regression_inputs = new_training_data

    def is_wrapper(self):
        return hasattr(self, '_wrapped_experiment_configuration')
