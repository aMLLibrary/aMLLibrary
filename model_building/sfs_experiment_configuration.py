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

from typing import List

import mlxtend.feature_selection
import numpy as np
import sklearn
import sys

import model_building.experiment_configuration


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute the MAPE

    Parameters
    ----------
    y_true: numpy.array
        The real values

    y_pred: numpy.array
        The predicted value

    Return
    ------
    float
        The computed MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class SFSExperimentConfiguration(model_building.experiment_configuration.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for SFS coupled with a generic regression

    Attributes
    ----------
    _wrapped_experiment_configuration : ExperimentConfiguration
        The regressor to be used in conjunction with sequential feature selection

    _sfs: SequentialFeatureSelector
        The actual sequential feature selector implemented by mlxtend library

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the estimated values for a give set of data

    print_model()
        Prints the model
    """
    def __init__(self, campaign_configuration, regression_inputs, prefix: List[str], wrapped_experiment_configuration):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved

        prefix: list of str
            The information used to identify this experiment

        wrapped_experiment_configuration: ExperimentConfiguration
            The regressor to be used in conjunction with sequential feature selection

        """
        self._wrapped_experiment_configuration = wrapped_experiment_configuration
        super().__init__(campaign_configuration, None, regression_inputs, prefix)
        verbose = 2 if self._campaign_configuration['General']['debug'] else 0
        temp_xdata, temp_ydata =  self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # if the maximum number of required features is greater than the number of existing features, exit
        if self._campaign_configuration['FeatureSelection']['max_features'] > temp_xdata.shape[1]:
            self._logger.error("ERROR: The maximum number of required features must be in range(1, %d)", temp_xdata.shape[1]+1)
            sys.exit(-10)
        self._sfs = mlxtend.feature_selection.SequentialFeatureSelector(estimator=self._wrapped_experiment_configuration.get_regressor(), k_features=(1, self._campaign_configuration['FeatureSelection']['max_features']), verbose=verbose, scoring=sklearn.metrics.make_scorer(mean_absolute_percentage_error, greater_is_better=False), cv=self._campaign_configuration['FeatureSelection']['folds'])
        self.technique = self._wrapped_experiment_configuration.technique

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration

        Parameters
        ----------
        prefix: list of str
            The signature of this experiment configuration without considering hyperparameters

        Returns
        -------
            The signature of the experiment
        """
        return self._wrapped_experiment_configuration.get_signature()

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # set the maximum number of required features to the minimum between itself and the number of existing features
        if self._campaign_configuration['FeatureSelection']['max_features'] > xdata.shape[1]:
            self._logger.info("Reduced maximum number of features from %d to %d", self._sfs.k_features[1], xdata.shape[1])
            self._sfs.k_features = (1,xdata.shape[1])
        self._sfs.fit(xdata, ydata)
        self._logger.debug("Selected features: %s", str(self._sfs.k_feature_names_))

        # Use the selected feature to retrain the regressor
        filtered_xdata = self._sfs.transform(xdata)
        self._regressor = self._wrapped_experiment_configuration.get_regressor()
        self._wrapped_experiment_configuration.get_regressor().fit(filtered_xdata, ydata)
        self._regression_inputs.x_columns = list(self._sfs.k_feature_names_)

    def compute_estimations(self, rows):
        """
        Compute the estimations and the MAPE for runs in rows

        Parameters
        ----------
        rows: list of integer
            The list of the input data to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, ydata = self._regression_inputs.get_xy_data(rows)
        ret = self._wrapped_experiment_configuration.get_regressor().predict(xdata)
        self._logger.debug("Using regressor on %s: %s vs %s", str(xdata), str(ydata), str(ret))

        return ret

    def print_model(self):
        """
        Print the model
        """
        return self._wrapped_experiment_configuration.print_model()
