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

import sklearn.linear_model as lr

import model_building.experiment_configuration as ec


class LRRidgeExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for linear regression

    Attributes
    ----------
    _regressor : LinearRegression
        The actual scikt object which performs the linear regression

    Methods
    -------
    _train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the estimated values for a give set of data

    print_model()
        Prints the model
    """
    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix: List[str]):
        """
        campaign_configuration: dict of dict:
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dictionary
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved
        """
        assert prefix
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.LR_RIDGE
        self._regressor = lr.Ridge()

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration
        """
        assert isinstance(prefix, list)
        signature = prefix.copy()
        signature.append("alpha_" + str(self._hyperparameters['alpha']))
        return signature

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        self._regressor = lr.Ridge(alpha=self._hyperparameters['alpha'])
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._regressor.fit(xdata, ydata)
        self._logger.debug("Model built")
        for idx, col_name in enumerate(self._regression_inputs.x_columns):
            self._logger.debug("The coefficient for %s is %f", col_name, self._regressor.coef_[idx])

    def compute_estimations(self, rows):
        """
        Compute the estimations and the MAPE for runs in rows
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        return self._regressor.predict(xdata)

    def print_model(self):
        """
        Print the model
        """
        ret_string = ""
        coefficients = self._regressor.coef_
        for column, coefficient in zip(self._regression_inputs.x_columns, coefficients):
            if ret_string != "":
                ret_string = ret_string + " + "
            ret_string = ret_string + "(" + str(coefficient) + "*" + column + ")"
        ret_string = ret_string + " + (" + str(self._regressor.intercept_) + ")"
        return ret_string
