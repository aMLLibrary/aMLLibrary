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

import sklearn.linear_model as lr
import model_building.experiment_configuration as ec

class LRRidgeExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for linear regression

    Attributes
    ----------
    _linear_regression : LinearRegression
        The actual scikt object which performs the linear regression

    Methods
    -------
    train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the estimated values for a give set of data
    """
    _linear_regression = lr.Ridge()

    def compute_signature(self):
        """
        Compute the signature associated with this experiment configuration
        """
        return "LRRidge_alpha_" + str(self._hyperparameters['alpha'])

    def train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self.compute_signature())
        self._linear_regression = lr.Ridge(alpha=self._hyperparameters['alpha'])
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.training_idx)
        self._linear_regression.fit(xdata, ydata)
        self._logger.debug("Model built")
        for idx, col_name in enumerate(self._regression_inputs.x_columns):
            self._logger.debug("The coefficient for %s is %f", col_name, self._linear_regression.coef_[idx])

    def compute_estimations(self, rows):
        """
        Compute the estimations and the MAPE for runs in rows
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        self._linear_regression.predict(xdata)
