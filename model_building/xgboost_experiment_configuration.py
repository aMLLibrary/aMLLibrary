"""
Copyright 2019 Marco Lattuada
Copyright 2019 Danilo Ardagna

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

#import sklearn.linear_model as lr

import xgboost as xgb

import model_building.experiment_configuration as ec


class XGBoostExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for linear regression

    Attributes
    ----------
    _linear_regression : LinearRegression
        The actual scikt object which performs the linear regression

    Methods
    -------
    _train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the estimated values for a give set of data
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
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.XGBOOST
        self._regressor = xgb.XGBRegressor()


    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration
        """
        signature = prefix.copy()
        signature.append("min_child_weight_" + str(self._hyperparameters['min_child_weight']))
        signature.append("gamma_" + str(self._hyperparameters['gamma']))
        signature.append("n_estimators_" + str(self._hyperparameters['n_estimators']))
        signature.append("learning_rate_" + str(self._hyperparameters['learning_rate']))
        signature.append("max_depth_" + str(self._hyperparameters['max_depth']))


        return signature

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        self._regressor = xgb.XGBRegressor(min_child_weight=self._hyperparameters['min_child_weight'],
                                           gamma=self._hyperparameters['gamma'],
                                           n_estimators=self._hyperparameters['n_estimators'],
                                           learning_rate=self._hyperparameters['learning_rate'],
                                           max_depth=self._hyperparameters['max_depth'])
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.training_idx)
        self._regressor.fit(xdata, ydata)
        self._logger.debug("Model built")

        #for idx, col_name in enumerate(self._regression_inputs.x_columns):
        #    self._logger.debug("The coefficient for %s is %f", col_name, self._linear_regression.coef_[idx])

    def compute_estimations(self, rows):
        """
        Compute the estimations and the MAPE for runs in rows
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        return self._regressor.predict(xdata)
