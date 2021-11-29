"""
Copyright 2019 Marco Lattuada
Copyright 2019 Danilo Ardagna
Copyright 2021 Bruno Guindani

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
import copy
import operator
import os
import warnings

import xgboost as xgb

import model_building.experiment_configuration as ec


class XGBoostExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for XGBoost

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the predictions for data points indicated in rows estimated by the regressor

    print_model()
        Print the representation of the generated model

    initialize_regressor()
        Initialize the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values

    repair_hyperparameters()
        Repair and return hyperparameter values which cause the regressor to raise errors

    get_weights_dict()
        Return a dictionary containing the regressor's normalized importance weights for each feature
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
        self._logger.debug("---Building model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._regressor.fit(xdata, ydata)
        self._logger.debug("---Model built")
        for key, val in self.get_weights_dict().items():
            self._logger.debug("The weight for %s is %f", key, val)

    def compute_estimations(self, rows):
        """
        Compute the predictions for data points indicated in rows estimated by the regressor

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        self._regressor.set_params(nthread=1)
        self._regressor._Booster.set_param('nthread', 1)
        return self._regressor.predict(xdata)

    def print_model(self):
        """
        Print the representation of the generated model
        """
        sorted_weights = dict(sorted(self.get_weights_dict().items(), key=operator.itemgetter(1), reverse=True))
        ret = "XGBoost weights: {\n"
        for key, val in sorted_weights.items():
            ret += "    " + str(round(val, 3)) + " " + key + "\n"
        ret += "}"
        return ret

    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments
        """
        if not getattr(self, '_hyperparameters', None):
            self._regressor = xgb.XGBRegressor()
        else:
            self._regressor = xgb.XGBRegressor(min_child_weight=self._hyperparameters['min_child_weight'], gamma=self._hyperparameters['gamma'], n_estimators=self._hyperparameters['n_estimators'], learning_rate=self._hyperparameters['learning_rate'], max_depth=self._hyperparameters['max_depth'], tree_method="hist", objective='reg:squarederror', n_jobs=1)

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return {'learning_rate': 0.1,
                'max_depth': 100,
                'gamma': 0.25,
                'min_child_weight': 1,
                'n_estimators': 500}

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
        new_hypers = copy.deepcopy(hypers)
        for key in ['max_depth', 'n_estimators']:
            new_hypers[key] = int(new_hypers[key])
        return new_hypers

    def get_weights_dict(self):
        """
        Return a dictionary containing the regressor's normalized importance weights for each feature
        """
        weights = self._regressor.get_booster().get_fscore()
        weights_sum = sum(weights.values())
        for key in weights:
            weights[key] /= weights_sum
        return weights
