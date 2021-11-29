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

import sklearn.ensemble as rf

import model_building.experiment_configuration as ec


class RandomForestExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for random forest

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the linear model

    initialize_regressor()
        Initialize the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values
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
            The prefix to be added to the signature of this experiment configuration
        """
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.RF

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
        signature.append("n_estimators_" + str(self._hyperparameters['n_estimators']))
        signature.append("criterion_" + str(self._hyperparameters['criterion']))
        signature.append("max_depth_" + str(self._hyperparameters['max_depth']))
        signature.append("max_features_" + str(self._hyperparameters['max_features']))
        signature.append("min_samples_split_" + str(self._hyperparameters['min_samples_split']))
        signature.append("min_samples_leaf_" + str(self._hyperparameters['min_samples_leaf']))

        return signature

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._regressor.fit(xdata, ydata)

    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments
        """
        if not getattr(self, '_hyperparameters', None):
            self._regressor = rf.RandomForestRegressor()
        else:
            self._regressor = rf.RandomForestRegressor(
                n_estimators=self._hyperparameters['n_estimators'],
                criterion=self._hyperparameters['criterion'],
                max_depth=self._hyperparameters['max_depth'],
                max_features=self._hyperparameters['max_features'],
                min_samples_split=self._hyperparameters['min_samples_split'],
                min_samples_leaf=self._hyperparameters['min_samples_leaf'])

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return {'n_estimators': 5,
                'criterion': 'mse',
                'max_depth': 3,
                'max_features': 'auto',
                'min_samples_split': 0.1,
                'min_samples_leaf': 1}
