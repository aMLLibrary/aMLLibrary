"""
Copyright 2019 Eugenio Gianniti

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

import model_building.experiment_configuration as ec
import model_building.stepwisefit as sw


class StepwiseExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for stepwise + linear regression

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the estimated values for a give set of data
    """

    def __init__(self, campaign_configuration, hyper_parameters, input_data, prefix):
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
        super().__init__(campaign_configuration, hyper_parameters, input_data, prefix)
        self.technique = ec.Technique.STEPWISE
        possible_flags = ["p_enter", "p_remove", "max_iter", "fit_intercept"]
        hp_flags = {
            label: self._hyperparameters[label]
            for label in possible_flags
            if label in self._hyperparameters
        }
        self._regressor = sw.Stepwise(**hp_flags)

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        assert self._regression_inputs
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._regressor.fit(xdata, ydata)
        self._logger.debug("Model built")
        for beta, col_name in zip(self._regressor.coef_, self._regressor.k_feature_names_):
            self._logger.debug("The coefficient for %s is %f", col_name, beta)

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
        assert isinstance(prefix, list)
        signature = prefix.copy()
        possible_flags = ["p_enter", "p_remove", "max_iter", "fit_intercept"]
        hp_flags = {
            label: self._hyperparameters[label]
            for label in possible_flags
            if label in self._hyperparameters
        }
        signature.extend(f"{name}_{value}" for name, value in hp_flags.items())
        return signature

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
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        return self._regressor.predict(xdata)
