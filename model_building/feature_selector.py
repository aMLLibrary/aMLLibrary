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

#TODO: Define feature selector enum

import abc

import model_building.design_space as mb

class FSExpConfsGenerator(mb.ExpConfsGenerator):
    """
    Abstract superclass for feature selector methods which exploits regressors

    Attributes
    ----------
    wrapped_generator: ExpConfsGenerator
        The ExpConfsGenerator used with the feature selector

    Methods
    ------
    generate_experiment_configurations()
        Generates the set of expriment configurations to be evaluated
    """

    _wrapped_generator = None

    def __init__(self, wrapped_generator, campaign_configuration, seed):
        """
        Parameters
        ----------
        wrapped_generator: ExpConfsGenerator
            The ExpConfsGenerator to be used coupled with the feature selector

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities
        """

        super().__init__(campaign_configuration, seed)
        self._wrapped_generator = wrapped_generator

    @abc.abstractmethod
    def generate_experiment_configurations(self):
        """
        Generates the set of experiment configurations to be evaluated

        Returns
        -------
        list
            a list of the experiment configurations
        """
