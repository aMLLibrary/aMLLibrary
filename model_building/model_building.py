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
import random

import model_building.generators_factory as gf

class ModelBuilding:
    """
    Entry point of the model building phase, i.e., where all the regressions are actually performed

    Attributes
    ----------
    random_generator : Random
        The internal random generator

    Methods
    ------
    process()
        Generates the set of expriment configurations to be evaluated
    """

    _random_generator = random.Random(0)

    def __init__(self, seed):
        """
        Parameters
        ----------
        seed: float
            The seed to be used for the internal random generator
        """
        self._random_generator = random.Random(seed)

    def process(self, campaign_configuration, regression_inputs):
        """
        Perform the actual regression

        Parameters
        ----------
        campaign_configuration: dictionary
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem
        """
        factory = gf.GeneratorsFactory(campaign_configuration, regression_inputs, self._random_generator.random())
        top_generator = factory.build()
        expconf = top_generator.generate_experiment_configurations()

        assert expconf
        for exp in expconf:
            exp.train()
            exp.evaluate()
