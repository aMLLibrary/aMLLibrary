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
import logging
import random

import model_building.generators_factory as gf
import results as re

class ModelBuilding:
    """
    Entry point of the model building phase, i.e., where all the regressions are actually performed

    Attributes
    ----------
    random_generator : Random
        The internal random generator

    _logger: Logger
        The logger used by this class

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
        self._logger = logging.getLogger(__name__)

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
        factory = gf.GeneratorsFactory(campaign_configuration, self._random_generator.random())
        top_generator = factory.build()
        expconfs = top_generator.generate_experiment_configurations([], regression_inputs)

        assert expconfs
        for exp in expconfs:
            exp.train()

        results = re.Results(expconfs)
        results.collect_data()

        for metric, mapes in results.raw_results.items():
            for experiment_configuration, mape in mapes.items():
                self._logger.info("%s of %s is %f", metric, experiment_configuration, mape)

        best_results = results.get_best_for_technique()
        for technique, best_result in best_results.items():
            self._logger.info("MAPE of best %s is for configuration %s: %f", technique, best_result[0], best_result[1])
