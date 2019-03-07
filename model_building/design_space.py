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

from enum import Enum

import abc
import itertools
import logging
import pprint
import random
import sys

import model_building.lr_ridge_experiment_configuration

class Technique(Enum):
    """
    Enum class listing the different refression techniques"
    """
    NONE = 0
    LR_RIDGE = 1
    #TODO: add extra techniques such as XGBoost, SVR, etc.

enum_to_configuration_label = {Technique.LR_RIDGE: 'LRRidge'}

class ExpConfsGenerator(abc.ABC):
    """
    Abstract class representing a generators of experiment configurations

    Attributes
    ----------
    _campaign_configuration: dict of dict
        The set of options specified by the user though command line and campaign configuration files

    _regression_inputs: RegressionInputs
        The input data of the regression problem

    _random_generator: RandomGenerator
        The random generator used to generate random numbers

    _experiment_configurations: list of ExperimentConfiguration
        The list of ExperimentConfiguration created

    _logger: Logger
        The logger associated with this class and its descendents

    Methods
    ------
    generate_experiment_configurations()
        Generates the set of expriment configurations to be evaluated
    """

    _campaign_configuration = {}

    _regression_inputs = None

    _random_generator = random.Random(0)

    _experiment_configurations = []

    _logger = None

    def __init__(self, campaign_configuration, regression_inputs, seed):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem

        seed: integer
            The seed to be used to initialize the random generator
        """

        #TODO: modify this constructor and all the other constructors which use it to pass the added attributes

        self._campaign_configuration = campaign_configuration
        self._regression_inputs = regression_inputs
        self._random_generator = random.Random(seed)
        self._logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def generate_experiment_configurations(self):
        """
        Generates the set of experiment configurations to be evaluated

        Returns
        -------
        list
            a list of the experiment configurations
        """

    def collect_data(self):
        """
        Return the results obtained with the different experiment configurations
        """

class MultiExpConfsGenerator(ExpConfsGenerator):
    """
    Specialization of MultiExpConfsGenerator which wraps multiple generators

    Attributes
    ----------
    generators: dict
        The ExpConfsGenerator to be used

    Methods
    ------
    generate_experiment_configurations()
        Generates the set of expriment configurations to be evaluated
    """

    _generators = []

    def __init__(self, campaign_configuration, regression_inputs, seed, generators):
        """
        Parameters
        ----------
        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem

        seed: integer
            The seed to be used in random based activities

        generators: dict
            The ExpConfsGenerator to be used
        """

        super().__init__(campaign_configuration, regression_inputs, seed)
        self._generators = generators


    def generate_experiment_configurations(self):
        """
        Collect the experiment configurations to be evaluated for all the generators

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """

        self._logger.debug("Calling generate_experiment_configurations in %s", self.__class__.__name__)
        return_list = []
        assert self._generators
        for generator in self._generators:
            return_list.extend(generator.generate_experiment_configurations())
        assert return_list
        return return_list

class MultiTechniquesExpConfsGenerator(MultiExpConfsGenerator):
    """
    Specialization of MultiExpConfsGenerator representing a set of experiment configurations related to multiple techniques

    This class wraps the single TechniqueExpConfsGenerator instances which refer to the single techinique

    Methods
    ------
    generate_experiment_configurations()
        Generates the set of expriment configurations to be evaluated
    """

class TechniqueExpConfsGenerator(ExpConfsGenerator):
    """
    Class which generalize classes for generate points to be explored for each technique
    #TODO: check if there is any technique which actually needs to extends this

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    _technique = Technique.NONE

    def __init__(self, campaign_configuration, regression_inputs, seed, technique):
        """
        Parameters
        ----------
        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem

        seed: integer
            the seed to be used in the generation (ignored in this class)
        """

        super().__init__(campaign_configuration, regression_inputs, seed)
        self._technique = technique

    def generate_experiment_configurations(self):
        """
        Collected the set of points to be evaluated for the single technique

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """

        first_key = ""
        #We expect that hyperparameters for a technique are stored in campaign_configuration[first_key] as a dictionary from string to list of values
        #TODO: build the grid search on the basis of the configuration and return the set of built points
        if self._technique == Technique.NONE:
            self._logger.error("Not supported regression technique")
            sys.exit(-1)

        first_key = enum_to_configuration_label[self._technique]
        hyperparams = self._campaign_configuration[first_key]
        self._logger.debug("Hyperparams are %s", pprint.pformat(hyperparams, width=1))
        hyperparams_names = []
        hyperparams_values = []
        #Adding dummy hyperparameter to have at least two hyperparameters
        hyperparams_names.append('dummy')
        hyperparams_values.append([0])

        logging.debug("Computing set of hyperparameters combinations")
        for hyperparam in hyperparams:
            hyperparams_names.append(hyperparam)
            hyperparams_values.append(hyperparams[hyperparam])

        #Cartesian product of parameters
        for combination in itertools.product(*hyperparams_values):
            hyperparams_point_values = {}
            for hyperparams_name, hyperparams_value in zip(hyperparams_names, combination):
                hyperparams_point_values[hyperparams_name] = hyperparams_value
            if self._technique == Technique.LR_RIDGE:
                point = model_building.lr_ridge_experiment_configuration.LRRidgeExperimentConfiguration(self._campaign_configuration, hyperparams_point_values, self._regression_inputs)
            else:
                self._logger.error("Not supported regression technique")
                point = None
                sys.exit(-1)
            self._experiment_configurations.append(point)

        assert self._experiment_configurations

        return self._experiment_configurations


class RepeatedExpConfsGenerator(MultiExpConfsGenerator):
    """
    Invokes n times the wrapped ExpConfsGenerator with n different seeds

    Attributes
    ----------
    _repetitions_number: integer
        The number of different seeds to be used

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    _repetitions_number = 0

    def __init__(self, repetitions_number, wrapped_generator, campaign_configuration):
        """
        Parameters
        ----------
        n: integer
            The number of different seeds to be used

        wrapped_generator: ExpConfsGenerator
            The wrapped generator to be invoked

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files
        """

        self._repetitions_number = repetitions_number
        #TODO generate the n generators passing different seeds

        wrapped_generators = []

        super().__init__(self, wrapped_generators, campaign_configuration, self._random_generator.random())

class KFoldExpConfsGenerator(MultiExpConfsGenerator):
    """
    Wraps k instances of a generator with different training set

    Attributes
    ----------
    k: integer
        The number of different folds to be used

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    k = 0

    def __init__(self, k, wrapped_generator, campaign_configuration, seed):
        """
        Parameters
        ----------
        k: integer
            The number of folds to be considered

        wrapped_generator: ExpConfsGenerator
            The wrapped generator to be duplicated and modified

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities
        """

        #TODO generate the k generators with different training set
        kfold_generators = []
        super().__init__(self, kfold_generators, campaign_configuration, seed)

        self.k = k

class RandomExpConfsGenerator(ExpConfsGenerator):
    """
    Wraps an experiment configuration generator randomly picking n experiment configurations

    Attributes
    ----------
    _experiment_configurations_number: integer
        The number of experiment configurations to be returned

    _wrapped_generator: ExpConfsGenerator
        The wrapped generator to be used

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    _experiment_configurations_number = 0

    _wrapped_generator = None

    def __init__(self, experiment_configurations_number, wrapped_generator, campaign_configuration, seed):
        """
        Parameters
        ----------
        n: integer
            The number of experiment configurations to be returned

        wrapped_generator: ExpConfsGenerator
            The wrapped generator

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities
        """
        super().__init__(self, campaign_configuration, seed)

        self._experiment_configurations_number = experiment_configurations_number
        self._wrapped_generator = wrapped_generator

    def generate_experiment_configurations(self):
        """
        Collected the set of points to be evaluated for the single technique

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """
        #TODO  call wrapped generator and randomly pick n experiment configurations
