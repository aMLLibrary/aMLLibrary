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

import itertools
import sys

class Technique(Enum):
    """
    Enum class listing the different refression techniques"
    """
    NONE = 0
    LR_RIDGE = 1
    #TODO: add extra techniques such as XGBoost, SVR, etc.

enum_to_configuration_label = {[Technique.LR_RIDGE, 'Ridge']}
enum_to_param_fields = {[Technique.LR_RIDGE, 'ridge_params']}

class ExpConfsGenerator:
    """
    Abstract class representing a generators of experiment configurations

    Attributes
    ----------
    campaign_configuration: #TODO add type
        The set of options specified by the user though command line and campaign configuration files

    data: dataframe
        The whole set of input data

    Methods
    ------
    generate_experiment_configurations()
        Generates the set of expriment configurations to be evaluated
    """

    _campaign_configuration = None

    _data = None

    _training_idx = []

    _xs = []

    _y = ""

    _seed = 0

    _experiment_configurations = []

    def __init__(self, campaign_configuration, seed):
        """
        Parameters
        ----------
        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities
        """

        #TODO: modify this constructor and all the other constructors which use it to pass the added attributes

        self._campaign_configuration = campaign_configuration

        self._seed = seed

    def generate_experiment_configurations(self):
        """
        Generates the set of experiment configurations to be evaluated

        Returns
        -------
        list
            a list of the experiment configurations
        """
        pass

    def collect_data(self):
        """
        Return the results obtained with the different experiment configurations
        """
        pass

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

    generators = None

    def __init__(self, generators, campaign_configuration, seed):
        """
        Parameters
        ----------
        generators: dict
            The ExpConfsGenerator to be used

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities
        """

        super().__init__(campaign_configuration, seed)


    def generate_experiment_configurations(self):
        """
        Collect the experiment configurations to be evaluated for all the generators

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """

        return_list = []
        for key,generator in generators:
            return_list = return_list.extend(generator.generate_experiment_configurations())
        return return_list

class MultiTechniquesExpConfsGenerator(MultiExpConfsGenerator):
    """
    Specialization of MultiExpConfsGenerator representing a set of experiment configurations related to multiple techniques

    This class wraps the single SingleTechniqueExpConfsGenerator instances which refer to the single techinique

    Methods
    ------
    generate_experiment_configurations()
        Generates the set of expriment configurations to be evaluated
    """

    def __init__(self, technique_generators, campaign_configuration, seed):
        """
        Parameters
        ----------
        technique_igeneratros: dict
            The ExpConfsGenerator to be used for each technique

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files
        """

        super().__init__(technique_generators, campaign_configuration, seed)


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

    def __init__(self, campaign_configuration, technique, seed):
        """
        Parameters
        ----------
        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            the seed to be used in the generation (ignored in this class)
        """

        super().__init__(campaign_configuration, seed)
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
        second_key = ""
        #We expect that hyperparameters for a technique are stored in campaign_configuration[first_key][second_key] as a dictionary from string to list of values
        #TODO: build the grid search on the basis of the configuration and return the set of built points
        if self._technique == Technique.NONE:
            logging.error("Not supported regression technique")
            sys.exit(-1)

        first_key = enum_to_configuration_label[self._technique]
        second_key = enum_to_param_fields[self._technique]
        hyperparams = campaign_configuration[first_key][second_key]
        hyperparams_names = []
        hyperparams_values = []
        for hyperparam in hyperparams:
            hyperparams_names.append(hyperparam)
            hyperparams_values.append(hyperparams[hyperparam])

        #Cartesian product of parameters
        for combination in itertools.product(*hyperparams_values):
            hyperparams_point_values = {}
            for hyperparams_name, hyperparams_value in hyperparams_names, combination:
                hyperparams_point_values[hyperparams_name] = hyperparams_value
            if self._technique == Technique.LR_RIDGE:
                point = lr_ridge_experiment_configuration.LRRidgeExperimentConfiguration(hyperparams_point_values, self._data, self._training_idx, self._xs, self._y)
            else:
                logging.error("Not supported regression technique")
                point = None
                sys.exit(-1)
            self._experiment_configurations.append(point)

        return self._experiment_configurations


class RepeatedExpConfsGenerator(MultiExpConfsGenerator):
    """
    Invokes n times the wrapped ExpConfsGenerator with n different seeds

    Attributes
    ----------
    n: integer
        The number of different seeds to be used

    wrapped_generator: ExpConfsGenerator
        The wrapped generator to be invoked

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    n = 0

    n_generators = None

    def __init__(self, n, wrapped_generator, campaign_configuration):
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
        super().__init__(self, n_generators, campaign_configuration, seed)

        self.n = n
        #TODO generate the n generators passing different seeds

class KFoldExpConfsGenerator(MultiExpConfsGenerator):
    """
    Wraps k instances of a generator with different training set

    Attributes
    ----------
    k: integer
        The number of different folds to be used

    wrapped_generators: ExpConfsGenerator
        The wrapped generators to be invoked

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    k = 0

    kfold_generators = None

    def __init__(self, k, wrapped_generator, campaign_configuration):
        """
        Parameters
        ----------
        k: integer
            The number of folds to be considered

        wrapped_generator: ExpConfsGenerator
            The wrapped generator to be duplicated and modified

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files
        """
        super().__init__(self, kfold_generators, campaign_configuration, seed)

        self.k = k
        #TODO generate the k generators with different training set

class RandomExpConfsGenerator(ExpConfsGenerator):
    """
    Wraps an experiment configuration generator randomly picking n experiment configurations

    Attributes
    ----------
    n: integer
        The number of experiment configurations to be returned

    wrapped_generator: ExpConfsGenerator
        The wrapped generator to be used

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    n = 0

    def __init__(self, n, wrapped_generator, campaign_configuration):
        """
        Parameters
        ----------
        n: integer
            The number of experiment configurations to be returned

        wrapped_generator: ExpConfsGenerator
            The wrapped generator

        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files
        """
        super().__init__(self, campaign_configuration, seed)

        self.n = n
        #TODO  call wrapped generator and randomly pick n experiment configurations
