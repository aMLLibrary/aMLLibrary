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

import abc
import copy
import itertools
import logging
import pprint
import random
import sys

import data_preparation.normalization
import data_preparation.random_splitting

import model_building.experiment_configuration as ec
import model_building.lr_ridge_experiment_configuration as lr
import model_building.xgboost_experiment_configuration as xgb
import model_building.decision_tree_experiment_configuration as dt
import model_building.random_forest_experiment_configuration as rf
import model_building.svr_experiment_configuration as svr
import model_building.nnls_experiment_configuration as nnls

class ExpConfsGenerator(abc.ABC):
    """
    Abstract class representing a generators of experiment configurations

    Attributes
    ----------
    _campaign_configuration: dict of dict
        The set of options specified by the user though command line and campaign configuration files

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

    def __init__(self, campaign_configuration, seed):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator
        """

        #TODO: modify this constructor and all the other constructors which use it to pass the added attributes

        self._experiment_configurations = []
        self._campaign_configuration = campaign_configuration
        self._random_generator = random.Random(seed)
        self._logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Generates the set of experiment configurations to be evaluated

        Returns
        -------
        list
            a list of the experiment configurations
        """

    def __deepcopy__(self, memo):
        raise NotImplementedError()

class MultiExpConfsGenerator(ExpConfsGenerator):
    """
    Specialization of MultiExpConfsGenerator which wraps multiple generators

    Attributes
    ----------
    generators: dict
        The ExpConfsGenerator to be used

    Methods
    -------
    _get_deep_copy_parameters()
        Computes the parameters to be used by __deepcopy__ of subclasses
    """

    def __init__(self, campaign_configuration, seed, generators):
        """
        Parameters
        ----------
        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities

        generators: dict
            The ExpConfsGenerator to be used
        """
        assert generators
        super().__init__(campaign_configuration, seed)
        self._generators = generators

    def __deepcopy__(self, memo):
        raise NotImplementedError()

    def _get_deep_copy_parameters(self):
        campaign_configuration = self._campaign_configuration
        seed = self._random_generator.random()
        generators = copy.deepcopy(self._generators)
        return campaign_configuration, seed, generators


class MultiTechniquesExpConfsGenerator(MultiExpConfsGenerator):
    """
    Specialization of MultiExpConfsGenerator representing a set of experiment configurations related to multiple techniques

    This class wraps the single TechniqueExpConfsGenerator instances which refer to the single techinique

    Methods
    ------
    generate_experiment_configurations()
        Generates the set of expriment configurations to be evaluated
    """
    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Collect the experiment configurations to be evaluated for all the generators

        Parameters
        ----------
        prefix: [str]
            The prefix to be used in the computation of the signatures of the experiment configurations

        regression_inputs: RegressionInputs
            The input of the regression problem to be used in experiment configurations generated by self

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """
        self._logger.debug("Calling generate experiment_configurations in %s %s %s", str(id(self)), str(prefix), self.__class__.__name__)
        self._logger.debug("Regression inputs is %s %s", str((id(regression_inputs))), str(regression_inputs))
        return_list = []
        assert self._generators
        for key, generator in self._generators.items():
            new_prefix = prefix.copy()
            new_prefix.append(key)
            assert new_prefix
            return_list.extend(generator.generate_experiment_configurations(new_prefix, regression_inputs))
        assert return_list
        return return_list

    def __deepcopy__(self, memo):
        campaign_configuration, seed, generators = self._get_deep_copy_parameters()
        return MultiTechniquesExpConfsGenerator(campaign_configuration, seed, generators)

class TechniqueExpConfsGenerator(ExpConfsGenerator):
    """
    Class which generalize classes for generate points to be explored for each technique
    #TODO: check if there is any technique which actually needs to extends this

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    def __init__(self, campaign_configuration, seed, technique):
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

        super().__init__(campaign_configuration, seed)
        self._technique = technique

    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Collected the set of points to be evaluated for the single technique

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """
        self._logger.debug("Calling generate_experiment_configurations in %s %s", self.__class__.__name__, str(id(self)))
        self._logger.debug("Regression inputs is %s %s", str((id(regression_inputs))), str(regression_inputs))
        assert prefix
        first_key = ""
        #We expect that hyperparameters for a technique are stored in campaign_configuration[first_key] as a dictionary from string to list of values
        if self._technique == ec.Technique.NONE:
            self._logger.error("Not supported regression technique")
            sys.exit(-1)

        first_key = ec.enum_to_configuration_label[self._technique]
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

        self._logger.debug("I am %s", str(id(self)))
        assert not self._experiment_configurations

        #Cartesian product of parameters
        for combination in itertools.product(*hyperparams_values):
            hyperparams_point_values = {}
            for hyperparams_name, hyperparams_value in zip(hyperparams_names, combination):
                hyperparams_point_values[hyperparams_name] = hyperparams_value
            if self._technique == ec.Technique.LR_RIDGE:
                point = lr.LRRidgeExperimentConfiguration(self._campaign_configuration, hyperparams_point_values, regression_inputs, prefix)
            elif self._technique == ec.Technique.XGBOOST:
                point = xgb.XGBoostExperimentConfiguration(self._campaign_configuration, hyperparams_point_values, regression_inputs, prefix)
            elif self._technique == ec.Technique.DT:
                point = dt.DecisionTreeExperimentConfiguration(self._campaign_configuration, hyperparams_point_values, regression_inputs, prefix)
            elif self._technique == ec.Technique.RF:
                point = rf.RandomForestExperimentConfiguration(self._campaign_configuration, hyperparams_point_values, regression_inputs, prefix)
            elif self._technique == ec.Technique.SVR:
                point = svr.SVRExperimentConfiguration(self._campaign_configuration, hyperparams_point_values, regression_inputs, prefix)
            elif self._technique == ec.Technique.NNLS:
                point = nnls.NNLSExperimentConfiguration(self._campaign_configuration, hyperparams_point_values, regression_inputs, prefix)
            else:
                self._logger.error("Not supported regression technique")
                point = None
                sys.exit(-1)
            self._experiment_configurations.append(point)

        assert self._experiment_configurations

        return self._experiment_configurations

    def __deepcopy__(self, memo):
        ret = TechniqueExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), self._technique)
        return ret

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

    def __init__(self, campaign_configuration, seed, repetitions_number, wrapped_generator):
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

        wrapped_generators = {}
        wrapped_generators["run_0"] = wrapped_generator

        for run_index in range(1, self._repetitions_number):
            wrapped_generators["run_" + str(run_index)] = copy.deepcopy(wrapped_generator)
        super().__init__(campaign_configuration, seed, wrapped_generators)

    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Collect the experiment configurations to be evaluated for all the generators

        Parameters
        ----------
        prefix: [str]
            The prefix to be used in the computation of the signatures of the experiment configurations

        regression_inputs: RegressionInputs
            The input of the regression problem to be used in experiment configurations generated by self

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """
        self._logger.debug("Calling generate_experiment_configurations in %s %s", self.__class__.__name__, str(id(self)))

        return_list = []
        assert self._generators
        for key, generator in self._generators.items():
            self._logger.debug("A000")
            new_prefix = prefix.copy()
            new_prefix.append(key)
            return_list.extend(generator.generate_experiment_configurations(new_prefix, regression_inputs))
        assert return_list
        return return_list

    def __deepcopy__(self, memo):
        raise NotImplementedError()

class SelectionValidationExpConfsGenerator(ExpConfsGenerator):
    """
    Base class for generator wrappers used to generate hy_selection or validation set

    Methods
    -------
    _get_selection_validation()
        Base method for get_selection_generator and get_validation_generator

    get_selection_generator()
        Static factory method to instantiate subclasses

    get_validation_generator()
        Static factory method to instantiate subclasses
    """
    def __init__(self, campaign_configuration, seed, is_validation):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator

        is_validation: bool
            True if the instance to be created is for validating, false if it is for hp selection
        """
        self._is_validation = is_validation
        super().__init__(campaign_configuration, seed)

    @staticmethod
    def _get_selection_validation_generator(campaign_configuration, seed, wrapped_generator, subclass_name, is_validation):
        """
        Static factory method to instantiate subclasses

        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator

        wrapped_generator: ExpConfsGenerator
            The generator to be wrapped

        subclass_name: str
            The name of the class to be generated

        is_validation: bool
            True if the instance to be created is for validating, false if it is for hp selection

        Returns
        -------
        The generated instance
        """
        if subclass_name == "All":
            return AllExpConfsGenerator(campaign_configuration, seed, wrapped_generator, is_validation)
        if subclass_name == "Extrapolation":
            if is_validation:
                #Split is performed as preprocessing step
                return AllExpConfsGenerator(campaign_configuration, seed, wrapped_generator, is_validation)
            else:
                logging.error("Extrapolation cannot be used to perform hp_selection")
                sys.exit(1)
        elif subclass_name == "HoldOut":
            return HoldOutExpConfsGenerator(campaign_configuration, seed, wrapped_generator, is_validation)
        elif subclass_name == "KFold":
            return KFoldExpConfsGenerator(campaign_configuration, seed, campaign_configuration['General']['folds'], wrapped_generator, is_validation)
        else:
            logging.error("Unknown hp_selection/validation method %s", subclass_name)
            sys.exit(1)

    @staticmethod
    def get_selection_generator(campaign_configuration, seed, wrapped_generator, hp_selection):
        """
        Static factory method to instantiate subclasses

        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator

        wrapped_generator: ExpConfsGenerator
            The generator to be wrapped

        hp_selection: str
            The name of the class to be generated

        Returns
        -------
        The generated instance
        """
        return SelectionValidationExpConfsGenerator._get_selection_validation_generator(campaign_configuration, seed, wrapped_generator, hp_selection, False)

    @staticmethod
    def get_validation_generator(campaign_configuration, seed, wrapped_generator, validation):
        """
        Static factory method to instantiate subclasses

        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator

        wrapped_generator: ExpConfsGenerator
            The generator to be wrapped

        validation: str
            The name of the class to be generated

        Returns
        -------
        The generated instance
        """
        return SelectionValidationExpConfsGenerator._get_selection_validation_generator(campaign_configuration, seed, wrapped_generator, validation, True)

    def __deepcopy__(self, memo):
        raise NotImplementedError()

class AllExpConfsGenerator(SelectionValidationExpConfsGenerator):
    """
    Wraps a generator and pass to it regression inputs without modification

    Methods
    -------
    generate_experiment_configurations()

    Calls generate_experiment_configurations of the wrapped generator
    """
    def __init__(self, campaign_configuration, seed, wrapped_generator, is_validation):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator

        wrapped_generator: ExpConfsGenerator
            The generator to be wrapped

        is_validation: bool
            True if the instance to be created is for validating, false if it is for hp selection
        """
        self._wrapped_generator = wrapped_generator
        super().__init__(campaign_configuration, seed, is_validation)

    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Calls generate_experiment_configurations of the wrapped generator

        Parameters
        ----------
        prefix: list of str
            The prefix to be considered

        regression_inputs
            The regression inputs to be used with the wrapped generator
        """
        self._logger.debug("Calling generate_experiment_configurations in %s %s", self.__class__.__name__, str(id(self)))
        local_prefix = copy.copy(prefix)
        local_prefix.append("All")
        return self._wrapped_generator.generate_experiment_configurations(local_prefix, regression_inputs)

    def __deepcopy__(self, memo):
        return AllExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), copy.deepcopy(self._wrapped_generator), self._is_validation)

class HoldOutExpConfsGenerator(SelectionValidationExpConfsGenerator):
    """
    Wraps a generator and pass to it the regression_inputs removing data for hp_selection/validation

    Methods
    -------
    generate_experiment_configurations()

    Calls generate_experiment_configurations of the wrapped generator
    """
    def __init__(self, campaign_configuration, seed, wrapped_generator, is_validation):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator

        wrapped_generator: ExpConfsGenerator
            The generator to be wrapped

        is_validation: bool
            True if the instance to be created is for validating, false if it is for hp selection
        """
        self._wrapped_generator = wrapped_generator
        super().__init__(campaign_configuration, seed, is_validation)

    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Calls generate_experiment_configurations of the wrapped generator

        Parameters
        ----------
        prefix: list of str
            The prefix to be considered

        regression_inputs
            The regression inputs to be used with the wrapped generator
        """
        self._logger.debug("Calling generate_experiment_configurations in %s %s", self.__class__.__name__, str(id(self)))
        local_prefix = copy.copy(prefix)
        local_prefix.append("HoldOut")
        local_regression_inputs = regression_inputs.copy()
        destinaion_set = "validation" if self._is_validation else "hp_selection"
        splitter = data_preparation.random_splitting.RandomSplitting(self._campaign_configuration, self._random_generator.random(), "training", destinaion_set)
        local_regression_inputs = splitter.process(local_regression_inputs)
        return self._wrapped_generator.generate_experiment_configurations(local_prefix, local_regression_inputs)

    def __deepcopy__(self, memo):
        return HoldOutExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), copy.deepcopy(self._wrapped_generator), self._is_validation)

class KFoldExpConfsGenerator(SelectionValidationExpConfsGenerator):
    """
    Wraps k instances of a generator with different training set

    Attributes
    ----------
    _k: integer
        The number of different folds to be used

    _validation: bool
        True if this KFold is used for validation, False if it is used to perform hyperparameter selection

    Methods
    ------
    generate_experiment_configurations()

    Generates the set of points to be evaluated
    """

    def __init__(self, campaign_configuration, seed, k, wrapped_generator, is_validation):
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
        self._k = k

        self._kfold_generators = {}

        for fold in range(0, self._k):
            self._kfold_generators[fold] = copy.deepcopy(wrapped_generator)

        super().__init__(campaign_configuration, seed, is_validation)

        for key, value in self._kfold_generators.items():
            self._logger.debug("Wrapped %s is %s", str(key), str(id(value)))


    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Generates the set of experiment configurations to be evaluated

        Returns
        -------
        list
            a list of the experiment configurations
        """
        self._logger.debug("Calling generate experiment_configurations in %s %s %s", str(id(self)), str(prefix), self.__class__.__name__)
        self._logger.debug("Regression inputs is %s %s", str((id(regression_inputs))), str(regression_inputs))
        assert prefix
        if len(regression_inputs.inputs_split["training"]) < self._k:
            self._logger.error("Too few samples to perform %d-Fold", self._k)
            sys.exit(1)
        return_list = []
        dataset_size = len(regression_inputs.inputs_split["training"])
        fold_size = int(dataset_size / self._k)
        all_training_idx = set(regression_inputs.inputs_split["training"])
        remaining = set(all_training_idx)
        for fold in range(0, self._k):
            assert prefix
            fold_prefix = copy.copy(prefix)
            assert fold_prefix
            fold_prefix.append("f" + str(fold))
            assert fold_prefix
            fold_regression_inputs = copy.copy(regression_inputs)
            if fold == self._k - 1:
                fold_testing_idx = remaining
            else:
                fold_testing_idx = set(self._random_generator.sample(remaining, fold_size))
            fold_training_idx = all_training_idx - fold_testing_idx
            remaining = remaining - fold_testing_idx
            fold_regression_inputs.inputs_split["training"] = list(fold_training_idx)
            second_set = "validation" if self._is_validation else "hp_selection"
            fold_regression_inputs.inputs_split[second_set] = list(fold_testing_idx)
            return_list.extend(self._kfold_generators[fold].generate_experiment_configurations(fold_prefix, fold_regression_inputs))
        return return_list

    def __deepcopy__(self, memo):
        return KFoldExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), self._k, self._kfold_generators[0], self._is_validation)

class NormalizationExpConfsGenerator(ExpConfsGenerator):
    """
    Wraps a generator and pass to it normalized regression inputs

    Methods
    -------
    generate_experiment_configurations()

    Calls generate_experiment_configurations of the wrapped generator with normalized data
    """

    def __init__(self, campaign_configuration, seed, wrapped_generator):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the random generator

        wrapped_generator: ExpConfsGenerator
            The wrapped generator to be duplicated and modified
        """
        self._wrapped_generator = wrapped_generator
        super().__init__(campaign_configuration, seed)

    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Calls generate_experiment_configurations of the wrapped generator

        Parameters
        ----------
        prefix: list of str
            The prefix to be considered

        regression_inputs
            The regression inputs to be used with the wrapped generator
        """
        self._logger.debug("Calling generate experiment_configurations in %s %s %s", str(id(self)), str(prefix), self.__class__.__name__)
        self._logger.debug("Regression inputs is %s", str((id(regression_inputs))))
        normalizer = data_preparation.normalization.Normalization(self._campaign_configuration)
        local_regression_inputs = copy.copy(regression_inputs)
        local_regression_inputs = normalizer.process(local_regression_inputs)
        return self._wrapped_generator.generate_experiment_configurations(prefix, local_regression_inputs)

    def __deepcopy__(self, memo):
        return NormalizationExpConfsGenerator(self._experiment_configurations, self._random_generator.random(), copy.deepcopy(self._wrapped_generator))


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

    def __init__(self, campaign_configuration, seed, experiment_configurations_number, wrapped_generator):
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
        super().__init__(campaign_configuration, seed)

        self._experiment_configurations_number = experiment_configurations_number
        self._wrapped_generator = wrapped_generator

    def generate_experiment_configurations(self, prefix, regression_inputs):
        """
        Collected the set of points to be evaluated for the single technique

        Returns
        -------
        list
            a list of the experiment configurations to be evaluated
        """
        #TODO  call wrapped generator and randomly pick n experiment configurations

    def __deepcopy__(self, memo):
        raise NotImplementedError()
