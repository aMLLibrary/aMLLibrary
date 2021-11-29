"""
Copyright 2019 Marco Lattuada
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

import random

import custom_logger
import model_building.design_space as ds
import model_building.experiment_configuration as ec


class GeneratorsFactory:
    """
    Factory calls to build the tree of generators.

    Root generator is the generator to which ExperimentConfigurations are required.
    Leaves generators are the generators (specialized for techniques) which actually generate the ExperimentConfiguration.
    Intermediate generators wraps the generators of the next level and propagate the requests from root to leaves, possibly manipulating the input data set of lower level generators by generating different training, hp_selection, and validation sets and by filtering columns

    The levels of the tree of the generators (going from the leaves to the root) are:
        - TechniqueExpConfsGenerator: these are the generators which actually generate the ExperimentConfigurations
        - SFSExpConfsGenerator (optional): columns (independently for each ExperimentConfiguration) are filtered using SFS
        - HyperoptExpConfsGenerator (optional): hyperparameters will be optimized with the Hyperopt library starting from a prior distribution(s) instead of using a grid search approach
        - HyperoptSFSExpConfsGenerator (optional): combines the above two procedures
        - MultiTechniquesExpConfsGenerator: wrap together generators of the single techniques
        - NormalizationExpConfsGenerator (optional): normalize the data according to the values of the training set
        - SelectionValidationExpConfsGenerator: extract the hp_selection set from training set
        - XGBoostFeatureSelectionExpConfsGenerator (optional): columns are filtered according to XGBoost score
        - SelectionValidationExpConfsGenerator: extract the validation set from training set
        - RepeatedExpConfsGenerator: duplicate multiple times all the nested generators to repeat multiple time the process; for the sake of generality this is added even if the number of runs is 1

    Attributes
    ----------
    _campaign_configuration: dict of str: dict of str: str
        The set of options specified by the user though command line and campaign configuration files

    Methods
    -------
    build()
        Build the required hierarchy of generators on the basis of the configuration file
    """
    def __init__(self, campaign_configuration, seed):
        """
        Parameters
        ----------
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities

        Returns
        -------
        ExpConfsGenerator
            The top level ExpConfsGenerator to be used to generate all the experiment configurations
        """
        self._campaign_configuration = campaign_configuration
        self._random_generator = random.Random(seed)
        self._logger = custom_logger.getLogger(__name__)

    def build(self):
        """
        Build the required hierarchy of generators on the basis of the configuration file

        The methods start from the leaves and go up. Intermediate wrappers must be added or not on the basis of the requirements of the campaign configuration

        """
        string_techique_to_enum = {v: k for k, v in ec.enum_to_configuration_label.items()}

        generators = {}

        for technique in self._campaign_configuration['General']['techniques']:
            self._logger.info("Building technique generator for %s", technique)
            generators[technique] = ds.TechniqueExpConfsGenerator(self._campaign_configuration, None, string_techique_to_enum[technique])
        assert generators

        # Add Hyperopt and/or SFS wrapper generators
        new_generators = {}
        if 'hyperparameter_tuning' in self._campaign_configuration['General'] and self._campaign_configuration['General']['hyperparameter_tuning'].lower() == 'hyperopt':
            # Hyperopt was requested: choose between Hyperopt+SFS or pure Hyperopt
            if 'FeatureSelection' in self._campaign_configuration and "method" in self._campaign_configuration['FeatureSelection'] and self._campaign_configuration['FeatureSelection']['method'] == 'SFS':        
                self._logger.info("Building Hyperopt-SFS generator")
                for technique, generator in generators.items():
                    new_generators[technique] = ds.HyperoptSFSExpConfsGenerator(generator, self._campaign_configuration, self._random_generator.random())
            else:
                self._logger.info("Building Hyperopt generator")
                for technique, generator in generators.items():
                    new_generators[technique] = ds.HyperoptExpConfsGenerator(generator, self._campaign_configuration, self._random_generator.random())
        else:
            # Hyperopt was not requested: choose between pure SFS or no additional wrapper
            if 'FeatureSelection' in self._campaign_configuration and "method" in self._campaign_configuration['FeatureSelection'] and self._campaign_configuration['FeatureSelection']['method'] == 'SFS':
                self._logger.info("Building SFS generator")
                for technique, generator in generators.items():
                    new_generators[technique] = ds.SFSExpConfsGenerator(generator, self._campaign_configuration, self._random_generator.random())
            else:
                new_generators = generators
        generators = new_generators

        # Wrap together different techniques
        self._logger.info("Building multi technique generator")
        generator = ds.MultiTechniquesExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), generators)

        # Add wrapper to perform normalization
        if 'normalization' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['normalization']:
            self._logger.info("Building normalization generator")
            generator = ds.NormalizationExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), generator)

        # Add wrapper to generate hp_selection
        self._logger.info("Building hp selection generator")
        generator = ds.SelectionValidationExpConfsGenerator.get_selection_generator(self._campaign_configuration, self._random_generator.random(), generator, self._campaign_configuration['General']['hp_selection'])

        # Add wrapper to perform XGBoost feature selection
        if 'FeatureSelection' in self._campaign_configuration and "method" in self._campaign_configuration['FeatureSelection'] and self._campaign_configuration['FeatureSelection']['method'] == 'XGBoost':
            self._logger.info("Building hp XGBoost preprocessing generator")
            generator = ds.XGBoostFeatureSelectionExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), generator)

        # Add wrapper to generate validation
        self._logger.info("Building validation generator")
        generator = ds.SelectionValidationExpConfsGenerator.get_validation_generator(self._campaign_configuration, self._random_generator.random(), generator, self._campaign_configuration['General']['validation'])

        # Add wrapper to perform multiple runs
        self._logger.info("Building multirun generator")
        top_generator = ds.RepeatedExpConfsGenerator(self._campaign_configuration, self._random_generator.random(), self._campaign_configuration['General']['run_num'], generator)
        return top_generator
