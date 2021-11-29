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
import logging
import multiprocessing
import os
import pickle
import random
import tqdm

import custom_logger
import data_preparation.normalization
import model_building.experiment_configuration as ec
import model_building.generators_factory as gf
import regressor
import results as re


class ModelBuilding:
    """
    Entry point of the model building phase, i.e., where all the regressions are actually performed

    The process method do the following steps:
        - Create the generators through the factory
        - Build the model for each ExperimentConfiguration
        - Evaluate the MAPE on different sets of each ExperimentConfiguration
        - Identify the best regressor of each technique
        - Retrain the best regressors with the whole dataset
        - Dump the best regressors in pickle format

    Attributes
    ----------
    random_generator : Random
        The internal random generator

    _logger: Logger
        The logger used by this class

    Methods
    ------
    _process_wrapper()
        Wrapper used internally for parallel execution of experiments
    process()
        Generates the set of expriment configurations to be evaluated
    """

    def __init__(self, seed):
        """
        Parameters
        ----------
        seed: integer
            The seed to be used for the internal random generator
        """
        self._random_generator = random.Random(seed)
        self._logger = custom_logger.getLogger(__name__)

    def _process_wrapper(self, experiment_configuration):
        """
        Wrapper used internally for parallel execution of experiments
        """
        if self.debug:
            # Do not use try-except mechanism
            experiment_configuration.train()
        else:
            try:
                experiment_configuration.train()
            except KeyboardInterrupt as ki:
                raise ki
            except:
                pass
        return experiment_configuration

    def process(self, campaign_configuration, regression_inputs, processes_number):
        """
        Perform the actual regression and prints out results

        Regression results, including description of the best model and its performance metrics, are both printed on screen and saved to the results.txt file

        Parameters
        ----------
        campaign_configuration: dict of str: dict of str: tr
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem

        processes_number: integer
            The number of processes which can be used

        Return
        ------
        Regressor
            The best regressor of the best technique
        """
        self.debug = campaign_configuration['General']['debug']
        self._logger.info("-->Generate generators")
        factory = gf.GeneratorsFactory(campaign_configuration, self._random_generator.random())
        top_generator = factory.build()
        self._logger.info("<--")
        self._logger.info("-->Generate experiments")
        expconfs = top_generator.generate_experiment_configurations([], regression_inputs)
        self._logger.info("<--")

        assert expconfs
        if processes_number == 1:
            self._logger.info("-->Run experiments (sequentially)")
            for exp in tqdm.tqdm(expconfs, dynamic_ncols=True):
                if self.debug:
                    # Do not use try-except mechanism
                    exp.train()
                else:
                    try:
                        exp.train()
                    except KeyboardInterrupt as ki:
                        raise ki
                    except:
                        pass
            self._logger.info("<--")
        else:
            self._logger.info("-->Run experiments (in parallel)")
            with multiprocessing.Pool(processes_number) as pool:
                expconfs = list(tqdm.tqdm(pool.imap(self._process_wrapper, expconfs), total=len(expconfs)))
            self._logger.info("<--")

        self._logger.info("-->Collecting results")
        results = re.Results(campaign_configuration, expconfs)
        results.collect_data()
        self._logger.info("<--Collected")

        for signature, mapes in results.raw_results.items():
            for experiment_configuration, mape in mapes.items():
                self._logger.debug("%s: MAPE on %s set is %f", signature, experiment_configuration, mape)

        best_confs, best_technique = results.get_bests()
        best_regressors = {}

        file_handler = logging.FileHandler(os.path.join(campaign_configuration['General']['output'], 'results.txt'), 'a+')
        self._logger.addHandler(file_handler)
        self._logger.info("-->Building the final regressors")
        self._logger.removeHandler(file_handler)

        # Create a shadow copy
        all_data = regression_inputs.copy()

        # Set all sets equal to whole input set
        all_data.inputs_split["training"] = all_data.inputs_split["all"]
        all_data.inputs_split["validation"] = all_data.inputs_split["all"]
        all_data.inputs_split["hp_selection"] = all_data.inputs_split["all"]

        for technique in best_confs:
            best_conf = best_confs[technique]
            # Get information about the used x_columns
            all_data.x_columns = best_conf.get_regressor().aml_features

            if 'normalization' in campaign_configuration['DataPreparation'] and campaign_configuration['DataPreparation']['normalization']:
                # Restore non-normalized columns
                for column in all_data.scaled_columns:
                    all_data.data[column] = all_data.data["original_" + column]
                    all_data.data = all_data.data.drop(columns=["original_" + column])

                all_data.scaled_columns = []
                self._logger.debug("Denormalized inputs are:%s\n", str(all_data))

                # Normalize
                normalizer = data_preparation.normalization.Normalization(campaign_configuration)
                all_data = normalizer.process(all_data)

            # Set training set
            best_conf.set_training_data(all_data)

            # Train and evaluate by several metrics
            best_conf.train()
            best_conf.evaluate()

            self._logger.addHandler(file_handler)
            self._logger.info("Validation metrics on full dataset for %s:", technique)
            self._logger.info("-->")
            self._logger.info("MAPE: %f", best_conf.mapes["validation"])
            self._logger.info("RMSE: %f", best_conf.rmses["validation"])
            self._logger.info("R^2 : %f", best_conf.r2s  ["validation"])
            self._logger.info("<--")
            self._logger.removeHandler(file_handler)

            # Build the regressor
            best_regressors[technique] = regressor.Regressor(campaign_configuration, best_conf.get_regressor(), best_conf.get_x_columns(), all_data.scalers)
            pickle_file_name = os.path.join(campaign_configuration['General']['output'], ec.enum_to_configuration_label[technique] + ".pickle")
            with open(pickle_file_name, "wb") as pickle_file:
                pickle.dump(best_regressors[technique], pickle_file, protocol=4)

        self._logger.addHandler(file_handler)
        self._logger.info("<--Built the final regressors")
        best_config = best_confs[best_technique]
        self._logger.info("Best model:")
        self._logger.info("-->")
        self._logger.info(best_config.print_model())
        self._logger.info("<--")
        self._logger.removeHandler(file_handler)
        file_handler.close()

        # Return the regressor
        return best_regressors[best_technique]
