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

import collections
import logging
import sys

from typing import Dict
from typing import List
from typing import Tuple

import model_building.experiment_configuration as ec

def recursivedict():
    return collections.defaultdict(recursivedict)

class Results:
    """
    Class collecting all the results of a campaign

    Attributes
    ----------
    _campaign_configuration: dict of dict:
        The set of options specified by the user though command line and campaign configuration files


    _exp_confs : List[ec.ExperimentConfiguration]
        The list of all the experiments

    raw_results : Dict[str, Dict[str, float]]
        All the raw results; first key is the type of the result (e.g., MAPE on validation set), second key is the signature of the experiment configuration

    Methods
    -------
    collect_data()
        Collect the data of all the considered experiment configurations

    get_best_for_technique()
        For each technique identify the best model
    """
    def __init__(self, campaign_configuration, exp_confs: List[ec.ExperimentConfiguration]):
        """
        Parameters
        ----------
        campaign_configuration: dict of dict:
            The set of options specified by the user though command line and campaign configuration files

        exp_confs: List[ec.ExperimentConfiguration]
            The list of the run experiment configurations
        """
        self._campaign_configuration = campaign_configuration
        self._exp_confs = exp_confs
        self.raw_results = {}
        self._logger = logging.getLogger(__name__)

    def collect_data(self):
        """
        Collect the data of all the performed experiments
        """
        exp_conf: ec.ExperimentConfiguration
        self.raw_results['validation_MAPE'] = {}
        self.raw_results['hp_selection_MAPE'] = {}
        for exp_conf in self._exp_confs:
            exp_conf.evaluate()
            if bool(self._campaign_configuration['General']['generate_plots']):
                exp_conf.generate_plots()
            self.raw_results['hp_selection_MAPE'][tuple(exp_conf.get_signature())] = exp_conf.hp_selection_mape
            self.raw_results['validation_MAPE'][tuple(exp_conf.get_signature())] = exp_conf.validation_mape

    def get_best_for_technique(self) -> Dict[ec.Technique, Tuple[str, float]]:
        """
        Identify for each considered technique, the configuration with the best validation MAPE

        Returns
        -------
        """
        validation = self._campaign_configuration['General']['validation']
        hp_selection = self._campaign_configuration['General']['hp_selection']

        if (validation, hp_selection) in {("All", "All"), ("Extrapolation", "All"), ("All", "HoldOut"), ("HoldOut", "All"), ("HoldOut", "HoldOut")}:
            #For each run, for each technique the best configuration
            run_tec_best_conf = recursivedict()

            #Hyperparameter search
            for conf in self._exp_confs:
                run = int(conf.get_signature()[0].replace("run_", ""))
                technique = conf.technique
                #First experiment for this technique or better than the current best
                if technique not in run_tec_best_conf[run] or conf.hp_selection_mape < run_tec_best_conf[run][technique].hp_selection_mape:
                    run_tec_best_conf[run][technique] = conf


            #Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                self._logger.info("Printing results for run %d")
                overall_best = None
                #Print data of single techniques
                for technique in run_tec_best_conf[run]:
                    temp = run_tec_best_conf[run][technique]
                    self._logger.info("   Best result for %s - Configuration is %s - Validation MAPE is %f (HP Selection MAPE is %f", technique, temp.get_signature()[4:], temp.validation_mape, temp.hp_selection_mape)

                    #Compute which is the best technique
                    if not overall_best or temp.hp_selection_mape < overall_best.hp_selection_mape:
                        overall_best = temp

                self._logger.info("   Overall best result is %s - Validation MAPE is %f (HP Selection MAPE is %f", overall_best.get_signature()[3:], overall_best.validation_mape, overall_best.hp_selection_mape)

        elif (validation, hp_selection) in {("KFold", "All"), ("KFold", "HoldOut")}:
            folds = float(self._campaign_configuration['General']['folds'])
            #For each run, for each fold, for each technique, the best configuration
            run_fold_tec_best_conf = recursivedict()

            #Hyperparameter search inside each fold
            for conf in self._exp_confs:
                run = int(conf.get_signature()[0].replace("run_", ""))
                fold = int(conf.get_signature()[1].replace("f", ""))
                technique = conf.technique
                #First experiment for this fold+technique or better than the current best
                if technique not in run_fold_tec_best_conf[run][fold] or conf.hp_selection_mape < run_fold_tec_best_conf[run][fold][technique].hp_selection_mape:
                    run_fold_tec_best_conf[run][fold][technique] = conf

            #Aggregate different folds (only the value of the validation_mape
            run_tec_validation_mape = recursivedict()
            for run in run_fold_tec_best_conf:
                for fold in run_fold_tec_best_conf[run]:
                    for tec in run_fold_tec_best_conf[run][fold]:
                        if fold in run_tec_validation_mape[run]:
                            run_tec_validation_mape[run][fold] = run_tec_validation_mape[run][fold] + run_fold_tec_best_conf[run][fold][tec].validation_mape
                        else:
                            run_tec_validation_mape[run][fold] = run_fold_tec_best_conf[run][fold][tec].validation_mape

            #Compute the average
            for run in run_tec_validation_mape:
                for tec in run_tec_validation_mape[run]:
                    run_tec_validation_mape[run][tec] = run_tec_validation_mape[run][tec]/folds

            #Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                self._logger.info("Printing results for run %d")
                overall_best = ()
                #Print data of single techniques
                for technique in run_tec_validation_mape[run]:
                    self._logger.info("   Average validation MAPE for best %s on different folds %f", technique, run_tec_validation_mape[run][technique])

                    #Compute which is the best technique
                    if not overall_best or run_tec_validation_mape[run][technique] < overall_best[1]:
                        overall_best = (technique, run_tec_validation_mape[run][technique])

                self._logger.info("   Overall best result is %s - Average Validation MAPE %f", overall_best[0], overall_best[1])

        elif (validation, hp_selection) in {("All", "KFold"), ("HoldOut", "KFold")}:
            folds = float(self._campaign_configuration['General']['folds'])
            #For each run, for each technique, for each configuration, the aggregated mape
            run_tec_conf_validation_mape = recursivedict()
            run_tec_conf_hp_selection_mape = recursivedict()

            #Hyperparameter search aggregating over folders
            for conf in self._exp_confs:
                run = int(conf.get_signature()[0].replace("run_", ""))
                fold = int(conf.get_signature()[2].replace("f", ""))
                technique = conf.technique
                configuration = tuple(conf.get_signature()[4:])
                if configuration not in run_tec_conf_validation_mape[run][technique]:
                    run_tec_conf_validation_mape[run][technique][configuration] = 0
                    run_tec_conf_hp_selection_mape[run][technique][configuration] = 0
                run_tec_conf_validation_mape[run][technique][configuration] = run_tec_conf_validation_mape[run][technique][configuration] + conf.hp_selection_mape
                run_tec_conf_hp_selection_mape[run][technique][configuration] = run_tec_conf_hp_selection_mape[run][technique][configuration] + conf.hp_selection_mape

            #Select the best configuration for each technique across different folders
            run_tec_best_conf = recursivedict()
            for run in run_tec_conf_hp_selection_mape:
                for tec in run_tec_conf_hp_selection_mape[run]:
                    for conf in run_tec_conf_hp_selection_mape[run][tec]:
                        if conf not in run_tec_best_conf[run][tec] or run_tec_conf_hp_selection_mape[run][tec][conf] < run_tec_best_conf[run][tec][1]:
                            run_tec_best_conf[run][tec] = (conf, run_tec_conf_hp_selection_mape[run][tec][conf])

            #Compute the average
            for run in run_tec_best_conf:
                for tec in run_tec_best_conf[run]:
                    run_tec_best_conf[run][tec] = (run_tec_best_conf[run][tec][0], run_tec_best_conf[run][tec][1]/folds)

            #Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                self._logger.info("Printing results for run %d")
                overall_best = ()
                #Print data of single techniques
                for technique in run_tec_best_conf[run]:
                    temp = run_tec_best_conf[run][tec]
                    self._logger.info("   Best result for %s - Configuration is %s - Validation MAPE is %f (HP Selection MAPE is %f", technique, temp[0], run_tec_conf_validation_mape[run][tec][conf]/folds, temp[1])

                    #Compute which is the best technique
                    if not overall_best or temp[1] < overall_best[2]:
                        overall_best = (technique, temp[0], temp[1])

                self._logger.info("   Overall best result is %s %s - Validation MAPE is %f (HP Selection MAPE is %f)", overall_best[0], overall_best[1], run_tec_conf_validation_mape[run][tec][overall_best[1]], overall_best[2])

        elif (validation, hp_selection) in {("KFold", "KFold")}:
            folds = float(self._campaign_configuration['General']['folds'])
            #For each run, for each external fold, for each technique, the aggregated mape
            run_efold_tec_conf_validation_mape = recursivedict()
            run_efold_tec_conf_hp_selection_mape = recursivedict()

            #Hyperparameter search aggregating over internal folders
            for conf in self._exp_confs:
                run = int(conf.get_signature()[0].replace("run_", ""))
                ext_fold = int(conf.get_signature()[2].replace("f", ""))
                technique = conf.technique
                configuration = tuple(conf.get_signature()[4:])
                if configuration not in run_efold_tec_conf_validation_mape[run][ext_fold][technique]:
                    run_efold_tec_conf_validation_mape[run][ext_fold][technique][configuration] = 0
                    run_efold_tec_conf_hp_selection_mape[run][ext_fold][technique][configuration] = 0
                run_efold_tec_conf_validation_mape[run][ext_fold][technique][configuration] = run_efold_tec_conf_validation_mape[run][ext_fold][technique][configuration] + conf.validation_mape
                run_efold_tec_conf_hp_selection_mape[run][ext_fold][technique][configuration] = run_efold_tec_conf_hp_selection_mape[run][ext_fold][technique][configuration] + conf.hp_selection_mape

            #Select the best configuration for each technique in each external fold across different internal folders
            run_efold_tec_best_conf = recursivedict()
            for run in run_efold_tec_conf_hp_selection_mape:
                for efold in run_efold_tec_conf_hp_selection_mape[run]:
                    for tec in run_efold_tec_conf_hp_selection_mape[run][efold]:
                        for conf in run_efold_tec_conf_hp_selection_mape[run][efold][tec]:
                            if conf not in run_efold_tec_best_conf[run][efold][tec] or run_efold_tec_conf_hp_selection_mape[run][efold][tec][conf] < run_efold_tec_best_conf[run][efold][tec]:
                                run_efold_tec_best_conf[run][efold][tec] = (conf, run_efold_tec_conf_hp_selection_mape[run][efold][tec][conf], run_efold_tec_conf_validation_mape[run][efold][tec][conf])

            #Compute the average
            for run in run_efold_tec_best_conf:
                for efold in run_efold_tec_best_conf[run]:
                    for tec in run_efold_tec_best_conf[run][efold]:
                        run_efold_tec_best_conf[run][efold][tec] = (run_efold_tec_best_conf[run][efold][tec][0], run_efold_tec_best_conf[run][efold][tec][1]/folds)

            #Aggregate on external folds
            run_tec_hp_selection_mape = recursivedict()
            run_tec_validation_mape = recursivedict()
            for run in run_efold_tec_best_conf:
                for efold in run_efold_tec_best_conf[run]:
                    for tec in run_efold_tec_best_conf[run][efold]:
                        if tec not in run_tec_hp_selection_mape[run]:
                            run_tec_hp_selection_mape[run][tec] = 0
                            run_tec_validation_mape[run][tec] = 0
                        run_tec_hp_selection_mape[run][tec] = run_tec_hp_selection_mape[run][tec] + run_efold_tec_best_conf[run][efold][tec][1]
                        run_tec_validation_mape[run][tec] = run_tec_validation_mape[run][tec] + run_efold_tec_best_conf[run][efold][tec][1]

            #Compute the average
            for run in run_tec_hp_selection_mape:
                for tec in run_tec_hp_selection_mape[run]:
                    run_tec_hp_selection_mape[run][tec] = run_tec_hp_selection_mape[run][tec]/folds
                    run_tec_validation_mape[run][tec] = run_tec_validation_mape[run][tec]/folds

            #Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                self._logger.info("Printing results for run %d")
                overall_best = ()
                #Print data of single techniques
                for technique in run_tec_validation_mape[run]:
                    self._logger.info("   Best result for %s - Validation MAPE is %f (HP Selection MAPE is %f", technique, run_tec_validation_mape[run][technique], run_tec_hp_selection_mape[run][technique])

                    #Compute which is the best technique
                    if not overall_best or run_tec_conf_hp_selection_mape[run][technique] < overall_best[2]:
                        overall_best = (technique, run_tec_validation_mape[run][technique], run_tec_hp_selection_mape[run][technique])

                self._logger.info("   Overall best result is %s - Validation MAPE is %f (HP Selection MAPE is %f)", overall_best[0], overall_best[1], overall_best[2])

        else:
            self._logger.error("Unexpected combination: %s", str((validation, hp_selection)))
            sys.exit(1)
