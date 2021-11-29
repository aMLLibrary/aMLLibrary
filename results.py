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

import collections
import logging
import multiprocessing
import os
import sys
from typing import Dict
from typing import List

import tqdm

import custom_logger
import model_building.experiment_configuration as ec


def evaluate_wrapper(experiment_configuration):
    if experiment_configuration.trained:
        experiment_configuration.evaluate()
    return experiment_configuration


def plot_wrapper(experiment_configuration):
    if experiment_configuration.trained:
        experiment_configuration.generate_plots()
    return experiment_configuration


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

    get_bests()
        Compute the best overall method
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
        self.techniques = campaign_configuration['General']['techniques']
        self._exp_confs = exp_confs
        self.raw_results: Dict[str, Dict] = {}
        self._logger = custom_logger.getLogger(__name__)

        # Logger writes to stdout and file
        file_handler = logging.FileHandler(os.path.join(self._campaign_configuration['General']['output'], 'results.txt'), 'a+')
        self._logger.addHandler(file_handler)

    def collect_data(self):
        """
        Collect the data of all the performed experiments
        """
        exp_conf: ec.ExperimentConfiguration
        processes_number = self._campaign_configuration['General']['j']
        if processes_number == 1:
            self._logger.info("-->Evaluate experiments (sequentially)")
            for exp_conf in tqdm.tqdm(self._exp_confs, dynamic_ncols=True):
                if not exp_conf.trained:
                    continue
                exp_conf.evaluate()
                if bool(self._campaign_configuration['General']['generate_plots']):
                    exp_conf.generate_plots()
            self._logger.info("<--")
        else:
            self._logger.info("-->Evaluate experiments (in parallel)")
            with multiprocessing.Pool(processes_number) as pool:
                self._exp_confs = list(tqdm.tqdm(pool.imap(evaluate_wrapper, self._exp_confs), total=len(self._exp_confs)))
                if bool(self._campaign_configuration['General']['generate_plots']):
                    self._exp_confs = list(tqdm.tqdm(pool.imap(plot_wrapper, self._exp_confs), total=len(self._exp_confs)))
                self._logger.info("<--")

        self.raw_results = {}
        for exp_conf in self._exp_confs:
            self.raw_results[tuple(exp_conf.get_signature())] = exp_conf.mapes

    def get_bests(self):
        """
        Identify for each considered technique, the configuration with the best validation MAPE, also recover all other metrics, and print the results

        Returns
        -------
        an ExperimentConfiguration instance
            the best-performing experiment configuration in terms of MAPE
        a Technique(Enum) instance
            Enum object indicating the technique of the best-performing experiment configuration (see experiment_configuration.py)
        """
        set_names = ["training", "hp_selection", "validation"]
        run_tec_conf_set = recursivedict()
        validation = self._campaign_configuration['General']['validation']
        hp_selection = self._campaign_configuration['General']['hp_selection']
        if (validation, hp_selection) in {("All", "All"), ("Extrapolation", "All"), ("All", "HoldOut"), ("HoldOut", "All"), ("HoldOut", "HoldOut"), ("Extrapolation", "HoldOut")}:
            # For each run, for each technique the best configuration
            run_tec_best_conf = recursivedict()

            # Hyperparameter search
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                technique = conf.technique
                run_tec_conf_set[run][technique][str(conf.get_signature()[4:])] = conf.mapes
                # First experiment for this technique or better than the current best
                if technique not in run_tec_best_conf[run] or conf.mapes["hp_selection"] < run_tec_best_conf[run][technique].mapes["hp_selection"]:
                    run_tec_best_conf[run][technique] = conf

            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("-->Printing results for run %s", str(run))
                overall_run_best = None
                # Print data of single techniques
                for technique in run_tec_best_conf[run]:
                    temp = run_tec_best_conf[run][technique]
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    self._logger.info("---Best result for %s - Configuration is %s - (Training MAPE is %f - HP Selection MAPE is %f) - Validation MAPE is %f", technique, temp.get_signature()[4:], temp.mapes["training"], temp.mapes["hp_selection"], temp.mapes["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or temp.mapes["hp_selection"] < overall_run_best.mapes["hp_selection"]:
                        overall_run_best = temp
                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result is %s", overall_run_best.get_signature()[3:])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                self._logger.info("MAPE: (Training %f - HP Selection %f) - Validation %f", overall_run_best.mapes["training"], overall_run_best.mapes["hp_selection"], overall_run_best.mapes["validation"])
                self._logger.info("RMSE: (Training %f - HP Selection %f) - Validation %f", overall_run_best.rmses["training"], overall_run_best.rmses["hp_selection"], overall_run_best.rmses["validation"])
                self._logger.info("R^2 : (Training %f - HP Selection %f) - Validation %f", overall_run_best.r2s  ["training"], overall_run_best.r2s  ["hp_selection"], overall_run_best.r2s  ["validation"])
                self._logger.info("<--")

        elif (validation, hp_selection) in {("KFold", "All"), ("KFold", "HoldOut")}:
            folds = float(self._campaign_configuration['General']['folds'])
            # For each run, for each fold, for each technique, the best configuration
            run_fold_tec_best_conf = recursivedict()

            # Hyperparameter search inside each fold
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                fold = int(conf.get_signature()[1].replace("f", ""))
                technique = conf.technique
                if "hp_selection" not in run_tec_conf_set[run][technique][str(conf.get_signature_string()[4:])]:
                    for set_name in set_names:
                        run_tec_conf_set[run][technique][str(conf.get_signature_string()[4:])][set_name] = 0
                for set_name in set_names:
                    run_tec_conf_set[run][technique][str(conf.get_signature_string()[4:])][set_name] = run_tec_conf_set[run][technique][str(conf.get_signature_string()[4:])][set_name] + conf.mapes[set_name] / folds
                # First experiment for this fold+technique or better than the current best
                if technique not in run_fold_tec_best_conf[run][fold] or conf.mapes["hp_selection"] < run_fold_tec_best_conf[run][fold][technique].mapes["hp_selection"]:
                    run_fold_tec_best_conf[run][fold][technique] = conf

            # Aggregate different folds (only the value of the metrics)
            run_tec_set = recursivedict()
            for run in run_fold_tec_best_conf:
                for fold in run_fold_tec_best_conf[run]:
                    for tec in run_fold_tec_best_conf[run][fold]:
                        if "hp_selection" not in run_tec_set[run][technique]:
                            for set_name in set_names:
                                run_tec_set[run][tec][set_name] = 0
                        for set_name in set_names:
                            run_tec_set[run][tec][set_name] = run_fold_tec_best_conf[run][fold][tec].mapes[set_name]
                            run_tec_set[run][tec]['rmses'][set_name] = run_fold_tec_best_conf[run][fold][tec].rmses[set_name]
                            run_tec_set[run][tec]['r2s'][set_name] = run_fold_tec_best_conf[run][fold][tec].r2s[set_name]
            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("Printing results for run %s", str(run))
                overall_run_best = ()
                # Print data of single techniques
                for technique in run_tec_set[run]:
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    self._logger.info("---Best result for %s - (Training MAPE is %f - HP Selection MAPE is %f) - Validation MAPE is %f", technique, run_tec_set[run][technique]["training"], run_tec_set[run][technique]["hp_selection"], run_tec_set[run][technique]["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or run_tec_set[run][technique]["hp_selection"] < overall_run_best[1]["hp_selection"]:
                        overall_run_best = (technique, run_tec_set[run][technique], run_tec_set[run][technique]['rmses'], run_tec_set[run][technique]['r2s'])

                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result is %s", overall_run_best[0])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                self._logger.info("MAPE: (Training %f - HP Selection %f) - Validation %f", overall_run_best[1]["training"], overall_run_best[1]["hp_selection"], overall_run_best[1]["validation"])
                self._logger.info("RMSE: (Training %f - HP Selection %f) - Validation %f", overall_run_best[2]["training"], overall_run_best[2]["hp_selection"], overall_run_best[2]["validation"])
                self._logger.info("R^2 : (Training %f - HP Selection %f) - Validation %f", overall_run_best[3]["training"], overall_run_best[3]["hp_selection"], overall_run_best[3]["validation"])
                self._logger.info("<--")

            # Overall best will contain as first argument the technique with the best (across runs) average (across folds) mape on validation; now we consider on all the runs and on all the folds the configuraiton of this technique with best validation mape

        elif (validation, hp_selection) in {("All", "KFold"), ("HoldOut", "KFold"), ("Extrapolation", "KFold")}:
            folds = float(self._campaign_configuration['General']['folds'])
            # For each run, for each technique, for each configuration, the aggregated mape
            run_tec_conf_set = recursivedict()

            # Hyperparameter search aggregating over folders
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                fold = int(conf.get_signature()[2].replace("f", ""))
                technique = conf.technique
                configuration = str(conf.get_signature()[4:])
                if "hp_selection" not in run_tec_conf_set[run][technique][configuration]:
                    for set_name in set_names:
                        run_tec_conf_set[run][technique][configuration][set_name] = 0
                        run_tec_conf_set[run][technique][configuration]["rmses"][set_name] = 0
                        run_tec_conf_set[run][technique][configuration]["r2s"][set_name] = 0
                for set_name in set_names:
                    run_tec_conf_set[run][technique][configuration][set_name] += conf.mapes[set_name] / folds
                    run_tec_conf_set[run][technique][configuration]["rmses"][set_name] += conf.rmses[set_name] / folds
                    run_tec_conf_set[run][technique][configuration]["r2s"][set_name] += conf.r2s[set_name] / folds

            # Select the best configuration for each technique across different folders
            run_tec_best_conf = recursivedict()
            for run in run_tec_conf_set:
                for tec in run_tec_conf_set[run]:
                    for conf in run_tec_conf_set[run][tec]:
                        if tec not in run_tec_best_conf[run] or run_tec_conf_set[run][tec][conf]["hp_selection"] < run_tec_best_conf[run][tec][1]["hp_selection"]:
                            run_tec_best_conf[run][tec] = (conf, run_tec_conf_set[run][tec][conf])

            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("Printing results for run %s", run)
                overall_run_best = ()  # (technique, configuration, mapes)
                # Print data of single techniques
                for technique in run_tec_best_conf[run]:
                    temp = run_tec_best_conf[run][technique]
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    self._logger.info("---Best result for %s - Configuration is %s - (Training MAPE is %f - HP Selection MAPE is %f) - Validation MAPE is %f", technique, temp[0], temp[1]["training"], temp[1]["hp_selection"], temp[1]["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or temp[1]["hp_selection"] < overall_run_best[2]["hp_selection"]:
                        overall_run_best = (technique, temp[0], temp[1])

                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result is %s %s", overall_run_best[0], overall_run_best[1])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                self._logger.info("MAPE: (Training %f - HP Selection %f) - Validation %f", overall_run_best[2]["training"], overall_run_best[2]["hp_selection"], overall_run_best[2]["validation"])
                self._logger.info("RMSE: (Training %f - HP Selection %f) - Validation %f", overall_run_best[2]["rmses"]["training"], overall_run_best[2]["rmses"]["hp_selection"], overall_run_best[2]["rmses"]["validation"])
                self._logger.info("R^2 : (Training %f - HP Selection %f) - Validation %f", overall_run_best[2]["r2s"]["training"], overall_run_best[2]["r2s"]["hp_selection"], overall_run_best[2]["r2s"]["validation"])
                self._logger.info("<--")

        elif (validation, hp_selection) in {("KFold", "KFold")}:
            folds = float(self._campaign_configuration['General']['folds'])
            # For each run, for each external fold, for each technique, the aggregated metrics
            run_efold_tec_conf_set = recursivedict()

            # Hyperparameter search aggregating over internal folders
            for conf in self._exp_confs:
                if not conf.trained:
                    continue
                run = int(conf.get_signature()[0].replace("run_", ""))
                ext_fold = int(conf.get_signature()[2].replace("f", ""))
                technique = conf.technique
                configuration = str(conf.get_signature()[4:])
                if "hp_selection" not in run_tec_conf_set[run][technique][configuration]:
                    for set_name in set_names:
                        run_tec_conf_set[run][technique][configuration][set_name] = 0
                        run_tec_conf_set[run][technique][configuration]["rmses"][set_name] = 0
                        run_tec_conf_set[run][technique][configuration]["r2s"][set_name] = 0
                for set_name in set_names:
                    run_tec_conf_set[run][technique][configuration][set_name] += (conf.mapes[set_name] / (folds * folds))
                    run_tec_conf_set[run][technique][configuration]["rmses"][set_name] += (conf.rmses[set_name] / (folds * folds))
                    run_tec_conf_set[run][technique][configuration]["r2s"][set_name] += (conf.r2s[set_name] / (folds * folds))
                if configuration not in run_efold_tec_conf_set[run][ext_fold][technique]:
                    for set_name in set_names:
                        run_efold_tec_conf_set[run][ext_fold][technique][configuration][set_name] = 0
                        run_efold_tec_conf_set[run][ext_fold][technique][configuration]["rmses"][set_name] = 0
                        run_efold_tec_conf_set[run][ext_fold][technique][configuration]["r2s"][set_name] = 0
                for set_name in set_names:
                    run_efold_tec_conf_set[run][ext_fold][technique][configuration][set_name] += (conf.mapes[set_name] / (folds * folds))
                    run_efold_tec_conf_set[run][ext_fold][technique][configuration]["rmses"][set_name] += (conf.rmses[set_name] / (folds * folds))
                    run_efold_tec_conf_set[run][ext_fold][technique][configuration]["r2s"][set_name] += (conf.r2s[set_name] / (folds * folds))

            # Select the best configuration for each technique in each external fold across different internal folders
            run_efold_tec_best_conf = recursivedict()
            for run in run_efold_tec_conf_set:
                for efold in run_efold_tec_conf_set[run]:
                    for tec in run_efold_tec_conf_set[run][efold]:
                        for conf in run_efold_tec_conf_set[run][efold][tec]:
                            if conf not in run_efold_tec_best_conf[run][efold][tec] or run_efold_tec_conf_set[run][efold][tec][conf]["hp_selection"] < run_efold_tec_best_conf[run][efold][tec][1]["hp_selection"]:
                                run_efold_tec_best_conf[run][efold][tec] = (conf, run_efold_tec_conf_set[run][efold][tec][conf], run_efold_tec_conf_set[run][efold][tec][conf])

            # Aggregate on external folds
            run_tec_set = recursivedict()
            for run in run_efold_tec_best_conf:
                for efold in run_efold_tec_best_conf[run]:
                    for tec in run_efold_tec_best_conf[run][efold]:
                        if "hp_selection" not in run_tec_set[run][tec]:
                            for set_name in set_names:
                                run_tec_set[run][tec][set_name] = 0
                                run_tec_set[run][tec]["rmses"][set_name] = 0
                                run_tec_set[run][tec]["r2s"][set_name] = 0
                        for set_name in set_names:
                            run_tec_set[run][tec][set_name] += run_efold_tec_best_conf[run][efold][tec][1][set_name]
                            run_tec_set[run][tec]["rmses"][set_name] += run_efold_tec_best_conf[run][efold][tec][1]["rmses"][set_name]
                            run_tec_set[run][tec]["r2s"][set_name] += run_efold_tec_best_conf[run][efold][tec][1]["r2s"][set_name]

            # Print results for each run
            for run in range(0, self._campaign_configuration['General']['run_num']):
                unused_techniques = self.techniques
                self._logger.info("Printing results for run %s", run)
                overall_run_best = ()
                # Print data of single techniques
                for technique in run_tec_set[run]:
                    print(">>", technique)
                    if ec.enum_to_configuration_label[technique] in unused_techniques:
                        unused_techniques.remove(ec.enum_to_configuration_label[technique])
                    self._logger.info("---Best result for %s - (Training MAPE is %f - HP Selection MAPE is %f) - Validation MAPE is %f", technique, run_tec_set[run][technique]["training"], run_tec_set[run][technique]["hp_selection"], run_tec_set[run][technique]["validation"])

                    # Compute which is the best technique
                    if not overall_run_best or run_tec_set[run][technique]["hp_selection"] < overall_run_best[1]["hp_selection"]:
                        overall_run_best = (technique, run_tec_set[run][technique])

                if not overall_run_best:
                    self._logger.error("No valid model was found")
                    exit(1)
                if unused_techniques:
                    self._logger.info("The following techniques had no successful runs: %s", str(unused_techniques))
                self._logger.info("<--Overall best result is %s", overall_run_best[0])
                self._logger.info("Metrics for best result:")
                self._logger.info("-->")
                self._logger.info("MAPE: (Training %f - HP Selection %f) - Validation %f", overall_run_best[1]["training"], overall_run_best[1]["hp_selection"], overall_run_best[1]["validation"])
                self._logger.info("RMSE: (Training %f - HP Selection %f) - Validation %f", overall_run_best[1]["rmses"]["training"], overall_run_best[1]["rmses"]["hp_selection"], overall_run_best[1]["rmses"]["validation"])
                self._logger.info("R^2 : (Training %f - HP Selection %f) - Validation %f", overall_run_best[1]["r2s"]["training"], overall_run_best[1]["r2s"]["hp_selection"], overall_run_best[1]["r2s"]["validation"])
                self._logger.info("<--")

        else:
            self._logger.error("Unexpected combination: %s", str((validation, hp_selection)))
            sys.exit(1)
        best_confs = {}
        best_technique = None
        for conf in self._exp_confs:
            if not conf.trained:
                continue
            technique = conf.technique
            if technique not in best_confs or conf.mapes["validation"] < best_confs[technique].mapes["validation"]:
                best_confs[technique] = conf
        for technique in best_confs:
            if not best_technique or best_confs[technique].mapes["validation"] < best_confs[best_technique].mapes["validation"]:
                best_technique = technique
        if bool(self._campaign_configuration['General']['details']):
            for run in run_tec_conf_set:
                for tec in run_tec_conf_set[run]:
                    for conf in run_tec_conf_set[run][tec]:
                        assert "hp_selection" in run_tec_conf_set[run][tec][conf]
                        assert "validation" in run_tec_conf_set[run][tec][conf], "training MAPE not found for " + str(run) + str(tec) + str(conf)
                        self._logger.info("Run %s - Technique %s - Conf %s - Training MAPE %f - Test MAPE %f", str(run), ec.enum_to_configuration_label[tec], str(conf), run_tec_conf_set[run][tec][conf]["hp_selection"], run_tec_conf_set[run][tec][conf]["validation"])
        return best_confs, best_technique
