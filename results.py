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

from typing import Dict
from typing import List
from typing import Tuple

import model_building.experiment_configuration as ec

class Results:
    """
    Class collecting all the results of a campaign

    Attributes
    ----------
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
    def __init__(self, exp_confs: List[ec.ExperimentConfiguration]):
        """
        Parameters
        ----------
        exp_confs: List[ec.ExperimentConfiguration]
            The list of the run experiment configurations
        """
        self._exp_confs = exp_confs
        self.raw_results = {}

    def collect_data(self):
        """
        Collect the data of all the performed experiments
        """
        exp_conf: ec.ExperimentConfiguration
        self.raw_results['validation_MAPE'] = {}
        for exp_conf in self._exp_confs:
            exp_conf.validate()
            self.raw_results['validation_MAPE'][exp_conf.get_signature_string()] = exp_conf.validation_mape

    def get_best_for_technique(self) -> Dict[ec.Technique, Tuple[str, float]]:
        """
        Identify for each considered technique, the configuration with the best validation MAPE

        Returns
        -------
            A
        """
        exp_conf: ec.ExperimentConfiguration
        self.raw_results['best_validation_MAPE'] = {}
        return_value: Dict[ec.Technique, Tuple[str, float]] = {}
        for exp_conf in self._exp_confs:
            technique = exp_conf.technique
            if technique not in return_value or exp_conf.validation_mape < return_value[technique][1]:
                return_value[technique] = (exp_conf.get_signature_string(), exp_conf.validation_mape)
        return return_value
