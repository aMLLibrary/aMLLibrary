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

import os
import sys

import pandas as pd

import data_preparation.data_preparation
import regression_inputs


class DataLoading(data_preparation.data_preparation.DataPreparation):
    """
    Step which load data from csv

    This step is the first to be executed in the whole flow

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Read the data
    """

    def get_name(self):
        """
        Return "DataLoading"

        Returns
        string
            The name of this step
        """
        return "DataLoading"

    def process(self, inputs):
        """
        Main method of the class which performs the actual load and return a RegressionInputs

        In the created RegressionInputs, training set is put equal to the whole input dataset

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        input_path = self._campaign_configuration['DataPreparation']['input_path']
        self._logger.info("Input reading: %s", input_path)
        if not os.path.exists(input_path):
            # # The absolute path of the current script
            # abs_script = os.path.abspath(sys.argv[0])

            # # The root directory of the script
            # abs_root = os.path.dirname(abs_script)

            # new_input_path = os.path.join(abs_root, "inputs", input_path)
            # if os.path.exists(new_input_path):
            #     self._logger.warning("%s not found. Trying %s", input_path, new_input_path)
            #     input_path = new_input_path
            # else:
                self._logger.error("%s not found", input_path)
                sys.exit(-1)

        data_frame = pd.read_csv(input_path)

        self._campaign_configuration['Features'] = {}
        self._campaign_configuration['Features']['Original_feature_names'] = []
        for column_name in data_frame.columns.values:
            if column_name != self._campaign_configuration['General']['y']:
                self._campaign_configuration['Features']['Original_feature_names'].append(column_name)

        inputs_split = {}
        inputs_split["training"] = data_frame.index.values.tolist()
        inputs_split["all"] = inputs_split["training"].copy()

        output = regression_inputs.RegressionInputs(data_frame, inputs_split, self._campaign_configuration['Features']['Original_feature_names'], self._campaign_configuration['General']['y'])

        return output
