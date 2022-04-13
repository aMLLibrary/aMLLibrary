"""
Copyright 2022 Bruno Guindani

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
import sys
import warnings

import numpy as np

import data_preparation.data_preparation


class Logarithm(data_preparation.data_preparation.DataPreparation):
    """
    Step adds new columns obtained by taking the natural logarithm of values in existing columns

    The set of columns to be transformed is listed in option "log" of "DataPreparation" section in campaign configuration.
    The name of the new columns is the name of the old columns with "log_" as prefix
    Original columns remain part of the input dataset

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Take logarithms of the specified columns
    """

    def get_name(self):
        """
        Return "Logarithm"

        Returns
        string
            The name of this step
        """
        return "Logarithm"

    def process(self, inputs):
        """
        Main method of the class which performs the actual computation

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        warnings.filterwarnings('error')
        np.seterr(all='warn')
        outputs = inputs

        to_be_logd_list = self._campaign_configuration['DataPreparation']['log']
        if to_be_logd_list == "[*]":
            to_be_logd_list = inputs.x_columns.copy()

        for column in to_be_logd_list:
            if inputs.data[column].dtype == bool:
                self._logger.debug("Skipping logarithm of boolean-valued column: %s", column)
                continue
            if inputs.data[column].dtype == object:
                self._logger.error("Trying to take logarithm of a string column: %s", column)
                sys.exit(-1)
            if any(inputs.data[column] <= 0):
                self._logger.error("Trying to take logarithm of non-positive value in column %s", column)
                sys.exit(-1)
            try:
                new_column = np.log(inputs.data[column])
            except Warning:
                self._logger.error("Error in computing logarithm of %s", column)
                sys.exit(1)
            new_feature_name = 'log_' + column
            outputs.data[new_feature_name] = new_column
            outputs.x_columns.append(new_feature_name)

        return outputs
