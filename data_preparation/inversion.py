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
import sys
import warnings

import numpy as np

import data_preparation.data_preparation


class Inversion(data_preparation.data_preparation.DataPreparation):
    """
    Step adds new columns obtained by inverting values in existing columns

    The set of columns to be inverted is listed in option "inverse" of "DataPreparation" section in campaign configuration.
    The name of the new columns is the name of the old columns with "inverse_" as prefix
    Original columns remain part of the input dataset

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Invert the specified columns
    """

    def get_name(self):
        """
        Return "Inversion"

        Returns
        string
            The name of this step
        """
        return "Inversion"

    def process(self, inputs):
        """
        Main method of the class which performs the actual check

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        warnings.filterwarnings('error')
        np.seterr(all='warn')
        outputs = inputs

        to_be_inv_list = self._campaign_configuration['DataPreparation']['inverse']
        if to_be_inv_list == "[*]":
            to_be_inv_list = inputs.x_columns.copy()

        for column in to_be_inv_list:
            if inputs.data[column].dtype == bool:
                continue
            if inputs.data[column].dtype == object:
                self._logger.error("Trying to invert a string column: %s", column)
                sys.exit(-1)
            try:
                new_column = 1 / np.array(inputs.data[column])
            except Warning:
                self._logger.error("Error in inverting %s", column)
                sys.exit(1)
            new_feature_name = 'inverse_' + column
            outputs.data[new_feature_name] = new_column
            outputs.x_columns.append(new_feature_name)

        return outputs

    @staticmethod
    def check_reciprocal(combination):
        """
        Static function used to avoid computing the product among columns which contain reciprocal values

        Parameters
        ----------
        combination: list of str
            The list of features to be checked for the presence of reciprocal

        Return
        ------
        true if there is at least one couple of reciprocal columns, false otherwise
        """
        for first_element in combination:
            if first_element.startswith("inverse_"):
                plain = first_element.replace("inverse_", "")
                if plain in combination:
                    return True
        return False
