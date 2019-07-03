#!/usr/bin/env python3
"""
Copyright 2019 Marjan Hosseini
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

import numpy
import sklearn.compose
import sklearn.preprocessing

import data_preparation.data_preparation as dp

class Normalization(dp.DataPreparation):
    """
    Step which normalizes input data

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Performs the actual normalization
    """

    def get_name(self):
        """
        Return "Normalization"

        Returns
        string
            The name of this step
        """
        return "Normalization"

    def process(self, inputs):
        """
        Normalizes the data using StandardScaler module

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be normalized

        Return
        ------
            The normalized data
        """

        outputs = inputs

        to_be_normalized = inputs.x_columns.copy()
        to_be_normalized.append(inputs.y_column)

        filtered_data = inputs.data.iloc[inputs.training_idx, :]
        #filtered_data = inputs.data

        #Extract the columns which have to be normalized
        for column in to_be_normalized:
            outputs.scaled_columns.append(column)
            data_to_be_normalized = filtered_data[column].to_numpy()
            data_to_be_normalized = numpy.reshape(data_to_be_normalized, (-1, 1))
            outputs.scalers[column] = sklearn.preprocessing.StandardScaler().fit(data_to_be_normalized)
            normalized_data = outputs.scalers[column].transform(data_to_be_normalized)
            outputs.data["original_" + column] = outputs.data[column]
            outputs.data[column] = normalized_data

        return outputs
