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

import numpy as np
import sklearn.compose
import sklearn.preprocessing

import data_preparation.data_preparation as dp


class Normalization(dp.DataPreparation):
    """
    Step which normalizes input data

    All the currently selected columns (i.e., x_columns are considered)

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

        The generated scalers are added to the RegressionInputs so that they can be used also to scale new data

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be normalized

        Return
        ------
            The normalized data
        """

        self._logger.debug(str(inputs))
        data = inputs

        to_be_normalized = inputs.x_columns.copy()
        to_be_normalized.append(inputs.y_column)

        filtered_data = inputs.data.iloc[inputs.inputs_split["training"], :]
        # filtered_data = inputs.data

        # Extract the columns which have to be normalized
        for column in to_be_normalized:
            data.scaled_columns.append(column)

            normalization_support = filtered_data[column].to_numpy()
            normalization_support = np.reshape(normalization_support, (-1, 1))
            data.scalers[column] = sklearn.preprocessing.StandardScaler().fit(normalization_support)

            data_to_be_normalized = data.data[column].to_numpy()
            data_to_be_normalized = np.reshape(data_to_be_normalized, (-1, 1))
            normalized_data = data.scalers[column].transform(data_to_be_normalized)
            data.data["original_" + column] = data.data[column]
            data.data[column] = normalized_data

        return data
