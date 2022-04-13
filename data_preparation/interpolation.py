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

import numpy as np

import data_preparation.data_preparation


class Interpolation(data_preparation.data_preparation.DataPreparation):
    """
    Step which prepares data for interpolation by computing the correct validation set

    This step looks for interpolation_columns field in campaign configuration and split data into training and validation using the specified interpolation step

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Split regression inputs according to interpolation values
    """

    def get_name(self):
        """
        Return "Interpolation"

        Returns
        string
            The name of this step
        """
        return "Interpolation"

    def process(self, inputs):
        """
        The main method which actually performs the split

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        data = inputs
        validation_data = []
        interpolation_columns = self._campaign_configuration['General']['interpolation_columns']
        for variable, step in interpolation_columns.items():
            values = np.unique(data.data[variable])
            validation_values = values[::step]
            for index, row in data.data.iterrows():
                if row[variable] in validation_values:
                    validation_data.append(index)

        # Place remaining data in the training set
        training_data = list(set(data.data.index) - set(validation_data))
        data.inputs_split["validation"] = validation_data
        data.inputs_split["training"] = training_data

        return data
