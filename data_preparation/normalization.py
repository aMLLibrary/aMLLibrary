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
        inputs.scaler = sklearn.preprocessing.StandardScaler()
        inputs.scaled_columns = []
        #Extract the columns which have to be normalized
        for column in inputs.x_columns:
            #FIXME: add a check if this column has actually be normalized by looking at the some field in the campaign_configuration
            inputs.scaled_columns.append(column)

        #Filter training data
        training_inputs = inputs.data.loc[inputs.training_idx, :]

        #Build the column transformer

        #Compute the normalization parameters
        column_transformer = sklearn.compose.ColumnTransformer([('fit_on_training', inputs.scaler, inputs.scaled_columns)], remainder='passthrough')
        column_transformer.fit(training_inputs)

        #Apply normalization to the whole data frame
        column_transformer.transform(inputs.data)
        return inputs
