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

import itertools

import numpy as np

import data_preparation.data_preparation

class Product(data_preparation.data_preparation.DataPreparation):
    """
    Step which load data from csv

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Add the product up to n grade

    _compute_column_name()
        Compute the name of the column produced as product of existing column
    """

    def get_name(self):
        """
        Return "Product"

        Returns
        string
            The name of this step
        """
        return "Product"

    def process(self, inputs):

        outputs = inputs

        max_degree = self._campaign_configuration["DataPreparation"]['product_max_degree']
        if str(max_degree) == "inf":
            max_degree = len(inputs.x_columns)

        features = sorted(set(inputs.x_columns))

        for degree in range(2, max_degree + 1):
            combinations = itertools.combinations(features, degree)
            for combination in combinations:
                if data_preparation.inversion.Inversion.check_reciprocal(combination):
                    continue
                #Compute the string for combination[:-2]
                base = self._compute_column_name(combination[:-1])
                new_column = np.array(outputs.data[base]) * np.array(outputs.data[combination[-1]])
                new_feature_name = self._compute_column_name(combination)
                outputs.data[new_feature_name] = new_column
                outputs.x_columns.append(new_feature_name)

        return outputs

    @staticmethod
    def _compute_column_name(combination):
        return "_".join(combination)
