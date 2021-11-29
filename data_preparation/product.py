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
import pandas as pd

import data_preparation.data_preparation


class Product(data_preparation.data_preparation.DataPreparation):
    """
    Step which generates new columns as product of existing columns

    The created columns are the combination of up to campaign_configuration["DataPreparation"]['product_max_degree']

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
        """
        Main method of the class which performs the actual product

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """

        outputs = inputs

        max_degree = self._campaign_configuration['DataPreparation']['product_max_degree']
        if str(max_degree) == 'inf':
            max_degree = len(inputs.x_columns)

        features = sorted(set(inputs.x_columns))

        for degree in range(2, max_degree + 1):
            if 'product_interactions_only' in self._campaign_configuration['DataPreparation'] and self._campaign_configuration['DataPreparation']['product_interactions_only'] == True:
                combinations = itertools.combinations(features, degree)
            else:
                combinations = itertools.combinations_with_replacement(features, degree)
            for combination in combinations:
                if data_preparation.inversion.Inversion.check_reciprocal(combination):
                    continue
                if data_preparation.onehot_encoding.OnehotEncoding.check_same_class(combination):
                    continue
                # Compute the string for combination[:-2]
                base = self._compute_column_name(combination[:-1])
                new_column = np.array(outputs.data[base]) * np.array(outputs.data[combination[-1]])
                new_feature_name = self._compute_column_name(combination)
                # Skip product if the combined name already exists
                if new_feature_name in outputs.data.columns:
                    self._logger.warning("%s column already exists, skipping product of features %s", new_feature_name, combination)
                    continue
                new_column_df = pd.DataFrame(new_column, columns=[new_feature_name])
                outputs.data = pd.concat([outputs.data, new_column_df], axis=1)
                outputs.x_columns.append(new_feature_name)

        return outputs

    @staticmethod
    def _compute_column_name(combination):
        """
        Static method used to compute the name of the new columns
        """

        return "__".join(combination)
