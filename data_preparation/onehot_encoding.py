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

import pandas as pd

import data_preparation.data_preparation


class OnehotEncoding(data_preparation.data_preparation.DataPreparation):
    """
    Step which transforms a categorical feature in some onehot encoded features

    All the categorical columns are transformed


    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Select the specified columns
    """

    def get_name(self):
        """
        Return "OnehotEncoding"

        Returns
        string
            The name of this step
        """
        return "OnehotEncoding"

    def process(self, inputs):
        """
        Main method of the class which performs the actual one hot encoding

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        data = inputs

        categorical_feature_mask = data.data.dtypes == object  # filter categorical columns using mask and turn it into a list

        categorical_cols = data.data.columns[categorical_feature_mask].tolist()

        categorical_cols = list(set(data.x_columns) & set(categorical_cols))

        self._logger.debug("Categorical columns %s", str(categorical_cols))

        for categorical_col in categorical_cols:
            original_columns = data.data.columns
            data.data = pd.get_dummies(data.data, columns=[categorical_col], prefix=[categorical_col + "_class"], dtype=bool)
            new_columns = list(set(data.data.columns) - set(original_columns))
            old_column_index = data.x_columns.index(categorical_col)
            data.x_columns[old_column_index:old_column_index + 1] = new_columns

        return data

    @staticmethod
    def check_same_class(combination):
        """
        Static method to avoid generation of zero column as product of mutual exclusive categories.

        Check if a set of columns there are at least two columns which have been built as one hot encoding of two different categories of the same original column. Since the values of these two columns can never be 1 at the same time, the column computed as product of all the features of combination will always 0

        Parameters
        ----------
        combination: list of str
            The list of columns to be checked

        Return
        ------
        true if the product would result in 0 column, false otherwise
        """

        classes = set()
        for element in combination:
            if "_class_" in element:
                class_name = element.split("_class_")[0]
                if class_name in classes:
                    return True
                classes.add(class_name)
        return False
