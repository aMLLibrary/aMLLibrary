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

import data_preparation.data_preparation


class RenameColumns(data_preparation.data_preparation.DataPreparation):
    """
    Step which renames columns

    This step renames columns on the basis of the content of campaign_configuration['DataPreparation']['rename_columns']

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Rename the specified columns
    """

    def get_name(self):
        """
        Return "RenameColumns"

        Returns
        string
            The name of this step
        """
        return "RenameColumns"

    def process(self, inputs):
        """
        Main method of the class which performs the renaming

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        data = inputs
        rename_columns = self._campaign_configuration['DataPreparation']['rename_columns']
        data.data.rename(columns=rename_columns, inplace=True)
        if 'extrapolation_columns' in self._campaign_configuration['General']:
            extrapolation_columns = self._campaign_configuration['General']['extrapolation_columns']
            extrapolation_columns_copy = extrapolation_columns.copy()
            for column, value in extrapolation_columns_copy.items():
                if column in rename_columns:
                    del extrapolation_columns[column]
                    extrapolation_columns[rename_columns[column]] = value
        return data
