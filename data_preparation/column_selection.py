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

import data_preparation.data_preparation


class ColumnSelection(data_preparation.data_preparation.DataPreparation):
    """
    Step which filters input data according to column names

    Filter is applied by simply modifiying the x_columns of the inputs, i.e., no columns is discarded.

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Select the specified columns
    """

    def get_name(self):
        """
        Return "ColumnSelection"

        Returns
        string
            The name of this step
        """
        return "ColumnSelection"

    def process(self, inputs):
        """
        Main method of the class which actually performs the filtering
        """
        data = inputs
        if "use_columns" in self._campaign_configuration['DataPreparation']:
            data.x_columns = self._campaign_configuration['DataPreparation']['use_columns'].copy()
            for column in data.x_columns:
                if column not in data.data:
                    self._logger.error("Column %s not found", column)
                    sys.exit(-1)
        elif "skip_columns" in self._campaign_configuration['DataPreparation']:
            columns_to_be_analyzed = data.x_columns.copy()
            data.x_columns = []
            skip_columns = set(self._campaign_configuration['DataPreparation']['skip_columns'])
            for column in columns_to_be_analyzed:
                if column not in skip_columns:
                    data.x_columns.append(column)
        else:
            self._logger.error("Unexpected condition")
            sys.exit(-1)
        return data
