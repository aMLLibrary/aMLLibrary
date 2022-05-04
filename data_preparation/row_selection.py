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

import sys

import data_preparation.data_preparation


class RowSelection(data_preparation.data_preparation.DataPreparation):
    """
    Step which filters input data according to row values

    Filter is applied by discarding rows if one or more feature values exceed the given threshold(s).

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Select the specified columns
    """

    def get_name(self):
        """
        Return "RowSelection"

        Returns
        string
            The name of this step
        """
        return "RowSelection"

    def process(self, inputs):
        """
        Main method of the class which actually performs the filtering
        """
        data = inputs
        if 'skip_rows' in self._campaign_configuration['DataPreparation']:
            skip_dict = self._campaign_configuration['DataPreparation']['skip_rows']
            for index, row in data.data.iterrows():
                remove = False
                for variable, bound in skip_dict.items():
                    if row[variable] > bound:
                        remove = True
                        break
                if remove:
                    data.data.drop(index, inplace=True)
        else:
            self._logger.error("Unexpected condition")
            sys.exit(-1)

        data_indexes = data.data.index.values.tolist()
        data.inputs_split["training"] = data_indexes.copy()
        data.inputs_split["all"] = data_indexes.copy()

        return data
