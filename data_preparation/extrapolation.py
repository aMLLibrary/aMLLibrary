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


class Extrapolation(data_preparation.data_preparation.DataPreparation):
    """
    Step which prepares data for extrapolation

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Split regression inputs according to extrapolation values
    """

    def get_name(self):
        """
        Return "Extrapolation"

        Returns
        string
            The name of this step
        """
        return "Extrapolation"

    def process(self, inputs):
        data = inputs
        data.inputs_split["training"] = []
        data.inputs_split["validation"] = []
        extrapolation_columns = self._campaign_configuration['General']['extrapolation_columns']
        for index, row in data.data.iterrows():
            validation = False
            for variable, bound in extrapolation_columns.items():
                if row[variable] > bound:
                    validation = True
                    break
            if validation:
                data.inputs_split["validation"].append(index)
            else:
                data.inputs_split["training"].append(index)
        return data
