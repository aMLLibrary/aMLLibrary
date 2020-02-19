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

import numpy as np

import data_preparation.data_preparation


class Ernest(data_preparation.data_preparation.DataPreparation):
    """
    Step which computes Ernest features

    This step expects to find "datasize" and "cores" columns. If their name is different in original data, RenameColumns should be used

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Add the Ernest features
    """

    def get_name(self):
        """
        Return "Ernest"

        Returns
        string
            The name of this step
        """
        return "Ernest"

    def process(self, inputs):
        """
        Main method of the class which performs the computation of the Ernest features

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """

        outputs = inputs

        starting_features = ["datasize", "cores"]
        for starting_feature in starting_features:
            if starting_feature not in outputs.data:
                self._logger.error("Column %s not found", starting_feature)
                sys.exit(1)
        outputs.data["datasize_over_cores"] = outputs.data["datasize"].div(outputs.data["cores"])
        outputs.data["log_cores"] = np.log(outputs.data["cores"])
        outputs.data["square_datasize_over_cores"] = np.sqrt(outputs.data["datasize_over_cores"])
        outputs.data["squared_datasize_over_cores"] = np.divide(np.sqrt(outputs.data["datasize"]), outputs.data["cores"])

        outputs.x_columns = ["datasize_over_cores", "log_cores", "square_datasize_over_cores", "squared_datasize_over_cores"]
        return outputs
