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


class DataCheck(data_preparation.data_preparation.DataPreparation):
    """
    Step which looks for infinite or nan input data

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Select the specified columns
    """

    def get_name(self):
        """
        Return "DataCheck"

        Returns
        string
            The name of this step
        """
        return "DataCheck"

    def process(self, inputs):
        """
        Main method of the class which performs the actual check

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """
        for column in inputs.x_columns:
            self._logger.debug("Checking column %s", column)
            if np.any(np.isnan(inputs.data[column])):
                self._logger.error("nan in column %s", column)
                sys.exit(-1)
            if not np.all(np.isfinite(inputs.data[column])):
                self._logger.error("infinte in column %s", column)
                sys.exit(-1)
        if np.any(np.isnan(inputs.data[inputs.y_column])):
            self._logger.error("nan in column %s", inputs.y_column)
            sys.exit(-1)
        if not np.all(np.isfinite(inputs.data[inputs.y_column])):
            self._logger.error("infinte in column %s")
            sys.exit(-1)
        return inputs
