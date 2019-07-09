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

class RegressionInputs:
    """
    Data structure storing inputs information for a regression problem

    Attributes
    ----------
    data: dataframe
        The whole dataframe

    training_idx: list of integers
        The indices of the rows of the data frame to be used to train the model

    validation_idx: list of integers
        The indices of the rows of the data frame to be used to validate the model

    x_columns: list of strings
        The labels of the columns of the data frame to be used to train the model

    y_column: string
        The label of the y column

    scalers: dict str->sklearn.preprocessing.StandardScaler
        The scaler which has been used to scale the input

    scaled_columns: list of strings
        The list of columns which have been scaled

    Methods
    -------
    _get_data()
        Extacts a portion of the data frame

    get_xy_data()
        Generates the two pandas data frame with x_columns and y

    """
    def __init__(self, data, training_idx, validation_idx, x_columns, y_column):
        """
        Parameters
        data: dataframe
            The whole dataframe

        training_idx: list of integers
            The indices of the rows of the data frame to be used to train the model

        validation_idx: list of integers
            The indices of the rows of the data frame to be used to validate the model

        x_columns: list of strings
            The labels of the columns of the data frame to be used to train the model

        y_column: string
            The label of the y column
        """
        self.data = data
        self.training_idx = training_idx
        self.validation_idx = validation_idx
        self.x_columns = x_columns
        self.scalers = {}
        self.y_column = y_column
        self.scaled_columns = []

    def __copy__(self):
        return RegressionInputs(self.data, self.training_idx.copy(), self.validation_idx.copy(), self.x_columns.copy(), self.y_column)

    def copy(self):
        return self.__copy__()

    def __str__(self):
        ret = "x_columns: " + str(self.x_columns) + " - y_column: " + self.y_column + "\n"
        ret = ret + "training_idx: " + str(self.training_idx) + "\n"
        ret = ret + "validation_idx: " + str(self.validation_idx) + "\n"
        ret = ret + self.data.to_string()
        return ret

    def _get_data(self, rows, columns):
        """
        Extract a portion of the data frame as a matrix

        Parameters
        ----------
        rows: list of integers
            The list of rows to be extracted

        columns: list of string
            The list of columns to be extracted

        Returns
        matrix
            The specified subset of the data frame
        """
        return self.data.loc[rows, columns]


    def get_xy_data(self, rows):
        """
        Generate the x and y pandas dataframes containing only the necessary information

        Parameters
        ----------
        rows: list of integer
            The list of rows to be considered

        Returns
        -------
        df,df
            The data frame containing the x_columns column and the data frame containing the y column
        """
        xdata = self._get_data(rows, self.x_columns)
        ydata = self._get_data(rows, self.y_column)
        return xdata, ydata
