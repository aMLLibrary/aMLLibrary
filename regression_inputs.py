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

    It wraps a pandas dataframe which actually includes all the data, including all the dataset (i.e., training, hyperparameter selection, validation) and all the columns (i.e., both original and derived by preprocessing steps).
    The dataframe is "filtered" by means of x_columns and input_split which determine which are the columns and rows to be considered.
    Moreover, it contains the y column and all the scalers used to generate scaled column.


    Attributes
    ----------
    data: dataframe
        The whole dataframe

    input_split: dict of str: set(int)
        For each considered set (i.e., training, hyperparameter selection, validation) the indices of the rows which belong to that set

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
    def __init__(self, data, inputs_split, x_cols, y_column):
        """
        Parameters
        data: dataframe
            The whole dataframe

        inputs_split: map of str to list of integers
            How the input is split. Key is the type of set (e.g., training, cv1, validation), value is the list of rows beloning to that set

        x_cols: list of strings
            The labels of the columns of the data frame to be used to train the model

        y_column: string
            The label of the y column
        """
        self.data = data
        self.inputs_split = inputs_split
        self.x_columns = x_cols
        self.scalers = {}
        self.y_column = y_column
        self.scaled_columns = []

    def __copy__(self):
        new_copy = RegressionInputs(self.data.copy(), self.inputs_split.copy(), self.x_columns.copy(), self.y_column)
        new_copy.scaled_columns = self.scaled_columns.copy()
        return new_copy

    def copy(self):
        return self.__copy__()

    def __str__(self):
        ret = "x_columns: " + str(self.x_columns) + " - y_column: " + self.y_column + "\n"
        for name, values in self.inputs_split.items():
            ret = ret + name + ": " + str(values) + "\n"
        ret = ret + "Dimensions: " + str(self.data.shape)
        return ret

    def _get_data(self, rows, columns):
        """
        Extract a portion of the data frame as a matrix

        Parameters
        ----------
        rows: list of integers
            The list of rows to be extracted

        columns: list of str
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
