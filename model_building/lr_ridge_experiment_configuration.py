#!/usr/bin/env python3
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

class LRRidgeExperimentConfiguration(ExperimentConfiguration):
    """
    Class representing a single experiment configuration for linear regression
    """

    def __init__(self, hyperparameters, data, training_idx, xs, y):
        """
        hyperparameters: dictionary
            The set of hyperparameters of this experiment configuration

        data: dataframe
            The whole dataframe

        training_idx: list of integers
            The indices of the rows of the data frame to be used to train the model

        xs: list of strings
            The labels of the columns of the data frame to be used to train the model

        y: string
            The label of the y column
        """

    def train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        #TODO: implement linear regression here

