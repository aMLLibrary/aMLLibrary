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

import random

import data_preparation.data_preparation


class RandomSplitting(data_preparation.data_preparation.DataPreparation):
    """
    Step which split data into training and validation

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Read the data
    """

    def __init__(self, campaign_configuration, seed, starting_set, new_set):
        """
        campaign_configuration: dict of dict:
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used to initialize the internal random generator

        starting_set: str
            The name of the set which has to be split

        new_set: str
            The name of the new set to be created
        """
        super().__init__(campaign_configuration)
        self._random_generator = random.Random(seed)
        self._starting_set = starting_set
        self._new_set = new_set

    def get_name(self):
        """
        Return "RandomSplitting"

        Returns
        string
            The name of this step
        """
        return "RandomSplitting"

    def process(self, inputs):
        """
        Main method of the class which performs the split

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be split
        """
        data = inputs

        assert data.inputs_split["training"]
        validation_size = int(float(len(data.inputs_split["training"])) * self._campaign_configuration['General']['hold_out_ratio'])

        data.inputs_split[self._new_set] = self._random_generator.sample(data.inputs_split[self._starting_set], validation_size)
        data.inputs_split[self._starting_set] = list(set(data.inputs_split[self._starting_set]) - set(data.inputs_split[self._new_set]))

        return data
