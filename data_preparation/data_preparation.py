"""
Copyright 2019 Marjan Hosseini
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

import abc

import custom_logger


class DataPreparation:
    """
    This is the parent abstract class of all the pre-processing steps

    Attributes
    ----------
    _campaign_configuration: dict of dict
        The set of options specified by the user though command line and campaign configuration files

    _logger: Logger
        The logger used by this class and by all the descendants

    Methods
    ------0
    get_name()
        Return the name of this step
    """

    def __init__(self, campaign_configuration):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files
        """
        assert campaign_configuration
        self._campaign_configuration = campaign_configuration
        self._logger = custom_logger.getLogger(__name__)

    @abc.abstractmethod
    def get_name(self):
        """
        Return the name of the current step
        """

    @abc.abstractmethod
    def process(self, inputs):
        """
        Process the data according to the actual specialization of the class

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be elaborated

        Return
        ------
            The elaborated data
        """
