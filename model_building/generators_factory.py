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

class GeneratorsFactory:
    """
    Factory calls to build the logical hierarchy of generators

    Methods
    build()
        Build the required hierarchy of generators on the basis of the configuration file
    """

    _campaign_configuration = None
    _seed = 0

    def __init__(self, campaign_configuration, seed):
        """
        Parameters
        ----------
        campaign_configuration: #TODO: add type
            The set of options specified by the user though command line and campaign configuration files

        seed: integer
            The seed to be used in random based activities
        """
        self._campaign_configuration = campaign_configuration
        self._seed = seed

    def build(self):
        """
        Build the required hierarchy of generators on the basis of the configuration file

        The methods start from the leaves and go up. Intermediate wrappers must be added or not on the basis of the requirements of the campaign configuration
        """

        #TODO: read from the campaign configuration file which are the techniques to be used; create a TechniqueExpConfsGenerator for each of them

        #TODO: if we want to use k-fold, wraps the generator with KFoldExpConfsGenerator

        #TODO: if we want to use feature selection, wraps with a subclass of FSExpConfsGenerator

        #TODO: if we want to run multiple times wraps the generator with RepeatedExpConfsGenerator

        #TODO: wrap everyhing with MultiTechniquesExpConfsGenerator


