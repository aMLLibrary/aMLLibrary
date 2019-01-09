import pandas as pd
import configparser as cp
import os
import logging


class SequenceDataProcessing:
    """This is the main class defining the pipeline of machine learning task"""
    def __init__(self):
        self.conf = cp.ConfigParser()
        # self.parameters = get_parameters(self, conf)
        self.parameters = {}

    def get_parameters(self, conf):
        """Gets the parameters from the config file named parameters.ini and put them into a dictionary
        named parameters"""
        conf.read('parameters.ini')
        self.parameters['Normalization'] = {}
        self.parameters['Normalization']['Normalize'] = self.conf['Normalization']['Normalize']
        self.parameters['Normalization']['RangeMin'] = self.conf.get('Normalization', 'RangeMin')
        self.parameters['Normalization']['RangeMax'] = self.conf.get('Normalization', 'RangeMax')
        self.parameters['CrossValidation'] = {}
        return self.parameters

