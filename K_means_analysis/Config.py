import configparser as cp
import json


class Config:
    """This class gets all the user defined configuration related to the pipeline of machine learning task and saves
    them in a dictionary"""

    def __init__(self):
        self.conf = cp.ConfigParser()
        self.parameters = {}
        self.get_parameters()

    def get_parameters(self):
        """Gets the parameters from the config file named parameters.ini and put them into the dictionary
        named parameters"""
        self.conf.read('parameters.ini')

        # Log related parameters
        self.parameters['Log'] = {}
        self.parameters['Log']['Logging'] = self.conf.get('Log', 'Logging')
        self.parameters['Log']['Logfile_path'] = self.conf.get('Log', 'Logfile_path')

        # Input related parameters
        self.parameters['Input'] = {}
        self.parameters['Input']['Input_file'] = self.conf.get('Input', 'Input_file')
        self.parameters['Input']['Input_path'] = self.conf.get('Input', 'Input_path')

        # Pre_processing related parameters: Normalization
        self.parameters['Normalization'] = {}
        self.parameters['Normalization']['Normalize'] = self.conf['Normalization']['Normalize']
        self.parameters['Normalization']['RangeMin'] = self.conf.get('Normalization', 'RangeMin')
        self.parameters['Normalization']['RangeMax'] = self.conf.get('Normalization', 'RangeMax')
        self.parameters['CrossValidation'] = {}

    def print_parameters(self):
        """Prints the parameters in a dictionary form"""
        print(json.dumps(self, indent=1))

