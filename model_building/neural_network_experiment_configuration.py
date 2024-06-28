"""
Copyright 2024 Andrea Di Carlo
Copyright 2024 Bruno Guindani

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
import logging
import numpy as np
import os
import pandas as pd

import model_building.experiment_configuration as ec


class NeuralNetworkExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for linear regression

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the linear model

    print_model()
        Print the representation of the generated model

    initialize_regressor()
        Initialize the regressor object for the experiments

    get_default_parameters()
        Get a dictionary with all technique parameters with default values
    """
    def __init__(self, campaign_configuration, hyperparameters, regression_inputs, prefix):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        hyperparameters: dict of str: object
            The set of hyperparameters of this experiment configuration

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved

        prefix: list of str
            The prefix to be added to the signature of this experiment configuration
        """
        assert prefix
        super().__init__(campaign_configuration, hyperparameters, regression_inputs, prefix)
        self.technique = ec.Technique.NEURAL_NETWORK
        # Get Keras backend or resort to default value
        self.backend = campaign_configuration['General'].get('keras_backend', 'tensorflow')
        self.use_cpu = campaign_configuration['General'].get('keras_use_cpu', False)

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this experiment configuration

        Parameters
        ----------
        prefix: list of str
            The signature of this experiment configuration without considering hyperparameters

        Returns
        -------
            The signature of the experiment
        """
        assert isinstance(prefix, list)
        signature = prefix.copy()
        signature.append("n_features" + ''.join([str(_) for _ in self._hyperparameters['n_features']]))
        if isinstance(self._hyperparameters['dropouts'], list):
            dropouts_sig = ''.join([str(_) for _ in self._hyperparameters['dropouts']])
        else:
            dropouts_sig = str(self._hyperparameters['dropouts'])
        signature.append("dropouts" + dropouts_sig)
        signature.append("activation" + self._hyperparameters['activation'])
        signature.append("optimizer" + self._hyperparameters['optimizer'])
        signature.append("learning_rate" + str(self._hyperparameters['learning_rate']))
        signature.append("loss" + self._hyperparameters['loss'])
        signature.append("batch_size" + str(self._hyperparameters['batch_size']))
        signature.append("epochs" + str(self._hyperparameters['epochs']))

        return signature
    

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        self._logger.debug("Building model for %s", self._signature)
        assert self._regression_inputs

        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])

        xdata = xdata.astype(float)
        ydata = ydata.astype(float)

        self._regressor.fit(xdata, ydata, verbose=0, batch_size=self._hyperparameters['batch_size'],
                                                     epochs=self._hyperparameters['epochs'])

        self._logger.debug("Model built")

    def compute_estimations(self, rows):
        """
        Compute the predictions for data points indicated in rows estimated by the regressor

        The actual implementation is demanded to the subclasses

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, _ = self._regression_inputs.get_xy_data(rows)
        xdata = xdata.astype(float)
        predictions = self._regressor.predict(xdata, verbose=0)
        return predictions

    def print_model(self):
        """
        Print the representation of the generated model
        """
        summary_list = []
        self._regressor.summary(print_fn=lambda x: summary_list.append(x))
        ret_string = '\n'.join(summary_list)
        return ret_string

    def initialize_regressor(self):
        """
        Initialize the regressor object for the experiments
        """
        os.environ['KERAS_BACKEND'] = self.backend
        if self.use_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # Disable logging (Keras uses the TensorFlow library even with different backends)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        import keras

        xdata, _ = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        input_shape = (xdata.shape[1],)
        layers_sizes = self._hyperparameters['n_features']
        dropout_ini = self._hyperparameters['dropouts']
        if isinstance(dropout_ini, list):
            dropouts = dropout_ini
        else:
            dropouts = len(layers_sizes) * [dropout_ini]

        # First layer with number of neurons based on input size
        layers = []
        layers.append(keras.layers.Input(shape=input_shape))
        # Intermediate layers
        for i in range(len(layers_sizes)):
            layers.append(keras.layers.Dense(layers_sizes[i], activation=self._hyperparameters['activation']))
            layers.append(keras.layers.Dropout(dropouts[i]))
        # Output layer
        layers.append(keras.layers.Dense(1))

        # Initialize and compile model
        self._regressor = keras.Sequential(layers)
        self._regressor.compile(loss=self._hyperparameters['loss'],
                                optimizer=self._hyperparameters['optimizer'],
                                metrics=[keras.metrics.RootMeanSquaredError()]
        )
        self._regressor.optimizer.learning_rate.assign(self._hyperparameters['learning_rate'])

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return {'n_features': [20, 10], 'dropouts': [0.2, 0.2], 'activation': 'relu', 'optimizer': 'adam',
                'learning_rate': 0.001, 'loss': 'mse', 'batch_size': 10, 'epochs': 5}
