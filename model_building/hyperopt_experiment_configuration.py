from typing import List

import logging
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll import scope

import model_building.experiment_configuration

class HyperoptExperimentConfiguration(model_building.experiment_configuration.ExperimentConfiguration):
    def __init__(self, campaign_configuration, regression_inputs, prefix: List[str], wrapped_experiment_configuration):
        self._wrapped_experiment_configuration = wrapped_experiment_configuration
        super().__init__(campaign_configuration, None, regression_inputs, prefix)
        self.technique = self._wrapped_experiment_configuration.technique

    def _compute_signature(self, prefix):
        return self._wrapped_experiment_configuration.get_signature()

    def _objective_function(self, params):
        X, y = params['X'], params['y']
        del params['X'], params['y']
        self._wrapped_regressor.__init__(**params)
        score = cross_val_score(self._wrapped_regressor, X, y, scoring='r2', cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    def _train(self):
        self._wrapped_regressor = self._wrapped_experiment_configuration.get_regressor()
        # Initialize parameter space and include datasets
        params = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
          'max_depth': scope.int(hp.quniform('max_depth', 100, 300, 100)),
          'gamma': hp.loguniform('gamma', np.log(0.1), np.log(1)),  # aka 'gamma'
          'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 1.01, 1)),  # is fixed
          'n_estimators': scope.int(hp.quniform('n_estimators', 500, 500.1, 1)),  # is fixed
        }
        params['X'], params['y'] = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # Temporarily disable output from fmin
        logging.getLogger('hyperopt.tpe').propagate = False
        # Call Hyperopt minimizer
        best_param = fmin(self._objective_function, params, algo=tpe.suggest,
                          max_evals=1, verbose=False)  # TODO max_evals
        # Restore output from fmin
        logging.getLogger('hyperopt.tpe').propagate = True
        # Convert floats to ints so that XGBoost won't complain
        for key in ['max_depth', 'min_child_weight', 'n_estimators']:
          best_param[key] = int(best_param[key])
        # Train model with the newfound optimal hypers
        self._wrapped_experiment_configuration._regressor = self._wrapped_regressor
        self._wrapped_experiment_configuration._hyperparameters = best_param
        self._wrapped_experiment_configuration._train()

    def compute_estimations(self, rows):
        xdata, ydata = self._regression_inputs.get_xy_data(rows)
        ret = self._wrapped_experiment_configuration.get_regressor().predict(xdata)
        self._logger.debug("Using regressor on %s: %s vs %s", str(xdata), str(ydata), str(ret))
        return ret

    def print_model(self):
        return "".join(("Optimal hyperparameter(s) found with hyperopt\n",
                        self._wrapped_experiment_configuration.print_model()))

    def initialize_regressor(self):
        self._wrapped_experiment_configuration.initialize_regressor()

    def get_regressor(self):
        return self._wrapped_experiment_configuration.get_regressor()
