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

from typing import List

import logging
import mlxtend.feature_selection
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score, make_scorer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import os
import sys
import pickle

import model_building.experiment_configuration as ec


class SFSExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Class representing a single experiment configuration for SFS coupled with a generic regression

    Attributes
    ----------
    _wrapped_experiment_configuration : ExperimentConfiguration
        The regressor to be used in conjunction with sequential feature selection

    _sfs: SequentialFeatureSelector
        The actual sequential feature selector implemented by mlxtend library

    Methods
    -------
    _compute_signature()
        Compute the signature (i.e., an univocal identifier) of this experiment

    _train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the estimated values for a give set of data

    print_model()
        Prints the model
    """
    def __init__(self, campaign_configuration, regression_inputs, prefix: List[str], wrapped_experiment_configuration):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        regression_inputs: RegressionInputs
            The input of the regression problem to be solved

        prefix: list of str
            The information used to identify this experiment

        wrapped_experiment_configuration: ExperimentConfiguration
            The regressor to be used in conjunction with sequential feature selection

        """
        self._wrapped_experiment_configuration = wrapped_experiment_configuration
        super().__init__(campaign_configuration, None, regression_inputs, prefix)
        temp_xdata, temp_ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # if the maximum number of required features is greater than the number of existing features, exit
        if self._campaign_configuration['FeatureSelection']['max_features'] > temp_xdata.shape[1]:
            self._logger.error("ERROR: The maximum number of required features must be in range(1, %d)", temp_xdata.shape[1]+1)
            sys.exit(-10)
        self.technique = self._wrapped_experiment_configuration.technique

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
        return self._wrapped_experiment_configuration.get_signature()

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        verbose = 2 if self._campaign_configuration['General']['debug'] else 0
        self._sfs = mlxtend.feature_selection.SequentialFeatureSelector(estimator=self._wrapped_experiment_configuration.get_regressor(), k_features=(1, self._campaign_configuration['FeatureSelection']['max_features']), verbose=verbose, scoring=sklearn.metrics.make_scorer(ec.mean_absolute_percentage_error, greater_is_better=False), cv=self._campaign_configuration['FeatureSelection']['folds'])
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # set the maximum number of required features to the minimum between itself and the number of existing features
        if self._campaign_configuration['FeatureSelection']['max_features'] > xdata.shape[1]:
            self._logger.info("Reduced maximum number of features from %d to %d", self._sfs.k_features[1], xdata.shape[1])
            self._sfs.k_features = (1,xdata.shape[1])
        # Perform feature selection
        self._sfs.fit(xdata, ydata)
        x_cols = list(self._sfs.k_feature_names_)
        self._logger.debug("Selected features: %s", str(x_cols))
        # Use the selected features to retrain the regressor, after restoring column names
        filtered_xdata = self._sfs.transform(xdata)  # is an np.array
        filtered_xdata = pd.DataFrame(filtered_xdata, columns=x_cols)
        self.set_x_columns(x_cols)
        self._wrapped_experiment_configuration.get_regressor().fit(filtered_xdata, ydata)

    def compute_estimations(self, rows):
        """
        Compute the estimations and the MAPE for runs in rows

        Parameters
        ----------
        rows: list of integer
            The list of the input data to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, ydata = self._regression_inputs.get_xy_data(rows)
        ret = self._wrapped_experiment_configuration.get_regressor().predict(xdata)
        self._logger.debug("Using regressor on %s: %s vs %s", str(xdata), str(ydata), str(ret))

        return ret

    def print_model(self):
        """
        Print the model
        """
        return "".join(("Selected features: ", str(self.get_x_columns()), "\n",
                        self._wrapped_experiment_configuration.print_model()))

    def initialize_regressor(self):
        self._wrapped_experiment_configuration.initialize_regressor()

    def get_regressor(self):
        return self._wrapped_experiment_configuration.get_regressor()

    def get_default_parameters(self):
        return self._wrapped_experiment_configuration.get_default_parameters()

    def get_x_columns(self):
        """
        Return
        ------
        list of str:
            the columns used in the regression
        """
        return self._wrapped_experiment_configuration.get_x_columns()

    def set_x_columns(self, x_cols):
        super().set_x_columns(x_cols)
        self._wrapped_experiment_configuration.set_x_columns(x_cols)



class HyperoptExperimentConfiguration(ec.ExperimentConfiguration):
    def __init__(self, campaign_configuration, regression_inputs, prefix: List[str], wrapped_experiment_configuration):
        self._wrapped_experiment_configuration = wrapped_experiment_configuration
        super().__init__(campaign_configuration, None, regression_inputs, prefix)
        self.technique = self._wrapped_experiment_configuration.technique
        self._hyperopt_max_evals = campaign_configuration['General']['hyperopt_max_evals']
        if 'hyperopt_save_interval' in campaign_configuration['General']:
            self._hyperopt_save_interval = campaign_configuration['General']['hyperopt_save_interval']
        else:
            self._hyperopt_save_interval = 0
        self._hyperopt_trained = False


    def _compute_signature(self, prefix):
        return self._wrapped_experiment_configuration.get_signature()

    def _objective_function(self, params):
        X, y = params['X'], params['y']
        del params['X'], params['y']
        self._wrapped_regressor.__init__(**params)
        score = sklearn.model_selection.cross_val_score(self._wrapped_regressor, X, y, scoring='r2', cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    def _run_hyperopt(self, params):
        # Include datasets and temporarily disable output from fmin
        params['X'], params['y'] = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        logging.getLogger('hyperopt.tpe').propagate = False
        # Call Hyperopt optimizer
        if self._hyperopt_save_interval == 0:
            # Do not perform periodic saves to Pickle files
            best_params = fmin(self._objective_function, params, algo=tpe.suggest, max_evals=self._hyperopt_max_evals, verbose=False)
        else:
            # Save Trials object every _hyperopt_save_interval iterations for fault tolerance
            curr_evals = 0
            trials = Trials()
            trials_pickle_path = os.path.join(self._experiment_directory, 'trials.pickle')
            while curr_evals < self._hyperopt_max_evals:
                # First, check for an existing, partially filled trials file, if any, and restart from there
                if os.path.isfile(trials_pickle_path):
                    with open(trials_pickle_path, 'rb') as f:
                        trials = pickle.load(f)
                    curr_evals = len(trials.trials)
                # Perform next _hyperopt_save_interval iterations and save Pickle file
                curr_evals = min(self._hyperopt_max_evals, curr_evals+self._hyperopt_save_interval)
                best_params = fmin(self._objective_function, params, algo=tpe.suggest, trials=trials, max_evals=curr_evals, verbose=False)
                with open(trials_pickle_path, 'wb') as f:
                    pickle.dump(trials, f)
            # Clear trials file after finished
            os.remove(trials_pickle_path)
        # Restore output from fmin
        logging.getLogger('hyperopt.tpe').propagate = True
        # Recover 'lost' params entries whose values was set, and were thus not returned by fmin()
        for par in params:
            if par not in ['X', 'y'] and par not in best_params:
                best_params[par] = params[par]
        best_params = self._wrapped_experiment_configuration.fix_hyperparameters(best_params)
        return best_params

    def _train(self):
        if self._hyperopt_trained:  # do not run Hyperopt again for the same exp.conf.
            self._wrapped_experiment_configuration._train()
            return
        self._wrapped_regressor = self._wrapped_experiment_configuration.get_regressor()
        prior_dict = self._wrapped_experiment_configuration._hyperparameters
        params = self._wrapped_experiment_configuration.get_default_parameters()
        for param in params:
            if param in prior_dict:
                prior = self._parse_prior(param, prior_dict[param])
                params[param] = prior
        # Call Hyperopt optimizer
        best_param = self._run_hyperopt(params)
        self._hyperopt_trained = True
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
        return "".join(("Optimal hyperparameter(s) found with hyperopt: ",
                        str(self._wrapped_experiment_configuration._hyperparameters),
                        "\n",
                        self._wrapped_experiment_configuration.print_model()))

    def initialize_regressor(self):
        self._wrapped_experiment_configuration.initialize_regressor()

    def get_regressor(self):
        return self._wrapped_experiment_configuration.get_regressor()

    def get_default_parameters(self):
        return self._wrapped_experiment_configuration.get_default_parameters()

    def get_x_columns(self):
        """
        Return
        ------
        list of str:
            the columns used in the regression
        """
        return self._wrapped_experiment_configuration.get_x_columns()

    def set_x_columns(self, x_cols):
        super().set_x_columns(x_cols)
        self._wrapped_experiment_configuration.set_x_columns(x_cols)

    def _parse_prior(self, param_name, prior_ini):
        try:
            prior_type, prior_args_strg = prior_ini.replace(' ', '').replace(')', '').split('(')
            if not hasattr(hp, prior_type):
                self._logger.error("Unrecognized prior type: %s", prior_type)
                sys.exit(1)
            prior_args = [float(a) for a in prior_args_strg.split(',')]
            # Get log of parameter values when appropriate
            if prior_type.startswith('log') or prior_type.startswith('qlog'):
                prior_args = [np.log(a) for a in prior_args]
            # Initialize hyperopt prior object
            prior = getattr(hp, prior_type)(param_name, *prior_args)
            # Handle case of quantized priors, which imply discrete values
            if prior_type.startswith('q'):
                prior = scope.int(prior)
            return prior
        except:
            self._logger.debug("%s was not recognized as a prior string. It will be asssumed it is a point-wise value", prior_ini)
            return prior_ini




class HyperoptSFSExperimentConfiguration(HyperoptExperimentConfiguration):
    def __init__(self, campaign_configuration, regression_inputs, prefix: List[str], wrapped_experiment_configuration):
        super().__init__(campaign_configuration, regression_inputs, prefix, wrapped_experiment_configuration)
        # Check if the maximum number of required features is greater than the number of existing features
        temp_xdata, temp_ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        if self._campaign_configuration['FeatureSelection']['max_features'] > temp_xdata.shape[1]:
            self._logger.error("ERROR: The maximum number of required features must be in range(1, %d)", temp_xdata.shape[1]+1)
            sys.exit(-10)

    def _get_standard_evaluator(self, scorer):
        def evaluator(model, X, y, trained=False):
            if not trained:
                model = model.fit(X, y)
            score = scorer(model, X, y)
            return model, score
        return evaluator

    def _get_cv_evaluator(self, scorer, cv=3):
        def evaluator(model, X, y, trained=False):
            scores = sklearn.model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)
            if not trained:
                model = model.fit(X, y)
            return model, np.mean(scores)
        return evaluator

    def print_model(self):
        """
        Print the model
        """
        return "".join(("Optimal hyperparameter(s) found with hyperopt: ",
                        str(self._wrapped_experiment_configuration._hyperparameters),
                        "\nSelected features: ", str(self.get_x_columns()), "\n",
                        self._wrapped_experiment_configuration.print_model()))

    def _train(self):
        if self._hyperopt_trained:  # do not run Hyperopt again for the same exp.conf.
            SFSExperimentConfiguration._train(self)
            return
        self._wrapped_regressor = self._wrapped_experiment_configuration.get_regressor()
        # Read parameter priors
        prior_dict = self._wrapped_experiment_configuration._hyperparameters
        params = self._wrapped_experiment_configuration.get_default_parameters()
        for param in params:
            if param in prior_dict:
                prior = self._parse_prior(param, prior_dict[param])
                params[param] = prior
        # Get training data
        X_train, y_train = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # Define evaluators
        candidates_evaluator = self._get_standard_evaluator(make_scorer(r2_score))
        candidates_argbest = np.argmax
        subsets_evaluator = self._get_standard_evaluator(make_scorer(r2_score))
        subsets_argbest = np.argmax
        # subsets_* will contain one value for each different dim
        subsets_best_models = []
        subsets_best_metrics = []
        subsets_best_features = []
        subsets_best_hyperparams = []
        selected_features = []
        all_features = X_train.columns
        ## STEP 1: TRAIN DUMMY MODEL
        model_dummy = sklearn.dummy.DummyRegressor()
        # Compute training metrics
        model_dummy, score_dummy = candidates_evaluator(model_dummy, X_train[[]], y_train)
        subsets_best_models.append(model_dummy)
        subsets_best_features.append([])
        subsets_best_hyperparams.append({})
        # Compute validation metric for step 3
        _, score_dummy = subsets_evaluator(model_dummy, X_train[[]], y_train, trained=True)
        subsets_best_metrics.append(score_dummy)
        # STEP 2: EVALUATE ALL CANDIDATES OF ALL DIMENSIONS
        max_dim = self._campaign_configuration['FeatureSelection']['max_features']
        for dim in range(1, max_dim+1):
            # Containers for candidate metrics and models for this dim
            candidate_metrics = []
            candidate_models = []
            # STEP 2a: TRAIN ALL CANDIDATES WITH THIS DIM
            remaining_features = all_features.difference(selected_features)
            # Pass training data
            params['X'] = X_train
            params['y'] = y_train
            # Call Hyperopt optimizer
            best_param = self._run_hyperopt(params)
            subsets_best_hyperparams.append(best_param)
            # Compute training scores
            for new_column in remaining_features:
                X_train_sub = X_train[selected_features+[new_column]]
                self._wrapped_regressor.__init__(**best_param)
                model = self._wrapped_regressor
                model, score = candidates_evaluator(model, X_train_sub, y_train)
                candidate_models.append(model)
                candidate_metrics.append(score)
            # STEP 2b: SELECT BEST CANDIDATE WITH THIS DIM
            idx_best_candidate = candidates_argbest(candidate_metrics)
            # Update selected feature
            selected_features.append(remaining_features[idx_best_candidate])
            # Save best candidate features
            best_features = selected_features.copy()
            subsets_best_features.append(best_features)
            best_subset_model = candidate_models[idx_best_candidate]
            subsets_best_models.append(best_subset_model)
            # Compute validation metric for step 3
            best_subset_X_train = X_train[best_features]
            _, score = subsets_evaluator(best_subset_model, best_subset_X_train,
                                         y_train, trained=True)
            subsets_best_metrics.append(score)
        # end of dim loop
        # STEP 3: SELECT OVERALL BEST CANDIDATE AMONG ONES WITH DIFFERENT DIMS
        idx_best = subsets_argbest(subsets_best_metrics)
        best_features = subsets_best_features[idx_best]
        self._wrapped_experiment_configuration._regressor = subsets_best_models[idx_best]
        self._hyperopt_trained = True
        self._wrapped_experiment_configuration._hyperparameters = subsets_best_hyperparams[idx_best]
        self._logger.debug("Selected features: %s", str(best_features))
        self.set_x_columns(best_features)
        self._wrapped_experiment_configuration.get_regressor().fit(X_train[best_features], y_train)
