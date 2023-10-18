"""
Copyright 2019 Marco Lattuada
Copyright 2021 Bruno Guindani
Copyright 2022 Nahuel Coliva

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
import mlxtend
import mlxtend.feature_selection
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import r2_score, make_scorer

from hyperopt_aml.hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt_aml.hyperopt.pyll import scope

import os
import sys
import copy

import model_building.experiment_configuration as ec



class WrapperExperimentConfiguration(ec.ExperimentConfiguration):
    """
    Template class representing a wrapper for experiment configurations

    This class is intended for experiment configurations that augment regular experiment configurations, by enriching its regression procedure.
    Examples of such classes are HyperoptExperimentConfiguration and SFSExperimentConfiguration.

    Attributes
    ----------
    _wrapped_experiment_configuration : ExperimentConfiguration
        The regressor to be used in conjunction with sequential feature selection

    _wrapped_regressor : Regressor
        The regressor object of the wrapped experiment configuration

    Methods
    -------
    _compute_signature()
        Compute the signature associated with this (wrapped) experiment configuration

    initialize_regressor()
        Initialize the (wrapped) regressor object for the experiments

    get_regressor()
        Return the regressor associated with this (wrapped) experiment configuration

    set_regressor()
        Set the regressor associated with this (wrapped) experiment configuration

    get_default_parameters()
        Get a dictionary with all technique parameters with default values

    get_x_columns()
        Return the columns used in the regression

    set_x_columns()
        Set the columns to be used in the regression

    get_hyperparameters()
        Return the values of the hyperparameters associated with this experiment configuration
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
        self.technique = self._wrapped_experiment_configuration.technique
        super().__init__(campaign_configuration, None, regression_inputs, prefix)

    def _compute_signature(self, prefix):
        """
        Compute the signature associated with this (wrapped) experiment configuration

        Parameters
        ----------
        prefix: list of str
            The signature of this experiment configuration without considering hyperparameters

        Return
        ------
            The signature of the experiment
        """
        return self._wrapped_experiment_configuration.get_signature()

    def initialize_regressor(self):
        """
        Initialize the (wrapped) regressor object for the experiments
        """
        self._wrapped_experiment_configuration.initialize_regressor()

    def get_regressor(self):
        """
        Return the regressor associated with this (wrapped) experiment configuration

        Return
        ------
        regressor object:
            the regressor associated with this (wrapped) experiment configuration
        """
        return self._wrapped_experiment_configuration.get_regressor()

    def set_regressor(self, reg):
        """
        Set the regressor associated with this (wrapped) experiment configuration

        Parameters
        ----------
        reg: regressor object
            the regressor to be set in this (wrapped) experiment configuration
        """
        self._wrapped_regressor = reg
        self._wrapped_experiment_configuration._regressor = reg

    def get_default_parameters(self):
        """
        Get a dictionary with all technique parameters with default values
        """
        return self._wrapped_experiment_configuration.get_default_parameters()

    def get_x_columns(self):
        """
        Return the columns used in the regression

        Return
        ------
        list of str:
            the columns used in the regression
        """
        x_cols = super().get_x_columns()
        return x_cols

    def set_x_columns(self, x_cols):
        """
        Set the columns to be used in the regression

        Parameters
        ----------
        x_cols: list of str
            the columns to be used in the regression
        """
        super().set_x_columns(x_cols)
        self._wrapped_experiment_configuration.set_x_columns(x_cols)

    def get_hyperparameters(self):
        """
        Return
        ------
        dict of str: object
            The hyperparameters associated with this experiment
        """
        return copy.deepcopy(self._wrapped_experiment_configuration._hyperparameters)




class SFSExperimentConfiguration(WrapperExperimentConfiguration):
    """
    Class representing a single experiment configuration for SFS coupled with a generic wrapped regression

    Attributes
    ----------
    _sfs: SequentialFeatureSelector
        The actual sequential feature selector implemented by mlxtend library

    Methods
    -------
    _check_num_features()
        Exit if the maximum number of required features is greater than the number of existing features

    _train()
        Performs the actual building of the linear model

    compute_estimations()
        Compute the estimated values for a given set of data

    print_model()
        Print the representation of the generated model
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
        super().__init__(campaign_configuration, regression_inputs, prefix, wrapped_experiment_configuration)
        self._sfs_trained = False
        self._check_num_features()

    def _check_num_features(self):
        """
        Exit if the maximum number of required features is greater than the number of existing features
        """
        temp_xdata, temp_ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        if self._campaign_configuration['FeatureSelection']['max_features'] > temp_xdata.shape[1]:
            self._logger.error("ERROR: The maximum number of required features must be in range(1, %d)", temp_xdata.shape[1]+1)
            sys.exit(-10)

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        if self._sfs_trained:  # do not run SFS again for the same exp.conf.
            self._logger.debug("SFS already run, thus performing model training only")
            self._wrapped_experiment_configuration._train()
            return
        verbose = 2 if self._campaign_configuration['General']['debug'] else 0
        min_features = self._campaign_configuration['FeatureSelection'].get('min_features', 1)
        self._sfs = mlxtend.feature_selection.SequentialFeatureSelector(estimator=self.get_regressor(), k_features=(min_features, self._campaign_configuration['FeatureSelection']['max_features']), verbose=verbose, scoring=sklearn.metrics.make_scorer(ec.mean_absolute_percentage_error, greater_is_better=False), cv=self._campaign_configuration['FeatureSelection']['folds'])
        xdata, ydata = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        # set the maximum number of required features to the minimum between itself and the number of existing features
        if self._campaign_configuration['FeatureSelection']['max_features'] > xdata.shape[1]:
            self._logger.info("Reduced maximum number of features from %d to %d", self._sfs.k_features[1], xdata.shape[1])
            self._sfs.k_features = (1,xdata.shape[1])
        # Perform feature selection
        self._sfs.fit(xdata, ydata)
        if self._sfs.interrupted_:
            raise KeyboardInterrupt

        x_cols = list(self._sfs.k_feature_names_)
        self._logger.debug("Selected features: %s", str(x_cols))
        # Use the selected features to retrain the regressor, after restoring column names
        filtered_xdata = self._sfs.transform(xdata)  # is an np.array
        filtered_xdata = pd.DataFrame(filtered_xdata, columns=x_cols)
        self.set_x_columns(x_cols)
        self.get_regressor().fit(filtered_xdata, ydata)
        self._sfs_trained = True

    def compute_estimations(self, rows):
        """
        Compute the predictions for data points indicated in rows estimated by the regressor

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, ydata = self._regression_inputs.get_xy_data(rows)
        
        filtered_xdata = xdata
        ret = self.get_regressor().predict(filtered_xdata)
        # self._logger.debug("Using regressor on %s: %s vs %s", str(filtered_xdata), str(ydata), str(ret))

        return ret

    def print_model(self):
        """
        Print the representation of the generated model
        """
        return "".join(("Features selected with SFS: ", str(self.get_x_columns()), "\n",
                        self._wrapped_experiment_configuration.print_model()))




class HyperoptExperimentConfiguration(WrapperExperimentConfiguration):
    """
    Class representing a single experiment configuration whose hyperparameters are to be computed by Hyperopt

    For this configuration, the user must provide the "hyperparameter_tuning = 'hyperopt'" flag in the configuration file.
    In this case, values for hyperparameters may be strings representing prior distributions instead of fixed values. (The use of both prior strings and fixed values is allowed.)
    Values for such hyperparameters will be computed via the Hyperopt library by leveraging Bayesian Optimization techniques.

    Attributes
    ----------
    _hyperopt_max_evals: int
        The maximum number of iterations allowed for the Bayesian Optimization engine

    _hyperopt_save_interval: int
        The number of Hyperopt iterations after which a Pickle checkpoint file is periodically saved

    _hyperopt_trained: bool
        A flag indicating whether or not the optimal hyperparameters have already been computed by Hyperopt

    Methods
    -------
    _objective_function()
        The objective function to be passed to Hyperopt for minimization

    _run_hyperopt()
        Method that calls Hyperopt to find optimal hyperparameters

    _train()
        Build the model with the experiment configuration represented by this object

    compute_estimations()
        Compute the predictions for data points indicated in rows estimated by the regressor

    print_model()
        Print the representation of the generated model

    _parse_prior()
        Interpret prior_ini string as a Hyperopt prior object
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
        super().__init__(campaign_configuration, regression_inputs, prefix, wrapped_experiment_configuration)
        self._hyperopt_max_evals = campaign_configuration['General']['hyperopt_max_evals']
        if 'hyperopt_save_interval' in campaign_configuration['General']:
            self._hyperopt_save_interval = campaign_configuration['General']['hyperopt_save_interval']
        else:
            self._hyperopt_save_interval = 0
        self._hyperopt_trained = False

    def _objective_function(self, params):
        """
        The objective function to be passed to Hyperopt for minimization

        It is the cross-validation R^2 of the wrapped regressor

        Parameters
        ----------
        params: dict of str: objects
            dict containing regression data and model hyperparameters

        Return
        ------
        loss: float
            negative value of the objective function, i.e. a loss

        status: str
            status value to report that the computation was successful
        """
        Xtrain, ytrain = self._regression_inputs.get_xy_data(self._regression_inputs.inputs_split["training"])
        self._wrapped_regressor.__init__(**params)
        score = sklearn.model_selection.cross_val_score(self._wrapped_regressor, Xtrain, ytrain, scoring='r2', cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    def _run_hyperopt(self, params):
        """
        Method that calls Hyperopt to find optimal hyperparameters

        Hyperopt performs a maximum of hyperopt_max_evals round of Bayesian optimization.
        Also, this method writes a Pickle object to a file every hyperopt_save_interval iterations as a checkpoint for fault tolerance.
        The value of both of these parameters value is given as user input in the configuration file.
        If the former is not given, or its value is 0, no Pickle checkpoint will be created.

        Parameters
        ----------
        params: dict of str: object
            dict containing either fixed values for hyperparameters, or Hyperopt prior objects for random hyperparameters

        Return
        ------
        best_params: dict of str: object
            the best hyperparameter configuration found by Hyperopt
        """
        # Temporarily disable output from fmin
        logging.getLogger('hyperopt_aml.hyperopt.tpe').propagate = False
        # Call Hyperopt optimizer
        if self._hyperopt_save_interval == 0:
            # Do not perform periodic saves to Pickle files
            best_params = fmin(self._objective_function, params, algo=tpe.suggest, max_evals=self._hyperopt_max_evals, verbose=False)
        else:
            # Save Trials object every _hyperopt_save_interval iterations for fault tolerance
            trials_pickle_path = os.path.join(self._experiment_directory, 'trials.pickle')

            #fmin retrieves the last saved .pickle file at each call (if any) and saves every self._hyperopt_save_interval iterations
            best_params = fmin(self._objective_function, params, algo=tpe.suggest, max_evals=self._hyperopt_max_evals, trials_save_file=trials_pickle_path, max_queue_len=self._hyperopt_save_interval, verbose=False)
            
            # Clear trials file after finished
            os.remove(trials_pickle_path)
            
        # Restore output from fmin
        logging.getLogger('hyperopt_aml.hyperopt.tpe').propagate = True
        # Recover 'lost' params entries whose values was set, and were thus not returned by fmin()
        for par in params:
            if par not in best_params:
                best_params[par] = params[par]
        best_params = self._wrapped_experiment_configuration.repair_hyperparameters(best_params)
        return best_params

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object
        """
        if self._hyperopt_trained:  # do not run Hyperopt again for the same exp.conf.
            self._logger.debug("Hyperopt already run, thus performing model training only")
            self._wrapped_experiment_configuration._train()
            return
        self._wrapped_regressor = self.get_regressor()
        # Get hyperparameters dictionary, with fixed values or Hyperopt priors
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
        """
        Compute the predictions for data points indicated in rows estimated by the regressor

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        xdata, ydata = self._regression_inputs.get_xy_data(rows)
        ret = self.get_regressor().predict(xdata)
        # self._logger.debug("Using regressor on %s: %s vs %s", str(xdata), str(ydata), str(ret))
        return ret

    def print_model(self):
        """
        Print the representation of the generated model
        """
        hypers = self._wrapped_experiment_configuration._hyperparameters.copy()
        for key in hypers:
            if isinstance(hypers[key], float):
                hypers[key] = round(hypers[key], 3)
        return "".join(("Optimal hyperparameter(s) found with Hyperopt: ", str(hypers), "\n",
                        self._wrapped_experiment_configuration.print_model()))

    def _parse_prior(self, param_name, prior_ini):
        """
        Interpret prior_ini string as a Hyperopt prior object

        This method looks for priors with the following structure: loguniform(0.01,1).
        If prior_ini cannot be interpreted as a prior object, it will be returned as-is, and it will be assumed that it is a point-wise parameter value.

        Parameters
        ----------
        param_name: str
            The name of the hyperparameter to be initialized

        prior_ini: str (or object)
            The string to be interpreted as a prior object; if this fails, it is interpreted as a point-wise parameter value instead

        Return
        ------
        hyperopt prior object (or generic object)
            The Hyperopt prior object, or the original prior_ini if interpretation was not successful
        """
        try:
            prior_type, prior_args_strg = prior_ini.replace(' ', '').replace(')', '').split('(')
            if not hasattr(hp, prior_type):
                self._logger.error("Unrecognized prior type: %s", prior_type)
                raise ValueError()
            # Recover arguments for the prior
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
            self._logger.debug("%s was not recognized as a prior string. It will be assumed it is a point-wise value", prior_ini)
            return prior_ini




class HyperoptSFSExperimentConfiguration(HyperoptExperimentConfiguration):
    """
    Experiment configuration wrapped for using both Hyperopt tuning and SFS selection

    The class is a combination of the above two, for cases when both Hyperopt and SFS are requested.
    A separate wrapper is needed because this combination requires SFS to be conducted manually, as it is intertwined with the Bayesian Optimization steps.
    In particular, Hyperopt is used to optimize hyperparamaters of models with every possible set of features, and the best performing one is kept. 

    Methods
    -------
    _get_standard_evaluator()
        Return the wrapper that applies scorer to model to evaluate it

    _get_cv_evaluator()
        Return the wrapper that applies cross-validation with scorer to model to evaluate it

    print_model()
        Print the representation of the generated model

    _train()
        Build the model with the experiment configuration represented by this object

    compute_estimations()
        Compute the predictions for data points indicated in rows estimated by the regressor
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
        super().__init__(campaign_configuration, regression_inputs, prefix, wrapped_experiment_configuration)
        self._sfs_trained = False
        SFSExperimentConfiguration._check_num_features(self)

    def _get_standard_evaluator(self, scorer):
        """
        Return the wrapper that applies scorer to model to evaluate it

        Parameters
        ----------
        scorer: callable object
            scorer to be applied to the model

        Return
        ------
        evaluator: function
            fits model for data X, evaluates it in y and returns both model and the evaluation score
        """
        def evaluator(model, X, y, trained=False):
            if not trained:
                model = model.fit(X, y)
            score = scorer(model, X, y)
            return model, score
        return evaluator

    def _get_cv_evaluator(self, scorer, cv=3):
        """
        Return the wrapper that applies cross-validation with scorer to model to evaluate it

        Parameters
        ----------
        scorer: callable object
            scorer to be applied to the model

        Return
        ------
        evaluator: function
            fits model for data X, evaluates it in y through cross-validation and returns both model and the evaluation score
        """
        def evaluator(model, X, y, trained=False):
            scores = sklearn.model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)
            if not trained:
                model = model.fit(X, y)
            return model, np.mean(scores)
        return evaluator

    def print_model(self):
        """
        Print the representation of the generated model
        """
        assert self.get_x_columns() == self._wrapped_experiment_configuration.get_x_columns()

        hypers = self._wrapped_experiment_configuration._hyperparameters.copy()
        for key in hypers:
            if isinstance(hypers[key], float):
                hypers[key] = round(hypers[key], 3)
        return "".join(("Optimal hyperparameter(s) found with Hyperopt: ", str(hypers),
                        "\nFeatures selected with SFS: ", str(self.get_x_columns()), "\n",
                        self._wrapped_experiment_configuration.print_model()))

    def _train(self):
        """
        Build the model with the experiment configuration represented by this object

        This method manually implements Feature Selection in order for it to work with Hyperopt parameter optimization
        """
        if self._hyperopt_trained:  # do not run Hyperopt again for the same exp_conf
            self._logger.debug("Hyperopt already run, thus performing SFS training only")
            SFSExperimentConfiguration._train(self)
            return
        if self._sfs_trained:
            self._logger.debug("SFS already run, thus performing hyperopt training only")
            HyperoptExperimentConfiguration._train(self)
            return
        self._wrapped_regressor = self.get_regressor()
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
        min_dim = self._campaign_configuration['FeatureSelection'].get('min_features', 1)
        max_dim = self._campaign_configuration['FeatureSelection']['max_features']
        for dim in range(min_dim, max_dim+1):
            # Containers for candidate metrics and models for this dim
            candidate_metrics = []
            candidate_models = []
            # STEP 2a: TRAIN ALL CANDIDATES WITH THIS DIM
            remaining_features = all_features.difference(selected_features)
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
                                         y_train, trained=False)
            subsets_best_metrics.append(score)
        # end of dim loop
        # STEP 3: SELECT OVERALL BEST CANDIDATE AMONG ONES WITH DIFFERENT DIMS
        idx_best = subsets_argbest(subsets_best_metrics)
        best_features = subsets_best_features[idx_best]
        self._wrapped_experiment_configuration._regressor = subsets_best_models[idx_best]
        self._hyperopt_trained = True
        self._sfs_trained = True
        self._wrapped_experiment_configuration._hyperparameters = subsets_best_hyperparams[idx_best]
        self._logger.debug("Selected features: %s", str(best_features))
        self.set_x_columns(best_features)
        self.get_regressor().fit(X_train[best_features], y_train)

    def compute_estimations(self, rows):
        """
        Compute the predictions for data points indicated in rows estimated by the regressor

        Parameters
        ----------
        rows: list of integers
            The set of rows to be considered

        Returns
        -------
            The values predicted by the associated regressor
        """
        return SFSExperimentConfiguration.compute_estimations(self, rows)
