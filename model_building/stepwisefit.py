"""
Copyright 2019 Eugenio Gianniti
Copyright 2021 Bruno Guindani

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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy import linalg, stats


class Stepwise(BaseEstimator):
    """
    Implementation of the Draper-Smith (1966) stepwise selection + linear regression technique

    Attributes
    ----------
    p_to_add: float
        The minimum significance to add a feature

    p_to_remove: float
        The maximum significance to not remove a feature

    fit_intercept: bool
        True if a constant term has to be added to the linear model

    max_iter: integer
        The maximum number of iterations of the algorithm of adding features

    coef_: numpy.array
        The coefficients of the linear model

    intercept_: float
        The value of the intercept term of the linear model

    k_feature_names_: list of str
        The list of selected features
    """
    def __init__(self, p_to_add=0.05, p_to_remove=0.1, fit_intercept=True, max_iter=100):
        """
        p_to_add: float
            The minimum significance to add a feature

        p_to_remove: float
            The maximum significance to not remove a feature

        fit_intercept: bool
            True if the intercept term has to be added to the linear model

        max_iter: integer
            The maximum number of iterations
        """
        self.p_to_add = p_to_add
        self.p_to_remove = p_to_remove
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
        self.k_feature_names_ = None

    def fit(self, Xinput, y):
        """
        Trains the model on the input data

        https://sourceforge.net/p/octave/statistics/ci/default/tree/inst/stepwisefit.m

        Parameters
        ----------
        Xinput: pandas.DataFrame
            The input of the regression

        y: pandas.DataFrame
            The expected results
        """
        # Remove columns with all identical values, to avoid issues with the correlation matrix
        columns_to_keep = []
        for col in Xinput.columns:
            if Xinput[col].min() != Xinput[col].max():
                columns_to_keep.append(col)
        X = Xinput[columns_to_keep]

        # Initialize relevant variables
        n_regressors = len(X.columns)
        go_on = True
        residuals = y
        counter = 0
        self.k_feature_names_ = []
        n = len(residuals)
        b = np.zeros(0)

        # Loop until nothing happens, or the iteration budget has run dry
        while go_on and counter < self.max_iter:
            counter += 1
            added = False
            dropped = False

            # Try and add a feature, if there is any not added yet
            if len(self.k_feature_names_) < n_regressors:
                # Find candidate features and their correlation matrix
                not_in_use = [c for c in X.columns if c not in self.k_feature_names_]
                possible_additions = X.loc[:, not_in_use]
                rho = possible_additions.join(residuals).corr()
                # Find optimal feature
                most_correlated = rho.iloc[-1, :-1].abs().idxmax(axis="columns")
                current_columns = self.k_feature_names_ + [most_correlated]
                current_data = X.loc[:, current_columns]
                # Perform regression and hypothesis test
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        b_new, b_int_new, r_new = self._regress(current_data, y)
                    z_new = np.abs(b_new[-1] / (b_int_new[-1, 1] - b_new[-1]))
                    if z_new > 1:  # which means you accept to add the feature
                        added = True
                except:  # in case of ill-conditioned matrices or other numerical issues
                    added = False
                if added:
                    b = b_new
                    b_int = b_int_new
                    residuals = pd.Series(r_new, name="r")
                    self.k_feature_names_.append(most_correlated)

            # Try and remove a feature, if there is any to remove
            if self.k_feature_names_:
                # Find candidate features
                variables = len(self.k_feature_names_)
                dof = n - variables - 1 if self.fit_intercept else n - variables
                t_ratio = stats.t.ppf(1 - self.p_to_remove / 2, dof) / stats.t.ppf(1 - self.p_to_add / 2, dof)
                if self.fit_intercept:
                    z = np.abs(b[1:] / (b_int[1:, 1] - b[1:]))
                else:
                    z = np.abs(b / (b_int[:, 1] - b))
                # Find optimal feature
                z_min = np.min(z, axis=None)
                idx = np.argmin(z, axis=None)
                # Perform hypothesis test
                if z_min < t_ratio:  # which means you accept to remove the feature
                    new_features = self.k_feature_names_.copy()
                    del new_features[idx]
                    cur_dat = X.loc[:, new_features]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            b_new, b_int_new, r_new = self._regress(cur_dat, y)
                        dropped = True
                    except:  # in case of ill-conditioned matrices or other numerical issues
                        dropped = False
                    if dropped:
                        b = b_new
                        b_int = b_int_new
                        residuals = pd.Series(r_new, name="r")
                        self.k_feature_names_ = new_features
                        current_data = cur_dat

            go_on = added or dropped
        # end of while loop

        # Save trained coefficients
        if len(b) > 0 and self.fit_intercept:
            self.intercept_ = b[0]
            self.coef_ = b[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = b


    def _regress(self, X_df, y_df):
        """
        Perform regression on the input data

        https://sourceforge.net/p/octave/statistics/ci/default/tree/inst/regress.m

        Parameters
        ----------
        X_df: pandas.DataFrame
            The input of the regression

        y_df: pandas.DataFrame
            The expected results

        Returns
        -------
        beta: numpy.array
            The estimated coefficients of the linear regression
        beta_interval: numpy.array
            The confidence interval for beta
        residuals: numpy.array
            The residuals of the model
        """
        X = X_df.to_numpy()
        y = y_df.to_numpy()
        n = y.size
        if self.fit_intercept:
            X = np.c_[np.ones(n), X]

        Q, R = linalg.qr(X, mode="economic")
        beta = linalg.solve(R, Q.T.dot(y))
        residuals = y - X.dot(beta)

        _, p = X.shape
        dof = n - p
        SSE = residuals.T.dot(residuals)
        MSE = SSE / dof
        t_alpha_2 = stats.t.ppf(self.p_to_add / 2, dof)
        c = np.diag(linalg.inv(R.T.dot(R)))
        # delta is negative, because alpha is small and t_alpha_2 negative
        delta = t_alpha_2 * np.sqrt(MSE * c)
        beta_interval = np.c_[beta + delta, beta - delta]

        return beta, beta_interval, residuals


    def predict(self, X_df):
        """
        Perform the prediction on a set of input data

        Parameters
        ----------
        X_df: pandas.DataFrame
            The input on which prediction has to be applied

        Returns
        -------
        numpy.array
            The values predicted by the trained model
        """
        X = X_df.loc[:, self.k_feature_names_]
        n = len(X.index)
        if self.fit_intercept:
            X = np.c_[np.ones(n), X]
            b = np.r_[self.intercept_, self.coef_]
        else:
            b = self.coef_
        return np.dot(X, b)
