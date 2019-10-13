"""
Copyright 2019 Eugenio Gianniti

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

import numpy
from scipy import linalg, stats


class Stepwise:
    def __init__(self, p_enter=0.05, p_remove=0.1, fit_intercept=True, max_iter=100):
        self._p_to_add = p_enter
        self._p_to_discard = p_remove
        self._add_intercept = fit_intercept
        self._max_iterations = max_iter
        self.coef_ = None
        self.k_feature_names_ = None

    def fit(self, X, y):
        """
        https://sourceforge.net/p/octave/statistics/ci/default/tree/inst/stepwisefit.m
        """
        regressors = len(X.columns)
        go_on = True
        residuals = y
        counter = 0
        self.k_feature_names_ = []
        n = len(residuals)

        while go_on and counter < self._max_iterations:
            counter += 1
            added = False
            dropped = False

            if len(self.k_feature_names_) < regressors:
                not_in_use = [c not in self.k_feature_names_ for c in X.columns]
                possible_additions = X[:, not_in_use]
                most_correlated = possible_additions.join(residuals).corr()[0, :-1].idxmax(axis="columns")
                column_new = X.columns[most_correlated]
                current_columns = self.k_feature_names_ + [column_new]
                current_data = X.loc[:, current_columns]
                b_new, b_int_new, r_new = self._regress(current_data, y)
                z_new = numpy.abs(b_new[-1] / (b_int_new[-1, 1] - b_new[-1]))

                if z_new > 1:  # which means you accept
                    added = True
                    self.coef_ = b_new
                    b_int = b_int_new
                    residuals = r_new
                    self.k_feature_names_.append(column_new)

            if self.k_feature_names_:
                variables = len(self.k_feature_names_)
                dof = n - variables - 1 if self._add_intercept else n - variables
                t_ratio = stats.t.ppf(1 - self._p_to_discard / 2, dof) / stats.t.ppf(1 - self._p_to_add / 2, dof)

                if self._add_intercept:
                    z = numpy.abs(self.coef_[1:] / (b_int[1:, 1] - self.coef_[1:]))
                else:
                    z = numpy.abs(self.coef_ / (b_int[:, 1] - self.coef_))

                z_min = numpy.min(z, axis=None)
                idx = numpy.argmin(z, axis=None)

                if z_min < t_ratio:
                    dropped = True
                    del self.k_feature_names_[idx]
                    current_data = X.loc[:, self.k_feature_names_]
                    self.coef_, b_int, residuals = self._regress(current_data, y)

            go_on = added or dropped

    def _regress(self, X_df, y_df):
        """
        https://sourceforge.net/p/octave/statistics/ci/default/tree/inst/regress.m
        """
        X = X_df.to_numpy()
        y = y_df.to_numpy()
        n = y.size

        if self._add_intercept:
            X = numpy.c_[numpy.ones(n), X]

        Q, R = linalg.qr(X, mode="economic")
        beta = linalg.solve(R, Q.T.dot(y))
        residuals = y - X.dot(beta)

        _, p = X.shape
        dof = n - p
        SSE = residuals.T.dot(residuals)
        MSE = SSE / dof
        t_alpha_2 = stats.t.ppf(self._p_to_add / 2, dof)
        c = numpy.diag(linalg.inv(R.T.dot(R)))
        # delta is negative, because alpha is small and t_alpha_2 negative
        delta = t_alpha_2 * numpy.sqrt(MSE * c)
        beta_interval = numpy.concatenate((beta + delta, beta - delta), axis=1)

        return beta, beta_interval, residuals

    def predict(self, X_df):
        X = X_df.loc[:, self.k_feature_names_]
        n = len(X.index)

        if self._add_intercept:
            X = numpy.c_[numpy.ones(n), X]

        return numpy.dot(X, self.coef_)
