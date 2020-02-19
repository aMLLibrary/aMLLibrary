"""
Copyright 2019 Marco Lattuada, Danilo Ardagna

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

import copy
import os

import eli5

import data_preparation.data_preparation
import model_building.model_building


class XGBoostFeatureSelection(data_preparation.data_preparation.DataPreparation):
    """
    Step which filters input data according to XGBoost score

    This step is integrated in the generators since it is performed on the different training set; it internally execute a regression flow using only XGBoost as technique

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Select the specified columns
    """

    def __init__(self, campaign_configuration, prefix):
        """
        campaign_configuration: dict of str: dict of str: str
            The set of options specified by the user though command line and campaign configuration files

        prefix: list of str
            The list of generators after which XGBoostFeatureSelectionExpConfsGenerator is plugged
        """
        self._prefix = prefix
        super().__init__(campaign_configuration)

    def get_name(self):
        """
        Return "XGBoostFeatureSelection"

        Returns
        string
            The name of this step
        """
        return "XGBoostFeatureSelection"

    def process(self, inputs):
        """
        Main method of the class

        This method creates an ad-hoc regression flow composed of only XGBoost; if its parameters are not provided, default are used.
        At the end information about the score of the single features are extracted from the regressor.
        Features are selected according to their relevance until the selected expected cumulative tolerance is reached

        Parameters
        ----------
        inputs: RegressionInputs
            The data to be analyzed
        """

        max_features = self._campaign_configuration['FeatureSelection']['max_features']

        # setting parameters for XGboost design space explooration
        xgboost_parameters = copy.deepcopy(self._campaign_configuration)

        xgboost_parameters['General']['techniques'] = ['XGBoost']

        xgboost_parameters['General']['run_num'] = 1

        local_root_directory = self._campaign_configuration['General']['output']
        for token in self._prefix:
            local_root_directory = os.path.join(local_root_directory, token)
        xgboost_parameters['General']['output'] = local_root_directory

        del xgboost_parameters['FeatureSelection']

        model_building_var = model_building.model_building.ModelBuilding(0)

        if 'XGBoost' not in xgboost_parameters:
            # default parameters if not provided in the ini file
            xgboost_parameters['XGBoost'] = {}
            xgboost_parameters['XGBoost']['min_child_weight'] = [1, 3]
            xgboost_parameters['XGBoost']['gamma'] = [0, 1]
            xgboost_parameters['XGBoost']['n_estimators'] = [50, 100, 150, 250]
            xgboost_parameters['XGBoost']['learning_rate'] = [0.01, 0.05, 0.1]
            xgboost_parameters['XGBoost']['max_depth'] = [1, 2, 3, 5, 9, 13]

        best_conf = model_building_var.process(xgboost_parameters, inputs, int(self._campaign_configuration['General']['j']))

        # best_conf is a XGBoost configuration experiment
        xgb_regressor = best_conf.get_regressor()

        # top = None means all
        expl = eli5.xgboost.explain_weights_xgboost(xgb_regressor, feature_names=inputs.x_columns, top=max_features, importance_type='gain')

        # text version
        expl_weights = eli5.format_as_text(expl)

        self._logger.debug("XGBoost feature scores:\n%s", str(expl_weights))

        df = eli5.format_as_dataframe(expl)  # data frame version

        xgb_sorted_features = df['feature'].values.tolist()  # features list

        features_sig = df['weight'].values.tolist()  # significance score weights

        cumulative_significance = 0

        tolerance = self._campaign_configuration['FeatureSelection']['XGBoost_tolerance']

        index = 0

        while cumulative_significance < tolerance and index < len(features_sig):
            cumulative_significance = cumulative_significance + features_sig[index]
            index = index + 1

        feat_res = xgb_sorted_features[0:index]

        self._logger.info("XGBoost selected features: %s", str(feat_res))

        data = inputs
        data.x_columns = feat_res

        return data
