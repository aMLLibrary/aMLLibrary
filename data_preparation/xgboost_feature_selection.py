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
import sys

import eli5

import data_preparation.data_preparation
import model_building.model_building


class XGBoostFeatureSelection(data_preparation.data_preparation.DataPreparation):
    """
    Step which filters input data according to XGBoost score

    Methods
    -------
    get_name()
        Return the name of this step

    process()
        Select the specified columns
    """

    def __init__(self, campaign_configuration, prefix):
        """
        campaign_configuration: dict of dict:
            The set of options specified by the user though command line and campaign configuration files

        prefix: List[str]
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

        max_features = self._campaign_configuration['FeatureSelection']['max_features']

        model_building_var = model_building.model_building.ModelBuilding(0)

        # setting parameters for XGboost design space expoloration
        xgboost_parameters = copy.deepcopy(self._campaign_configuration)

        xgboost_parameters['General']['techniques'] = ['XGBoost']

        xgboost_parameters['General']['run_num'] = 1

        local_root_directory = self._campaign_configuration['General']['output']
        for token in self._prefix:
            local_root_directory = os.path.join(local_root_directory, token)
        xgboost_parameters['General']['output'] = local_root_directory

        del xgboost_parameters['FeatureSelection']

        if 'XGBoost' not in xgboost_parameters:
            # default parameters if not provided in the ini file
            xgboost_parameters['XGBoost'] = {}
            xgboost_parameters['XGBoost']['min_child_weight'] = [1, 3]
            xgboost_parameters['XGBoost']['gamma'] = [0, 1]
            xgboost_parameters['XGBoost']['n_estimators'] = [50, 100, 150, 250]
            xgboost_parameters['XGBoost']['learning_rate'] = [0.01, 0.05, 0.1]
            xgboost_parameters['XGBoost']['max_depth'] = [1, 2, 3, 5, 9, 13]

        expconfs = model_building_var.process(xgboost_parameters, inputs, int(self._campaign_configuration['General']['j']))

        hp_selection = self._campaign_configuration['General']['hp_selection']

        if hp_selection in {'All', 'HoldOut'}:
            best_conf = None
            # Hyperparameter search
            for conf in expconfs:
                if not best_conf or conf.hp_selection_mape < best_conf.hp_selection_mape:
                    best_conf = conf

        elif hp_selection == "KFold":

            # Aggregate data on different folds

            # for each configuration, the aggregated mape
            conf_hp_selection_mape = {}

            # Hyperparameter search aggregating over different folds on same configuration
            for conf in expconfs:
                configuration = tuple(conf.get_signature()[4:])
                if configuration not in conf_hp_selection_mape:
                    conf_hp_selection_mape[configuration] = 0.0
                conf_hp_selection_mape[configuration] = conf_hp_selection_mape[configuration] + conf.hp_selection_mape

            # Select the best configuration (we did not divide by folds number since it does not change the best)
            best_conf = None
            best_conf_value = 0.0
            for conf in conf_hp_selection_mape:
                if not best_conf or best_conf_value > conf_hp_selection_mape[conf]:
                    best_conf = conf
                    best_conf_value = conf_hp_selection_mape[conf]

            example_conf = None
            for conf in expconfs:
                configuration = tuple(conf.get_signature()[4:])
                if configuration == best_conf:
                    example_conf = conf
                    break

            assert example_conf
            hyperparameters = example_conf.get_hyperparameters()

            best_conf = model_building.xgboost_experiment_configuration.XGBoostExperimentConfiguration(self._campaign_configuration, hyperparameters, inputs, self._prefix)
            best_conf.train()

        else:
            self._logger.error("Unexpected hp selection in XGBoost feature selection: %s", hp_selection)
            sys.exit(1)

        # best_conf is a XGBoost configuration exeperiment
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
