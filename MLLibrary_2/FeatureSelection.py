from SequenceDataProcessing import *
import logging
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import KFold


class FeatureSelection(DataPrepration):
    """This class performs feature selection as the last step of the data preparation"""
    def __init__(self):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, train_features, train_labels, test_features, test_labels, parameters, run_info):
        """calculate how many features are allowed to be selected, then using cross validation searches for the best
        parameters, then trains the model using the best parametrs"""

        # perform grid search
        if parameters['FS']['select_features_sfs']:
            self.logger.info("Grid Search: SFS and Ridge")

            y_pred_train, y_pred_test, run_info = self.gridSearch(train_features, train_labels, test_features, test_labels,
                                                        parameters, run_info)

            # y_pred_train, y_pred_test, run_info = self.Ridge_SFS_GridSearch(train_features, train_labels,
            #                                                                    test_features, test_labels,
            #                                                                    k_features, parameters, run_info)

        if parameters['FS']['XGBoost']:
            self.logger.info("Grid Search: XGBoost")
            print('train_features shape: ', train_features.shape)
            print('test_features shape:  ',test_features.shape )

            y_pred_train, y_pred_test, run_info = self.XGBoost_Gridsearch2(train_features, train_labels,
                                                                          test_features, test_labels,
                                                                          parameters, run_info)
        return y_pred_train, y_pred_test

    def prepareGridInOut(self, train_features, train_labels):

        # obtain the matrix of training data for doing grid search
        X = pd.DataFrame.as_matrix(train_features)
        Y = pd.DataFrame.as_matrix(train_labels)

        return X, Y

    def getFSParams_SFS_Ridge(self, parameters):

        # vector containing parameters to be search in
        alpha_v = parameters['Ridge']['ridge_params']
        param_list = parameters['FS']['SFS_Ridge_param_list']

        # the number of folds in the cross validation of SFS
        fold_num = parameters['FS']['fold_num']
        is_floating = parameters['FS']['is_floating']
        features_names = parameters['Features']['Extended_feature_names']
        min_features = parameters['FS']['min_features']
        max_features = parameters['FS']['max_features']

        # calculate range of features for SFS
        k_features = self.calc_k_features(min_features, max_features, features_names)

        # save k_features:
        parameters['FS']['k_features'] = k_features

        return param_list, alpha_v, k_features, is_floating, fold_num, features_names

    def gridParamRidge_SFS(self, alpha_v):

        param_df = pd.DataFrame(data = alpha_v, index = range(len(alpha_v)), columns = ['alpha'])

        return param_df

    def gridCV(self, X, Y, param_df, k_features, is_floating, fold_num, features_names, run_info):
        cv_info = {}
        cv_info['MSE_overal'] = []
        cv_info['Sel_F_idx'] = []
        cv_info['Sel_F_names'] = []

        # If FS method is SFS_Ridge
        k_fold = KFold(n_splits=fold_num, shuffle=True, random_state=None)

        for i in range(param_df.shape[0]):

            node = float(param_df.iloc[i, :].values)
            nodeMSE = []

            sel_f_set = set()

            # building the sfs
            for k, (train, val) in enumerate(k_fold.split(X, Y)):

                # for feature selection
                model = Ridge(node)

                # for evaluation
                ridge = Ridge(node)

                # building the sfs
                sfs = SFS(clone_estimator=True,
                          estimator=model,
                          k_features=k_features,
                          forward=True,
                          floating=is_floating,
                          scoring='neg_mean_squared_error',
                          cv=0,
                          n_jobs=16)

                # fit the sfs on training part and evaluate the score on test part of this fold
                sfs = sfs.fit(X[train, :], Y[train])
                sel_F_idx = sfs.k_feature_idx_
                # fit the ridge model on training part and evaluate the ridge score on test part of this fold
                # Rows and columns selection should be done in different steps:
                xTRtemp = X[:, sfs.k_feature_idx_]
                ridge.fit(xTRtemp[train, :], Y[train])

                # evaluate the RSE error on test part of this fold
                xVALtemp = X[:, sfs.k_feature_idx_]
                Y_hat = ridge.predict(xVALtemp[val, :])

                nodeMSE.append(mean_squared_error(Y[val], Y_hat))
                sel_f_set = sel_f_set.union(sel_F_idx)


            cv_info['MSE_overal'].append(np.mean(nodeMSE))
            cv_info['Sel_F_idx'].append(sorted(list(sel_f_set)))
            cv_info['Sel_F_names'].append(features_names[sorted(list(sel_f_set))])

            print('parameter: ', node, ' ------- sel_F: ', features_names[sorted(list(sel_f_set))], '------- MSE: ', np.mean(nodeMSE))

        best_param_idx = cv_info['MSE_overal'].index(min(cv_info['MSE_overal']))
        run_info[-1]['cv_info'] = cv_info
        return best_param_idx, cv_info

    def get_best_param(self, best_param_idx, param_df, param_list, run_info):

        best_param_node = param_df.iloc[best_param_idx]
        best_param = {}
        for i in range(len(param_list)):
            best_param[param_df.columns.values[i]] = best_param_node[i]
        run_info[-1]['best_param'] = best_param
        return best_param

    def get_selected_features(self, cv_info, best_param_idx, run_info):

        selected_features_idx = cv_info['Sel_F_idx'][best_param_idx]
        selected_features_names = cv_info['Sel_F_names'][best_param_idx]
        run_info[-1]['Sel_features'] = selected_features_idx
        run_info[-1]['Sel_features_names'] = selected_features_names
        return selected_features_idx, selected_features_names

    def predict_ridge(self, best_param, selected_features_idx, train_features, train_labels, test_features,
                      test_labels, run_info):

        # Since the data for classsifier selection is too small, we only calculate the train error
        X_train = pd.DataFrame.as_matrix(train_features)
        Y_train = pd.DataFrame.as_matrix(train_labels)
        X_test = pd.DataFrame.as_matrix(test_features)
        Y_test = pd.DataFrame.as_matrix(test_labels)

        param_list = list(best_param.keys())

        best_alpha = best_param[param_list[0]]

        best_trained_model = Ridge(best_alpha)
        best_trained_model.fit(X_train[:, selected_features_idx], Y_train)
        y_pred_train = best_trained_model.predict(X_train[:, selected_features_idx])
        y_pred_test = best_trained_model.predict(X_test[:, selected_features_idx])
        run_info[-1]['best_model'] = best_trained_model
        run_info[-1]['scaled_y_pred_train'] = y_pred_train
        run_info[-1]['scaled_y_pred_test'] = y_pred_test

        return y_pred_train, y_pred_test


    def gridSearch(self, train_features, train_labels, test_features, test_labels, parameters, run_info):

        # prepare Input and output
        X, Y = self.prepareGridInOut(train_features, train_labels)

        # get FS parameters:
        # if FS method is SFS_Ridge
        param_list, alpha_v, k_features, is_floating, fold_num, features_names = self.getFSParams_SFS_Ridge(parameters)
        # print('param_list:', param_list)

        # Prepare Gridsearch parameters according to the type of feature selection:
        # if FS method is SFS_Ridge
        param_df = self.gridParamRidge_SFS(alpha_v)
        # print('param_df:', param_df)

        # If FS method is SFS_Ridge??? we should put it in grid search? yes
        best_param_idx, cv_info = self.gridCV(X, Y, param_df, k_features, is_floating, fold_num, features_names, run_info)

        # find the best parameters
        best_param = self.get_best_param(best_param_idx, param_df, param_list, run_info)
        # print('best_param:', best_param)

        # find the selected features
        selected_features_idx, selected_features_names = self.get_selected_features(cv_info, best_param_idx, run_info)

        # predict using best params:
        y_pred_train, y_pred_test = self.predict_ridge(best_param, selected_features_idx, train_features, train_labels,
                                                        test_features, test_labels, run_info)

        return y_pred_train, y_pred_test, run_info

    def getFSParams_XGBoost(self, parameters):
        features_names = parameters['Features']['Extended_feature_names']
        param_list = parameters['XGBoost']['grid_elements']
        learning_rate_v = parameters['XGBoost']['learning_rate_v']
        n_estimators_v = parameters['XGBoost']['n_estimators_v']
        reg_lambda_v = parameters['XGBoost']['reg_lambda_v']
        min_child_weight_v = parameters['XGBoost']['min_child_weight_v']
        max_depth_v = parameters['XGBoost']['max_depth_v']
        fold_num = parameters['FS']['fold_num']

        return param_list, learning_rate_v, reg_lambda_v, min_child_weight_v, max_depth_v, fold_num, features_names

    def gridParamXGBoost(self, param_list, learning_rate_v, reg_lambda_v, min_child_weight_v, max_depth_v):
        param_df = pd.DataFrame(0, index=range(len(learning_rate_v) * len(reg_lambda_v) * len(min_child_weight_v) *
                                               len(max_depth_v)), columns=param_list)
        row = 0
        for l in learning_rate_v:
            for rl in reg_lambda_v:
                for mw in min_child_weight_v:
                    for md in max_depth_v:
                        param_df.iloc[row, :] = [l, rl, mw, md]
                        row += 1
        return param_df

    def gridCVXGBoost(self, X, Y, train_data_dmatrix, param_df, fold_num, run_info):

        cv_info = {}
        cv_info['MSE_overal'] = []

        for node in range(param_df.shape[0]):
            node_param = param_df.iloc[node, :]
            [l, rl, mw, md] = param_df.iloc[node, :]

            xgboost_params = {"silent" : 1, "learning_rate": l, 'reg_lambda': rl, 'min_child_weight': mw ,
                              'max_depth': int(md)}

            cv_results = xgb.cv(params=xgboost_params, dtrain=train_data_dmatrix, nfold=fold_num,
                                num_boost_round=100, early_stopping_rounds=10, metrics="rmse",
                                verbose_eval=None, as_pandas=True, seed=123)

            cv_info['MSE_overal'].append((cv_results["test-rmse-mean"].iloc[-1])**2)

        best_param_idx = cv_info['MSE_overal'].index(min(cv_info['MSE_overal']))
        run_info[-1]['cv_info'] = cv_info
        return best_param_idx, cv_info

    def predict_XGBoost(self, best_param, train_features, train_labels, test_features, test_labels, run_info):
        xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, params=best_param, verbosity=0)
        xg_reg.fit(train_features, train_labels)

        y_pred_train = xg_reg.predict(train_features)

        y_pred_test = xg_reg.predict(test_features)

        run_info[-1]['best_model'] = xg_reg
        run_info[-1]['scaled_y_pred_train'] = y_pred_train
        run_info[-1]['scaled_y_pred_test'] = y_pred_test
        run_info[-1]['fscore'] = xg_reg.feature_importances_
        run_info[-1]['names_list'] = train_features.columns.values

        return y_pred_train, y_pred_test

    def XGBoost_Gridsearch2(self, train_features, train_labels, test_features, test_labels, parameters, run_info):

        # obtain the matrix of training data for doing grid search
        X, Y = self.prepareGridInOut(train_features, train_labels)

        train_data_dmatrix = xgb.DMatrix(data=train_features, label=train_labels)

        param_list, learning_rate_v, reg_lambda_v, min_child_weight_v, max_depth_v, fold_num, features_names =\
            self.getFSParams_XGBoost(parameters)

        param_df = self.gridParamXGBoost(param_list, learning_rate_v, reg_lambda_v, min_child_weight_v, max_depth_v)

        best_param_idx, cv_info = self.gridCVXGBoost(X, Y, train_data_dmatrix, param_df, fold_num, run_info)

        best_param = self.get_best_param(best_param_idx, param_df, param_list, run_info)

        y_pred_train, y_pred_test = self.predict_XGBoost(best_param, train_features, train_labels,
                                                                    test_features, test_labels, run_info)

        return y_pred_train, y_pred_test, run_info


    def XGBoost_Gridsearch(self, train_features, train_labels, test_features, test_labels, parameters, run_info):

        # obtain the matrix of training data for doing grid search
        X = pd.DataFrame.as_matrix(train_features)
        Y = pd.DataFrame.as_matrix(train_labels)
        ext_feature_names = train_features.columns.values

        train_data_dmatrix = xgb.DMatrix(data=train_features, label=train_labels)


        learning_rate_v = parameters['XGBoost']['learning_rate_v']
        n_estimators_v = parameters['XGBoost']['n_estimators_v']
        reg_lambda_v = parameters['XGBoost']['reg_lambda_v']
        min_child_weight_v = parameters['XGBoost']['min_child_weight_v']
        max_depth_v = parameters['XGBoost']['max_depth_v']
        grid_elements = ['learning_rate_v', 'reg_lambda_v', 'min_child_weight_v', 'max_depth_v']

        fold_num = parameters['FS']['fold_num']

        param_overal_MSE = []

        param_grid = pd.DataFrame(0, index=range(len(learning_rate_v) * len(reg_lambda_v) * len(min_child_weight_v) *
                                                 len(max_depth_v)), columns=grid_elements)

        cv_info = {}
        row = 0
        for l in learning_rate_v:
            for rl in reg_lambda_v:
                for mw in min_child_weight_v:
                    for md in max_depth_v:

                        param_grid.iloc[row, :] = [l, rl, mw, md]
                        xgboost_params = {"silent" : 1, "learning_rate": l, 'reg_lambda': rl, 'min_child_weight': mw ,
                                          'max_depth': md}
                        cv_info[str(row)] = {}

                        cv_results = xgb.cv(params=xgboost_params, dtrain=train_data_dmatrix, nfold=fold_num,
                                            num_boost_round=100, early_stopping_rounds=10, metrics="rmse",
                                            verbose_eval=None, as_pandas=True, seed=123)

                        param_overal_MSE.append(cv_results["test-rmse-mean"].iloc[-1])
                        cv_info[str(row)]['MSE'] = cv_results["test-rmse-mean"].iloc[-1]
                        row += 1

        MSE_best = param_overal_MSE.index(min(param_overal_MSE))
        print(MSE_best)

        learning_rate, reg_lambda, min_child_weight, max_depth = param_grid.iloc[MSE_best, :]
        best_params = {"learning_rate": learning_rate, 'reg_lambda': int(reg_lambda), 'min_child_weight':
                                                        int(min_child_weight), 'max_depth': int(max_depth)}

        xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, params=best_params, verbosity = 0)
        xg_reg.fit(train_features, train_labels)

        y_pred_train = xg_reg.predict(train_features)
        train_rmse = np.sqrt(mean_squared_error(train_labels, y_pred_train))
        train_mse = mean_squared_error(train_labels, y_pred_train)


        y_pred_test = xg_reg.predict(test_features)
        test_rmse = np.sqrt(mean_squared_error(test_labels, y_pred_test))
        test_mse = mean_squared_error(test_labels, y_pred_test)


        run_info[-1]['cv_info'] = cv_info
        # run_info[-1]['Sel_features'] = list(sel_idx)
        # run_info[-1]['Sel_features_names'] = [features_names[i] for i in sel_idx]
        run_info[-1]['best_param'] = best_params
        run_info[-1]['best_model'] = xg_reg
        run_info[-1]['scaled_y_pred_train'] = y_pred_train
        run_info[-1]['scaled_y_pred_test'] = y_pred_test
        run_info[-1]['names_list'] = train_features.columns.values
        run_info[-1]['fscore'] = xg_reg.feature_importances_

        return y_pred_train, y_pred_test, run_info


    def Ridge_SFS_GridSearch(self, train_features, train_labels, test_features, test_labels, k_features, parameters, run_info):
        """select the best parameres using CV and sfs feature selection"""

        # obtain the matrix of training data for doing grid search
        X = pd.DataFrame.as_matrix(train_features)
        Y = pd.DataFrame.as_matrix(train_labels)
        ext_feature_names = train_features.columns.values
        features_names = parameters['Features']['Extended_feature_names']

        # vector containing parameters to be search in
        alpha_v = parameters['Ridge']['ridge_params']

        # the number of folds in the cross validation of SFS
        fold_num = parameters['FS']['fold_num']

        # list of MSE error for different alpha values:
        param_overal_MSE = []

        # list of MAPE error for different alpha values:
        Mape_overal_error = []

        # Dictionary keeping information about all scores and values and selected features in all iterations for all params
        cv_info = {}

        # Selected features for each alpha
        sel_F = []

        # Selected features names for each alpha
        sel_F_names = []

        for a in alpha_v:
            # building the models
            ridge = Ridge(a)
            model = Ridge(a)
            this_a = 'alpha = ' + str(a)
            cv_info[this_a] = {}

            # building the sfs
            sfs = SFS(clone_estimator=True,
                      estimator=model,
                      k_features=k_features,
                      forward=True,
                      floating=False,
                      scoring='neg_mean_squared_error',
                      cv=fold_num,
                      n_jobs=16)

            # fit the sfs on training part (scaled) and evaluate the score on test part of this fold
            sfs = sfs.fit(X, Y)
            sel_F_idx = sfs.k_feature_idx_
            sel_F.append(sel_F_idx)
            cv_info[this_a]['Selected_Features'] = list(sel_F_idx)

            sel_F_names.append(ext_feature_names[list(sel_F_idx)].tolist())
            cv_info[this_a]['Selected_Features_Names'] = ext_feature_names[list(sel_F_idx)].tolist()

            # fit the ridge model with the scaled version of the selected features
            ridge.fit(X[:, sfs.k_feature_idx_], Y)

            # evaluate the MSE error on the whole (scaled) training data only using the selected features
            Y_hat = ridge.predict(X[:, sfs.k_feature_idx_])
            MSE = self.calcMSE(Y_hat, Y)
            param_overal_MSE.append(MSE)
            cv_info[this_a]['MSE_error'] = MSE

            # evaluate the MAPE error on the whole training data only using the selected features
            Mape_error = self.calcMAPE(Y_hat, Y)
            Mape_overal_error.append(Mape_error)
            cv_info[this_a]['MAPE_error'] = Mape_error
            print('alpha = ', a, '     MSE Error= ', MSE, '    MAPE Error = ', Mape_error, '    Ridge Coefs= ',
                  ridge.coef_, '     Intercept = ', ridge.intercept_, '     SEL = ', ext_feature_names[list(sel_F_idx)])

        # get the results:
        # select the best alpha based on obtained values
        MSE_index = param_overal_MSE.index(min(param_overal_MSE))

        # report the best alpha based on obtained values
        print('Least_MSE_Error_index = ', MSE_index, ' => Least_RSE_Error_alpha = ', alpha_v[MSE_index])
        Least_MSE_alpha = alpha_v[MSE_index]
        best_trained_model = Ridge(Least_MSE_alpha)
        best_trained_model.fit(X[:, sel_F[MSE_index]], Y)
        sel_idx = sel_F[MSE_index]

        # Since the data for classsifier selection is too small, we only calculate the train error
        X_train = pd.DataFrame.as_matrix(train_features)
        Y_train = pd.DataFrame.as_matrix(train_labels)
        X_test = pd.DataFrame.as_matrix(test_features)
        Y_test = pd.DataFrame.as_matrix(test_labels)

        # if data_conf["input_name"] == "classifierselection":
        #     y_pred_test = []
        # else:
        #     y_pred_test = best_trained_model.predict(X_test[:, sel_idx])

        y_pred_test = best_trained_model.predict(X_test[:, sel_idx])
        y_pred_train = best_trained_model.predict(X_train[:, sel_idx])

        # populate obtained values in the run_info variable
        run_info[-1]['cv_info'] = cv_info
        run_info[-1]['Sel_features'] = list(sel_idx)
        run_info[-1]['Sel_features_names'] = [features_names[i] for i in sel_idx]
        run_info[-1]['best_param'] = Least_MSE_alpha
        run_info[-1]['best_model'] = best_trained_model
        run_info[-1]['scaled_y_pred_train'] = y_pred_train
        run_info[-1]['scaled_y_pred_test'] = y_pred_test

        return y_pred_train, y_pred_test, run_info


    def calc_k_features(self, min_features, max_features, features_names):
        """calculate the range of number of features that sfs is allowed to select"""

        # Selecting from all features
        if max_features == -1:
            k_features = (min_features, len(features_names))
            # Selecting from the given range
        if max_features != -1:
            k_features = (min_features, max_features)
        return k_features

    def calcMSE(self, Y_hat, Y):
        MSE = np.mean((Y_hat - Y) ** 2)
        return MSE

    def calcMAPE(self, Y_hat, Y):
        """given true and predicted values returns MAPE error"""
        Mapeerr = np.mean(np.abs((Y - Y_hat) / Y)) * 100
        return Mapeerr