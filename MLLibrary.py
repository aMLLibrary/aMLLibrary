import logging
from sklearn.preprocessing import StandardScaler
import pandas as pd
import configparser as cp
import numpy as np
import ast
import itertools
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error
import xgboost as xgb


class SequenceDataProcessing(object):
    """ main class """
    def __init__(self):
        """algorithm related parameters to be get from params.ini file"""
        self.conf = cp.ConfigParser()
        self.parameters = {}
        self.get_parameters()

        self.debug = self.parameters['DebugLevel']['debug']
        logging.basicConfig(level=logging.DEBUG) if self.debug else logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # data dictionary storing independent runs information
        self.run_info = []

        # number of independent runs
        self.run_num = self.parameters['General']['run_num']

        # main folder for saving the results
        self.result_path = self.parameters['DataPreparation']['result_path']

        # number of folds in sfs cross validation
        self.fold_num = self.parameters['FS']['fold_num']

        # input name
        self.input_name = self.parameters['DataPreparation']['input_name']

        # creating object of classes
        self.feature_selection = FeatureSelection()
        self.regression = Regression()
        self.results = Results()
        self.preliminary_data_processing = PreliminaryDataProcessing("P8_kmeans.csv")
        self.data_preprocessing = DataPreprocessing()
        self.data_splitting = Splitting()
        self.normalization = Normalization()

    def get_parameters(self):
        """Gets the parameters from the config file named parameters.ini and put them into a dictionary
        named parameters"""
        self.conf.read('params.ini')

        self.parameters['DebugLevel'] = {}
        self.parameters['DebugLevel']['debug'] = ast.literal_eval(self.conf['DebugLevel']['debug'])

        self.parameters['General'] = {}
        self.parameters['General']['run_num'] = int(self.conf['General']['run_num'])

        self.parameters['DataPreparation'] = {}
        self.parameters['DataPreparation']['result_path'] = self.conf['DataPreparation']['result_path']
        self.parameters['DataPreparation']['input_name'] = self.conf.get('DataPreparation', 'input_name')
        self.parameters['DataPreparation']['input_path'] = self.conf.get('DataPreparation', 'input_path')
        self.parameters['DataPreparation']['target_column'] = int(self.conf['DataPreparation']['target_column'])
        self.parameters['DataPreparation']['split'] = self.conf.get('DataPreparation', 'split')
        self.parameters['DataPreparation']['case'] = self.conf.get('DataPreparation', 'case')
        self.parameters['DataPreparation']['use_spark_info'] = self.conf.get('DataPreparation', 'use_spark_info')
        self.parameters['DataPreparation']['irrelevant_column_name'] = self.conf.get('DataPreparation', 'irrelevant_column_name')

        self.parameters['FS'] = {}
        self.parameters['FS']['select_features_vif'] = bool(ast.literal_eval(self.conf.get('FS', 'select_features_vif')))
        self.parameters['FS']['select_features_sfs'] = bool(ast.literal_eval(self.conf.get('FS', 'select_features_sfs')))
        print(self.parameters['FS']['select_features_sfs'])
        self.parameters['FS']['XGBoost'] = bool(ast.literal_eval(self.conf.get('FS', 'XGBoost')))
        print(self.parameters['FS']['XGBoost'])

        self.parameters['FS']['min_features'] = int(self.conf['FS']['min_features'])
        self.parameters['FS']['max_features'] = int(self.conf['FS']['max_features'])
        self.parameters['FS']['is_floating'] = bool(ast.literal_eval(self.conf.get('FS', 'is_floating')))
        self.parameters['FS']['fold_num'] = int(self.conf['FS']['fold_num'])
        self.parameters['FS']['Confidence_level'] = self.conf['FS']['Confidence_level']
        self.parameters['FS']['clipping_no'] = int(self.conf['FS']['clipping_no'])
        self.parameters['FS']['degree'] = int(self.conf['FS']['degree'])


        self.parameters['XGBoost'] = {}
        self.parameters['XGBoost']['learning_rate_v'] = self.conf['XGBoost']['learning_rate_v']
        self.parameters['XGBoost']['learning_rate_v'] = [i for i in ast.literal_eval(self.parameters['XGBoost']['learning_rate_v'])]

        self.parameters['XGBoost']['reg_lambda_v'] = self.conf['XGBoost']['reg_lambda_v']
        self.parameters['XGBoost']['reg_lambda_v'] = [int(i) for i in ast.literal_eval(self.parameters['XGBoost']['reg_lambda_v'])]


        self.parameters['XGBoost']['n_estimators_v'] = self.conf['XGBoost']['n_estimators_v']
        self.parameters['XGBoost']['n_estimators_v'] = [int(i) for i in ast.literal_eval(self.parameters['XGBoost']['n_estimators_v'])]


        self.parameters['XGBoost']['min_child_weight_v'] = self.conf['XGBoost']['min_child_weight_v']
        self.parameters['XGBoost']['min_child_weight_v'] = [int(i) for i in ast.literal_eval(self.parameters['XGBoost']['min_child_weight_v'])]


        self.parameters['XGBoost']['max_depth_v'] = self.conf['XGBoost']['max_depth_v']
        self.parameters['XGBoost']['max_depth_v'] = [int(i) for i in ast.literal_eval(self.parameters['XGBoost']['max_depth_v'])]


        self.parameters['XGBoost']['grid_elements'] = self.conf['XGBoost']['grid_elements']
        self.parameters['XGBoost']['grid_elements'] = self.conf.get('XGBoost', 'grid_elements')
        self.parameters['XGBoost']['grid_elements'] = [i for i in ast.literal_eval(self.parameters['XGBoost']['grid_elements'])]


        self.parameters['Ridge'] = {}
        self.parameters['Ridge']['ridge_params'] = self.conf['Ridge']['ridge_params']
        self.parameters['Ridge']['ridge_params'] = [i for i in ast.literal_eval(self.parameters['Ridge']['ridge_params'])]

        self.parameters['Lasso'] = {}
        self.parameters['Lasso']['lasso_params'] = self.conf['Lasso']['lasso_params']
        self.parameters['Lasso']['lasso_params'] = [i for i in ast.literal_eval(self.parameters['Lasso']['lasso_params'])]

        self.parameters['Splitting'] = {}
        self.parameters['Splitting']['seed_vector'] = self.conf['Splitting']['seed_vector']
        self.parameters['Splitting']['seed_vector'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['seed_vector'])]

        self.parameters['Splitting']['image_nums_train_data'] = self.conf['Splitting']['image_nums_train_data']
        self.parameters['Splitting']['image_nums_train_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['image_nums_train_data'])]

        self.parameters['Splitting']['image_nums_test_data'] = self.conf.get('Splitting', 'image_nums_test_data')
        self.parameters['Splitting']['image_nums_test_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['image_nums_test_data'])]

        self.parameters['Splitting']['core_nums_train_data'] = self.conf.get('Splitting', 'core_nums_train_data')
        self.parameters['Splitting']['core_nums_train_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['core_nums_train_data'])]

        self.parameters['Splitting']['core_nums_test_data'] = self.conf.get('Splitting', 'core_nums_test_data')
        self.parameters['Splitting']['core_nums_test_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['core_nums_test_data'])]

        self.parameters['Splitting']['criterion_col_list'] = self.conf.get('Splitting', 'criterion_col_list')
        self.parameters['Splitting']['criterion_col_list'] = [i for i in ast.literal_eval(self.parameters['Splitting']['criterion_col_list'])]

        self.parameters['Inverse'] = {}
        self.parameters['Inverse']['to_be_inv_List'] = [str(self.conf['Inverse']['to_be_inv_List'])]

        self.parameters['Regression'] = {}
        self.parameters['Regression']['regressor_name'] = str(self.conf['Regression']['regressor_name'])

        self.parameters['dt'] = {}
        self.parameters['dt']['max_features_dt'] = str(self.conf['dt']['max_features_dt'])
        self.parameters['dt']['max_depth_dt'] = str(self.conf['dt']['max_depth_dt'])
        self.parameters['dt']['min_samples_leaf_dt'] = str(self.conf['dt']['min_samples_leaf_dt'])
        self.parameters['dt']['min_samples_split_dt'] = str(self.conf['dt']['min_samples_split_dt'])

    def process(self):
        """the main code"""

        self.logger.info("Start of the algorithm")
        # performs reading data, drops irrelevant columns
        df = self.preliminary_data_processing.process(self.parameters)

        # performs inverting of the columns and adds combinatorial terms to the df
        ext_df = self.data_preprocessing.process(df, self.parameters)

        # performs the algorithm multiple time and each time changes the seed to shuffle
        for iter in range(self.run_num):

            this_run = 'run_' + str(iter)
            print('==================================================================================================')
            print(this_run)

            # the list containing all the information about runs of algorithm, for each run a dictionary is added
            # to the list
            self.run_info.append({})

            # performs data splitting and returns splitted data
            train_features, train_labels, test_features, test_labels  = \
                    self.data_splitting.process(ext_df, self.parameters, self.run_info)

            # does the feature selection using training data and finds the best parameters in a grid search, then
            # predicts the output of test data
            y_pred_train, y_pred_test = self.feature_selection.process(train_features,
                                                                       train_labels,
                                                                       test_features,
                                                                       test_labels,
                                                                       self.parameters,
                                                                       self.run_info)

            # computes the mean absolute percentage error for training and test set, also returns prediction and true
            # values of application completion time for future computations
            err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test = \
                                        self.regression.process(y_pred_test, y_pred_train,
                                                                test_features, test_labels,
                                                                train_features,
                                                                train_labels,
                                                                self.parameters,
                                                                self.run_info)

            # save the run_info variable as string in a temporary file in the result folder
            self.results.save_temporary_results(self.run_info)

        # saves the best run results and necessary plots in the defined folder in result directory
        self.results.process(ext_df, self.run_info, self.parameters)


class Task(object):
    def __init__(self):
        self.inputDF = None  # Check with Marco this is a DF, I would create an empy DF
        self.outputDF = None


class DataPrepration(Task):
    """This is the main class defining the pipeline of machine learning task"""

    def __init__(self):
        Task.__init__(self)


class Normalization(DataPrepration):

    def __init__(self):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, inputDF):
        """Normalizes the data using StandardScaler module"""
        self.logger.info("Scaling: ")
        self.inputDF = inputDF
        self.outputDF, scaler = self.scale_data(self.inputDF)
        return self.outputDF, scaler

    def scale_data(self, df):
        """scale the dataframe"""

        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df.values)
        scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
        return scaled_df, scaler

    def denormalize_data(self, scaled_df, scaler):
        """descale the dataframe"""

        array = scaler.inverse_transform(scaled_df)
        df = pd.DataFrame(array, index=scaled_df.index, columns=scaled_df.columns)
        return df


class PreliminaryDataProcessing(DataPrepration):
    """Perform preliminary prossing of data"""
    def __init__(self, input_file):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, parameters):
        """Get the csv file, drops the irrelevant columns and change it to data frame as output"""
        input_path = parameters['DataPreparation']['input_path']
        self.logger.info("Input reading: " + input_path)
        self.outputDF = pd.read_csv(input_path)

        # drop the run column
        dropping_col = parameters['DataPreparation']['irrelevant_column_name']
        self.outputDF = self.outputDF.drop(dropping_col, axis=1)

        # manually decrease the number of target column since we dropped the first columns, this maybe changed in
        # different input files
        parameters['DataPreparation']['target_column'] -= 1

        # drop constant columns
        self.outputDF = self.outputDF.loc[:, (self.outputDF != self.outputDF.iloc[0]).any()]
        return self.outputDF


class DataPreprocessing(DataPrepration):
    """performs invesing of needed features and adds them to the dataframe, extends the dataframe to a degree"""
    def __init__(self):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, inputDF, parameters):
        """inversing and extension of features in the dataframe"""
        self.inputDF = inputDF

        to_be_inv_list = parameters['Inverse']['to_be_inv_List']
        target_column = parameters['DataPreparation']['target_column']
        degree = parameters['FS']['degree']

        self.logger.info("Inverting relevant features: ")
        # add the inverted column(s)
        self.outputDF, inversing_cols = self.add_inverse_features(self.inputDF, to_be_inv_list)
        parameters['Inverse']['inversing_cols'] = inversing_cols

        self.logger.info("Extending features: ")
        # add combinatorial terms
        self.outputDF = self.add_all_comb(self.outputDF, inversing_cols, target_column, degree)

        # populate parameter variable with newly computed variables
        parameters['Features'] = {}
        parameters['Features']['Extended_feature_names'] = self.outputDF.columns.values[1:]
        parameters['Features']['Original_feature_names'] = self.inputDF.columns.values[1:]
        return self.outputDF

    def add_inverse_features(self, df, to_be_inv_list):
        """Given a dataframe and the name of columns that should be inversed, add the needed inversed columns and
        returns the resulting df and the indices of two reciprocals separately"""

        # a dictionary of DataFrame used for adding the inverted columns
        df_dict = dict(df)
        for c in to_be_inv_list:
            new_col = 1 / np.array(df[c])
            new_feature_name = 'inverse_' + c
            df_dict[new_feature_name] = new_col

        # convert resulting dictionary to DataFrame
        inv_df = pd.DataFrame.from_dict(df_dict)

        # returns the indices of the columns that should be inverted and their inverted in one tuple
        inverting_cols = []
        for c in to_be_inv_list:
            cidx = inv_df.columns.get_loc(c)
            cinvidx = inv_df.columns.get_loc('inverse_' + c)
            inv_idxs = (cidx, cinvidx)
            inverting_cols.append(inv_idxs)
        return inv_df, inverting_cols

    def add_all_comb(self, inv_df, inversed_cols_tr, output_column_idx, degree):
        """Given a dataframe, returns an extended df containing all combinations of columns except the ones that are
        inversed"""

        # obtain needed parameters for extending DataFrame
        features_names = inv_df.columns.values
        df_dict = dict(inv_df)
        data_matrix = pd.DataFrame.as_matrix(inv_df)
        indices = list(range(data_matrix.shape[1]))

        # compute all possible combinations with replacement
        for j in range(2, degree + 1):
            combs = list(itertools.combinations_with_replacement(indices, j))
            # finds the combinations containing features and inverted of them
            remove_list_idx = []
            for ii in combs:
                for kk in inversed_cols_tr:
                    if len(list(set.intersection(set(ii), set(kk)))) >= 2:
                        remove_list_idx.append(ii)
                if output_column_idx in ii:
                    remove_list_idx.append(ii)
            # removes the combinations containing features and inverted of them
            for r in range(0,len(remove_list_idx)):
                combs.remove(remove_list_idx[r])
            # compute resulting column of the remaining combinations and add to the df
            for cc in combs:
                new_col = self.calculate_new_col(data_matrix, list(cc))
                new_feature_name = ''
                for i in range(len(cc)-1):
                    new_feature_name = new_feature_name+features_names[cc[i]]+'_'
                new_feature_name = new_feature_name+features_names[cc[i+1]]

                # adding combinations each as a column to a dictionary
                df_dict[new_feature_name] = new_col
        # convert the dictionary to a dataframe
        ext_df = pd.DataFrame.from_dict(df_dict)
        return ext_df

    def calculate_new_col(self, X, indices):
        """Given two indices and input matrix returns the multiplication of the columns corresponding columns"""
        index = 0
        new_col = X[:, indices[index]]
        for ii in list(range(1, len(indices))):
            new_col = np.multiply(new_col, X[:, indices[index + 1]])
            index += 1
        return new_col


class Splitting(DataPrepration):
    """performs splitting of the data based on the input parameters and scaling them """
    def __init__(self):
        DataPrepration.__init__(self)
        self.normalization = Normalization()
        self.logger = logging.getLogger(__name__)

    def process(self, inputDF, parameters, run_info):
        """performs scaling and splitting"""

        self.logger.info("Splitting dataset: ")
        self.inputDF = inputDF

        # retrieve the seed and shuffle
        seed = parameters['Splitting']['seed_vector'][len(run_info)-1]
        self.inputDF = shuffle(self.inputDF, random_state=seed)

        # find the samples in training and test set
        train_indices, test_indices = self.getTRTEindices(self.inputDF, parameters)

        # populate the run_info variable
        run_info[-1]['train_indices'] = train_indices
        run_info[-1]['test_indices'] = test_indices

        # scale the data
        scaled_inputDF, scaler = self.normalization.process(self.inputDF)

        # split the data
        train_df = scaled_inputDF.ix[train_indices]
        test_df = scaled_inputDF.ix[test_indices]
        train_labels = train_df.iloc[:, 0]
        train_features = train_df.iloc[:, 1:]

        test_labels = test_df.iloc[:, 0]
        test_features = test_df.iloc[:, 1:]

        # populate the necessary variables in run_info for later use
        run_info[-1]['scaler'] = scaler
        run_info[-1]['train_features'] = train_features
        run_info[-1]['train_labels'] = train_labels
        run_info[-1]['test_features'] = test_features
        run_info[-1]['test_labels'] = test_labels

        return train_features, train_labels, test_features, test_labels

    def getTRTEindices(self, df, parameters):
        """find training and test indices in the data based on the datasize and core numbers"""

        # retrieve needed parameters
        image_nums_train_data = parameters['Splitting']['image_nums_train_data']
        image_nums_test_data = parameters['Splitting']['image_nums_test_data']
        core_nums_train_data = parameters['Splitting']['core_nums_train_data']
        core_nums_test_data = parameters['Splitting']['core_nums_test_data']

        # group the samples according to the dataSize column of dataset since this column is one the features that tell
        # with samples are in training set and which are in the test set
        data_size_indices = pd.DataFrame(
            [[k, v.values] for k, v in df.groupby('dataSize').groups.items()], columns=['col', 'indices'])

        # save the samples that can be possible in the training(test) set because their dataSize column value
        # correspond to the predefined value from the input parameters
        data_size_train_indices = \
            data_size_indices.loc[(data_size_indices['col'].isin(image_nums_train_data))]['indices']
        data_size_test_indices = \
            data_size_indices.loc[(data_size_indices['col'].isin(image_nums_test_data))]['indices']

        # gather all the candidate samples from above into a list for training(test) set
        data_size_train_indices = np.concatenate(list(data_size_train_indices), axis=0)
        data_size_test_indices = np.concatenate(list(data_size_test_indices), axis=0)

        # group the samples according to the nContainers column of dataset since this column is one the features that
        # tell with samples are in training set and which are in the test set
        core_num_indices = pd.DataFrame(
            [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
            columns=['col', 'indices'])

        # For interpolation and extrapolation, put all the cores to the test set.
        print('image_nums_train_data: ', image_nums_train_data)
        print('image_nums_test_data: ', image_nums_test_data)
        if set(image_nums_train_data) != set(image_nums_test_data):
            core_nums_test_data = core_nums_test_data + core_nums_train_data

        # save the samples that can be possible in the training(test) set because their core number column value
        # correspond to the predefined value from the input parameters
        core_num_train_indices = \
            core_num_indices.loc[(core_num_indices['col'].isin(core_nums_train_data))]['indices']
        core_num_test_indices = \
            core_num_indices.loc[(core_num_indices['col'].isin(core_nums_test_data))]['indices']

        # gather all the candidate samples from above into a list for training(test) set
        core_num_train_indices = np.concatenate(list(core_num_train_indices), axis=0)
        core_num_test_indices = np.concatenate(list(core_num_test_indices), axis=0)

        # data_conf["core_nums_train_data"] = core_nums_train_data
        # data_conf["core_nums_test_data"] = core_nums_test_data

        # Take the intersect of indices of datasize and core
        train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
        test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)

        return train_indices, test_indices


class FeatureSelection(DataPrepration):
    """This class performs feature selection as the last step of the data preparation"""
    def __init__(self):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, train_features, train_labels, test_features, test_labels, parameters, run_info):
        """calculate how many features are allowed to be selected, then using cross validation searches for the best
        parameters, then trains the model using the best parametrs"""

        # retrieve necessary information from parameters variable
        features_names = parameters['Features']['Extended_feature_names']
        min_features = parameters['FS']['min_features']
        max_features = parameters['FS']['max_features']
        k_features = self.calc_k_features(min_features, max_features, features_names)
        parameters['FS']['k_features'] = k_features


        # perform grid search
        if parameters['FS']['select_features_sfs']:
            self.logger.info("Grid Search: SFS and Ridge")
            y_pred_train, y_pred_test, run_info = self.Ridge_SFS_GridSearch(train_features, train_labels,
                                                                                test_features, test_labels,
                                                                                k_features, parameters, run_info)

        if parameters['FS']['XGBoost']:
            self.logger.info("Grid Search: XGBoost")
            y_pred_train, y_pred_test, run_info = self.XGBoost_Gridsearch(train_features, train_labels,
                                                                          test_features, test_labels,
                                                                          parameters, run_info)

        return y_pred_train, y_pred_test



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

        param_grid = pd.DataFrame(0, index=range(len(learning_rate_v) * len(reg_lambda_v) * len(min_child_weight_v) *len(max_depth_v)),

                                         columns=grid_elements)
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

                        cv_results = xgb.cv(params = xgboost_params, dtrain=train_data_dmatrix, nfold=fold_num,
                                            num_boost_round=100, early_stopping_rounds=10, metrics="rmse",
                                            verbose_eval=None, as_pandas=True, seed=123)

                        param_overal_MSE.append(cv_results["test-rmse-mean"].iloc[-1])
                        cv_info[str(row)]['MSE'] = cv_results["test-rmse-mean"].iloc[-1]
                        row += 1

        MSE_best = param_overal_MSE.index(min(param_overal_MSE))

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


class DataAnalysis(Task):
    """This is class defining machine learning task"""

    def __init__(self):
        Task.__init__(self)
        self.parameters = {}


class Regression(DataAnalysis):

    def __init__(self):
        DataAnalysis.__init__(self)
        self.conf = cp.ConfigParser()
        self.logger = logging.getLogger(__name__)

    def process(self, y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, parameters, run_info):
        """computes the MAPE error of the real predication by first scaling the predicted values"""

        # retrieve the same scaler used for normalization to perform inverse transform
        scaler = run_info[-1]['scaler']

        # retrieve all the feature names
        features_names = parameters['Features']['Extended_feature_names']

        self.logger.info("Computing MAPE: ")
        # compute MAPE error for not scaled data
        err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores = \
            self.mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features,
                                                train_labels, scaler, features_names)

        # populate run_info with results
        run_info[-1]['MAPE_test'] = err_test
        run_info[-1]['MAPE_train'] = err_train
        run_info[-1]['y_true_train'] = y_true_train
        run_info[-1]['y_pred_train'] = y_pred_train
        run_info[-1]['y_true_test'] = y_true_test
        run_info[-1]['y_pred_test'] = y_pred_test
        run_info[-1]['y_true_train_cores'] = y_true_train_cores
        run_info[-1]['y_pred_train_cores'] = y_pred_train_cores
        run_info[-1]['y_true_test_cores'] = y_true_test_cores
        run_info[-1]['y_pred_test_cores'] = y_pred_test_cores

        return err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test

    def mean_absolute_percentage_error(self, y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler, features_names):
        """computess MAPE error in real data by first scaling back the data (denormalize)"""

        # create the Data the same size and concatenate once real scaled output and once predicted output to make
        # data ready for inverse transform
        train_features_org = train_features
        if "y_true_train" in train_features_org.columns.values:
            train_features_org.drop("y_true_train", axis=1, inplace=True)
        if "y_pred_train" in train_features_org.columns.values:
            train_features_org.drop("y_pred_train", axis=1, inplace=True)

        test_features_org = test_features
        if 'y_true_test' in test_features_org.columns.values:
            test_features_org.drop('y_true_test', axis=1, inplace=True)
        if 'y_pred_test' in test_features_org.columns.values:
            test_features_org.drop('y_pred_test', axis=1, inplace=True)


        # add the scaled prediction data and real value to 2 different dataframe
        if y_pred_test != []:

            # Test error
            y_true_test = test_labels
            test_features_with_true = pd.DataFrame(test_features_org)
            test_features_with_pred = pd.DataFrame(test_features_org)

            # make the DataFrame out of array
            y_true_test = pd.DataFrame(y_true_test)
            y_pred_test = pd.DataFrame(y_pred_test)

            # make the index of test data alligned with training set
            y_pred_test.index = y_true_test.index

            # concatenate output columns
            test_features_with_true.insert(0, "y_true_test", y_true_test)
            test_features_with_pred.insert(0, "y_pred_test", y_pred_test)

            # make sure that each data has only predicted OR true value
            test_features_with_true.drop('y_pred_test', axis = 1, inplace=True)
            test_features_with_pred.drop('y_true_test', axis = 1, inplace=True)

            # do the inverse transform
            test_data_with_true = pd.DataFrame(scaler.inverse_transform(test_features_with_true.values))
            test_data_with_pred = pd.DataFrame(scaler.inverse_transform(test_features_with_pred.values))

            # label the columns names as it was before and add true and predicted value as column name
            test_data_with_true_cols = ['y_true_test']
            test_data_with_pred_cols = ['y_pred_test']
            for elem in features_names:
                test_data_with_true_cols.append(elem)
                test_data_with_pred_cols.append(elem)

            # make the indexing correct as it was before
            test_data_with_true.columns = test_data_with_true_cols
            test_data_with_pred.columns = test_data_with_pred_cols
            test_data_with_true.index = y_true_test.index
            test_data_with_pred.index = y_true_test.index

            cores = test_data_with_true['nContainers'].unique().tolist()
            cores = list(map(lambda x: int(x), cores))
            # if set(cores) == set(self.data_conf["core_nums_test_data"]):
            y_true_test_cores = cores

            cores = test_data_with_pred['nContainers'].unique().tolist()
            cores = list(map(lambda x: int(x), cores))
            # if set(cores) == set(self.data_conf["core_nums_test_data"]):
            y_pred_test_cores = cores

            #for col in test_data_with_true:
                #cores = test_data_with_true[col].unique().tolist()
                #cores = list(map(lambda x: int(x), cores))
                #if set(cores) == set(data_conf["core_nums_test_data"]):
                    #y_true_test_cores = test_data_with_true[col].tolist()
            # for col in test_data_with_pred:
                # cores = test_data_with_pred[col].unique().tolist()
                # cores = list(map(lambda x: int(x), cores))
                # if set(cores) == set(data_conf["core_nums_test_data"]):
                    # y_pred_test_cores = test_data_with_pred[col].tolist()

            # select just the output to compute MAPE
            y_true_test = test_data_with_true.iloc[:, 0]
            y_pred_test = test_data_with_pred.iloc[:, 0]

            err_test = np.mean(np.abs((y_true_test - y_pred_test) / y_true_test)) * 100
        #if y_pred_test == []:
        #    err_test = -1

        # Train error
        y_true_train = train_labels

        # make the DataFrame out of matrix of training input
        train_features_with_true = pd.DataFrame(train_features_org)
        train_features_with_pred = pd.DataFrame(train_features_org)

        # make the DataFrame out of matrix of training output
        y_true_train = pd.DataFrame(y_true_train)
        y_pred_train = pd.DataFrame(y_pred_train)
        y_pred_train.index = y_true_train.index

        # concatenate output columns
        train_features_with_true.insert(0, "y_true_train", y_true_train)
        train_features_with_pred.insert(0, "y_pred_train", y_pred_train)

        # make sure DataFrame has only true OR predicted value
        train_features_with_true.drop('y_pred_train', axis=1, inplace=True)
        train_features_with_pred.drop('y_true_train', axis=1, inplace=True)

        # do the inverse transform
        train_data_with_true = pd.DataFrame(scaler.inverse_transform(train_features_with_true.values))
        train_data_with_pred = pd.DataFrame(scaler.inverse_transform(train_features_with_pred.values))

        # label the columns names as it was before and add true and predicted value as column name
        train_data_with_true_cols = ['y_true_train']
        train_data_with_pred_cols = ['y_pred_train']
        for elem in features_names:
            train_data_with_true_cols.append(elem)
            train_data_with_pred_cols.append(elem)

        # adjust the columns and indices names to be aligned
        train_data_with_true.columns = train_data_with_true_cols
        train_data_with_pred.columns = train_data_with_pred_cols
        train_data_with_true.index = y_true_train.index
        train_data_with_pred.index = y_true_train.index

        cores = train_data_with_true['nContainers'].unique().tolist()
        cores = list(map(lambda x: int(x), cores))
        # if set(cores) == set(self.data_conf["core_nums_train_data"]):
        y_true_train_cores = cores

        cores = train_data_with_pred['nContainers'].unique().tolist()
        cores = list(map(lambda x: int(x), cores))
        # if set(cores) == set(self.data_conf["core_nums_train_data"]):
        y_pred_train_cores = cores


        # for col in train_data_with_true:
            # cores = train_data_with_true[col].unique().tolist()
            # cores = list(map(lambda x: int(x), cores))
            # if set(cores) == set(data_conf["core_nums_train_data"]):
                # y_true_train_cores = train_data_with_true[col].tolist()
        # for col in train_data_with_pred:
            # cores = train_data_with_pred[col].unique().tolist()
            # cores = list(map(lambda x: int(x), cores))
            # if set(cores) == set(data_conf["core_nums_train_data"]):
                #y_pred_train_cores = train_data_with_pred[col].tolist()

        # select just the output to compute MAPE
        y_true_train = train_data_with_true.iloc[:, 0]
        y_pred_train = train_data_with_pred.iloc[:, 0]

        err_train = np.mean(np.abs((y_true_train - y_pred_train) / y_true_train)) * 100

        return err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores


class Results(Task):
    """This is class defining machine learning task"""

    def __init__(self):
        Task.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, ext_df, run_info, parameters):

        self.logger.info("Preparing results and plots: ")
        # retrieve needed variables
        k_features = parameters['FS']['k_features']

        # find the best run among elements of run_info according to the minimum MAPE error
        best_run = self.select_best_run(run_info)

        # create the result name based on desired input parameters
        result_name = self.get_result_name(parameters, k_features)

        # create the path for saving the results
        result_path = self.save_results(parameters, best_run, result_name)

        # save the necessary plots in the made result folder
        self.plot_predicted_true(result_path, best_run, parameters)
        self.plot_cores_runtime(result_path, best_run, parameters, ext_df)
        self.plot_MSE_Errors(result_path, run_info)
        self.plot_MAPE_Errors(result_path, run_info)

        if parameters['FS']['select_features_sfs']:
            self.plot_histogram(result_path, run_info, parameters)
            self.plot_Model_Size(result_path, run_info)

        if parameters['FS']['XGBoost']:
            self.plot_xgboost_features_importance(result_path, best_run)




        # save the best_run variable as string
        target = open(os.path.join(result_path, "best_run"), 'a')
        target.write(str(best_run))
        target.close()

    def select_best_run(self, run_info):
        """selects the best run among independent runs of algorithm based on the minimum obtained MAPE error"""
        Mape_list = []
        for i in range(len(run_info)):
            Mape_list.append(run_info[i]['MAPE_test'])

        best_run_idx = Mape_list.index(min(Mape_list))

        # transfer just the best run element of run_info list to a dictionary variable: best_run
        best_run = run_info[best_run_idx]
        return best_run

    def get_result_name(self, parameters, k_features):
        """makes a name for saving the results and plots based on the current input parameters"""

        # retrieve necessary information to built the name of result folder
        degree = parameters['FS']['degree']
        select_features_sfs = parameters['FS']['select_features_sfs']
        select_features_vif = parameters['FS']['select_features_vif']
        is_floating = parameters['FS']['is_floating']
        run_num = parameters['General']['run_num']
        image_nums_train_data = parameters['Splitting']['image_nums_train_data']
        image_nums_test_data = parameters['Splitting']['image_nums_test_data']
        degree = str(degree)

        # concatenate parameters in a meaningful way
        result_name = "d=" + degree + "_"

        if select_features_vif:
            result_name += "vif_"

        if select_features_sfs and not is_floating:
            result_name += "sfs_"
            result_name += str(k_features[0]) + '_'
            result_name += str(k_features[1])

        if select_features_sfs and is_floating:
            result_name += "sffs_"
            result_name += str(k_features[0]) + '_'
            result_name += str(k_features[1])

        result_name = result_name + '_' + str(run_num) + '_runs'

        # add dataSize in training and test samples in the name
        Tr_size = '_Tr'
        for sz in image_nums_train_data:
            Tr_size = Tr_size + '_' + str(sz)

        Te_size = '_Te'
        for sz in image_nums_test_data:
            Te_size = Te_size + '_' + str(sz)

        result_name = result_name + Tr_size + Te_size
        return result_name

    def save_temporary_results(self, run_info):
        """save the temporary result at the end of each run"""

        # save the whole run_info so at each run the elements of dump variable changes
        target = open(os.path.join('./results/', "temp_run_info"), 'a')
        target.write(str(run_info))
        target.close()

    def save_results(self, parameters, best_run, result_name):
        """ save the extended results in the best_run dictionary and make the folder to save them and return the
        folder path"""

        # retrieve needed parameters
        degree = parameters['FS']['degree']
        ridge_params = parameters['Ridge']['ridge_params']
        result_path = parameters['DataPreparation']['result_path']

        # add excess information in the best run dictionary
        best_run["regressor_name"] = 'lr'
        best_run['parameters'] = parameters
        best_run["n_terms"] = degree
        best_run["param_grid"] = ridge_params
        best_run["best_estimator"] = best_run['best_model']._estimator_type

        # make the directory to save the data and plots
        result_path = os.path.join(result_path, result_name)

        # create the folder
        if os.path.exists(result_path) == False:
            os.mkdir(result_path)

        return result_path

    def plot_predicted_true(self, result_path, best_run, parameters):
        """plot the true values of application completion time versus the predicted ones"""

        # retrieve necessary information from the best_run variable
        y_true_train = best_run["y_true_train"]
        y_pred_train = best_run["y_pred_train"]
        y_true_test = best_run["y_true_test"]
        y_pred_test = best_run["y_pred_test"]

        # adjust parameters of the plot
        params_txt = 'best alpha: ' + str(best_run['best_param'])
        font = {'family': 'normal', 'size': 15}
        matplotlib.rc('font', **font)
        plot_path = os.path.join(result_path, "True_Pred_Plot")
        colors = cm.rainbow(np.linspace(0, 0.5, 3))
        fig = plt.figure(figsize=(9, 6))

        # scatter values
        plt.scatter(y_pred_train, y_true_train, marker='o', s=300, facecolors='none', label="Train Set",
                    color=colors[0])
        plt.scatter(y_pred_test, y_true_test, marker='^', s=300, facecolors='none', label="Test Set",
                    color=colors[1])
        # if y_pred_test != []:
        min_val = min(min(y_pred_train), min(y_true_train), min(y_pred_test), min(y_true_test))
        max_val = max(max(y_pred_train), max(y_true_train), max(y_pred_test), max(y_true_test))
        # if y_pred_test == []:
        # min_val = min(min(y_pred_train), min(y_true_train))
        # max_val = max(max(y_pred_train), max(y_true_train))
        lines = plt.plot([min_val, max_val], [min_val, max_val], '-')
        plt.setp(lines, linewidth=0.9, color=colors[2])

        # title and labels
        plt.title("Predicted vs True Values for " + 'lr' + "\n" + \
                  parameters['DataPreparation']['input_name'] + " " + str(parameters['DataPreparation']['case']) + " " + \
                  str(parameters['Splitting']["image_nums_train_data"]) + \
                  str(parameters['Splitting']["image_nums_test_data"]))
        plt.xlabel("Predicted values of applicationCompletionTime (ms)")
        plt.ylabel("True values of " + "\n" + "applicationCompletionTime (ms)")
        fig.text(.5, .01, params_txt, ha='center')
        plt.grid(True)
        plt.tight_layout()
        plt.legend(prop={'size': 20})
        plt.savefig(plot_path + ".pdf")

    def plot_cores_runtime(self, result_path, best_run, parameters, df):
        """plots the true and predicted values of training and test set according to their core numbers feature"""

        # retrieve the necessary information
        core_nums_train_data = parameters['Splitting']['core_nums_train_data']
        core_nums_test_data = parameters['Splitting']['core_nums_test_data']

        y_true_train = best_run['y_true_train']
        y_pred_train = best_run['y_pred_train']
        y_true_test = best_run['y_true_test']
        y_pred_test = best_run['y_pred_test']

        # make the plot ready
        font = {'family': 'normal', 'size': 15}
        matplotlib.rc('font', **font)
        plot_path = os.path.join(result_path, "cores_runtime_plot")
        colors = cm.rainbow(np.linspace(0, 0.5, 3))
        fig = plt.figure(figsize=(9, 6))

        params_txt = 'best alpha: ' + str(best_run['best_param'])
        regressor_name = parameters['Regression']['regressor_name']

        # group the indices corresponding to number of cores
        core_num_indices = pd.DataFrame(
            [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
            columns=['col', 'indices'])

        # scatter each point using its core number
        # Training
        legcount1 = 0
        for Trcore in core_nums_train_data:

            # plot_dict[str(Trcore)] ={}
            # DF of samples having the core number equal to Trcore
            y_idx = core_num_indices.loc[core_num_indices['col'] == Trcore]['indices']

            # convert them to list
            y_idx_list = y_idx.iloc[0].tolist() # no need to iterate

            for yi in y_idx_list:
                if yi in y_true_train.index:
                    legcount1 += 1
                    if legcount1 <= 1:
                        plt.scatter(Trcore, y_pred_train.loc[yi], marker='o', s=300, facecolors='none',
                            label="Train Predicted Values", color=colors[1])
                        plt.scatter(Trcore, y_true_train.loc[yi], marker='o', s=300, facecolors='none',
                            label="Train True Values", color=colors[2])

                    if legcount1 > 1:

                        plt.scatter(Trcore, y_pred_train.loc[yi], marker='o', s=300, facecolors='none', color=colors[1])
                        plt.scatter(Trcore, y_true_train.loc[yi], marker='o', s=300, facecolors='none', color=colors[2])

        legcount2 = 0
        for Tecore in core_nums_test_data:

            # DF of samples having the core number equal to Tecore
            y_idx_te = core_num_indices.loc[core_num_indices['col'] == Tecore]['indices']

            # convert them to list
            y_idx_te_list = y_idx_te.iloc[0].tolist() # no need to iterate

            for yie in y_idx_te_list:
                if yie in y_true_test.index:
                    legcount2 += 1
                    if legcount2 <= 1:
                        plt.scatter(Tecore, y_pred_test.loc[yie], marker='^', s=300, facecolors='none',
                            label="Test Predicted Values", color='C1')
                        plt.scatter(Tecore, y_true_test.loc[yie], marker='^', s=300, facecolors='none',
                            label="Test True Values", color='C3')

                    if legcount2 > 1:
                        plt.scatter(Tecore, y_pred_test.loc[yie], marker='^', s=300, facecolors='none', color='C1')
                        plt.scatter(Tecore, y_true_test.loc[yie], marker='^', s=300, facecolors='none', color='C3')

        plt.title("Predicted and True Values for " + regressor_name + "\n" + \
                  parameters['DataPreparation']['input_name'] + " " + str(parameters['DataPreparation']['case']) + " "
                  + str(parameters['Splitting']["image_nums_train_data"])
                  + str(parameters['Splitting']["image_nums_test_data"]))
        plt.xlabel("Number of cores")
        plt.ylabel("applicationCompletionTime (ms)")
        fig.text(.5, .01, params_txt, ha='center')
        plt.grid(True)
        plt.tight_layout()
        plt.legend(prop={'size': 20})
        plt.savefig(plot_path + ".pdf")


    def plot_histogram(self, result_path, run_info, parameters):
        """plots the histogram of frequency of selected features"""

        # retrieve necessary information
        degree = parameters['FS']['degree']
        plot_path = os.path.join(result_path, "Features_Ferquency_Histogram_plot")
        names_list = parameters['Features']['Extended_feature_names']

        # count the selected features in all runs
        name_count = []
        for i in range(len(names_list)):
            name_count.append(0)
        iternum = len(run_info)

        for i in range(iternum):
            for j in run_info[i]['Sel_features']:
                name_count[j] += 1

        # for degrees more than just add the selected features in the target plot to increase readability
        if degree > 1:
            zero_f_idx = []
            for i in range(len(name_count)):
                if name_count[i] == 0:
                    zero_f_idx.append(i)

            newname_list = []
            newname_count = []
            for i in range(len(name_count)):
                if name_count[i] != 0:
                    newname_list.append(names_list[i])
                    newname_count.append(name_count[i])

            names_list = newname_list
            name_count = newname_count

        # make the plot parameters ready
        font = {'family':'normal','size': 10}
        matplotlib.rc('font', **font)
        colors = cm.rainbow(np.linspace(0, 0.5, 3))
        fig = plt.figure(figsize=(9,6))
        plt.bar(range(len(names_list)), name_count)
        plt.xticks(range(len(names_list)), names_list)
        plt.xticks(rotation = 90)
        plt.title('Histogram of features selection frequency in '+str(len(run_info))+' runs')
        plt.tight_layout()
        fig.savefig(plot_path + ".pdf")

    def plot_MSE_Errors(self, result_path, run_info):
        """plots MSE error for unscaled values of prediction and true in different runs of the algorithm"""

        plot_path = os.path.join(result_path, "MSE_Error_plot")

        # make an object from FeatureSelection class to use functions
        fs = FeatureSelection()

        # gather all the MSE errors in training and test set in 2 lists
        MSE_list_TR = []
        MSE_list_TE = []

        for i in range(len(run_info)):
            y_true_train_val = run_info[i]['y_true_train']
            y_pred_train_val = run_info[i]['y_pred_train']
            msetr = fs.calcMSE(y_pred_train_val, y_true_train_val)

            y_true_test_val = run_info[i]['y_true_test']
            y_pred_test_val = run_info[i]['y_pred_test']
            msete = fs.calcMSE(y_pred_test_val, y_true_test_val)
            MSE_list_TR.append(msetr)
            MSE_list_TE.append(msete)

        # make the plot parameters
        font = {'family': 'normal', 'size': 15}
        matplotlib.rc('font', **font)
        colors = cm.rainbow(np.linspace(0, 0.5, 3))
        fig1 = plt.figure(figsize=(9, 6))
        plt.plot(range(1, len(run_info) + 1), MSE_list_TR, 'bs', range(1, len(run_info) + 1), MSE_list_TE, 'r^')
        plt.xlabel('runs')
        plt.ylabel('MSE Error')
        plt.title('MSE Error in Training and Test Sets in ' + str(len(run_info)) + ' runs')
        plt.xlim(1, len(MSE_list_TE))
        fig1.savefig(plot_path + ".pdf")

    def plot_MAPE_Errors(self, result_path, run_info):
        """plots MAPE error for unscaled values of prediction and true in different runs of the algorithm"""

        plot_path = os.path.join(result_path, "MAPE_Error_plot")

        # make an object from FeatureSelection class to use functions
        fs = FeatureSelection()

        # gather all the MAPE errors in training and test set in 2 lists
        MAPE_list_TR = []
        MAPE_list_TE = []
        for i in range(len(run_info)):
            y_true_train_val = run_info[i]['y_true_train']
            y_pred_train_val = run_info[i]['y_pred_train']
            mapetr = fs.calcMAPE(y_pred_train_val, y_true_train_val)

            y_true_test_val = run_info[i]['y_true_test']
            y_pred_test_val = run_info[i]['y_pred_test']
            mapete = fs.calcMAPE(y_pred_test_val, y_true_test_val)

            MAPE_list_TR.append(mapetr)
            MAPE_list_TE.append(mapete)

        # make the plot parameters
        font = {'family':'normal','size': 15}
        matplotlib.rc('font', **font)
        colors = cm.rainbow(np.linspace(0, 0.5, 3))
        fig2 = plt.figure(figsize=(9,6))
        plt.plot(range(1, len(run_info) + 1), MAPE_list_TR, 'bs', range(1, len(run_info) + 1), MAPE_list_TE, 'r^')
        plt.xlabel('runs')
        plt.ylabel('MAPE Error')
        plt.title('MAPE Error in Training and Test Sets in '+str(len(run_info))+' runs')
        plt.xlim(1, len(MAPE_list_TE))
        fig2.savefig(plot_path + ".pdf")

    def plot_Model_Size(self, result_path, run_info):
        """plots selected model size in different runs of the algorithm"""

        plot_path = os.path.join(result_path, "Model_Size_Plot")

        # list containing model size (number of selected features) in different runs
        model_size_list = []
        for i in range(len(run_info)):
            len(run_info[i]['Sel_features'])
            model_size_list.append(len(run_info[i]['Sel_features']))

        # make the plot parameters
        font = {'family':'normal','size': 15}
        matplotlib.rc('font', **font)
        colors = cm.rainbow(np.linspace(0, 0.5, 3))
        fig3 = plt.figure(figsize=(9,6))
        plt.bar(range(1, len(run_info) + 1), model_size_list)
        plt.xlabel('runs')
        plt.ylabel('Model Size')
        plt.title('Number of Selected Features in '+str(len(run_info))+' runs')
        plt.xlim(1, len(model_size_list))
        plt.ylim(1, max(model_size_list))
        fig3.savefig(plot_path + ".pdf")

    def plot_xgboost_features_importance(self, result_path, best_run):

        plot_path = os.path.join(result_path, "XGBoost_Feature_importance ")
        font = {'family':'normal', 'size': 15}
        matplotlib.rc('font', **font)
        colors = cm.rainbow(np.linspace(0, 0.5, 3))

        names_list = best_run['names_list']
        fscore = best_run['fscore']
        fig = plt.figure(figsize=(10, 14))
        plt.bar(range(len(names_list)), fscore)
        plt.xticks(range(len(names_list)), names_list)
        plt.xticks(rotation=90)
        plt.title('XGBoost Feature Importance')
        plt.savefig(plot_path + ".pdf")

