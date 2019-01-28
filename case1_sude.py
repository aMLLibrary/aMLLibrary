import sys
sys.modules[__name__].__dict__.clear()
import pickle
import os
import pandas as pd
import numpy as np
import math
import itertools
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import json
from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')

# different input options:
#[DataPreparation]
target_column = 1
input_name = 'kmeans'
split = 'split2'
case = 'case1'
use_spark_info = False
input_path = 'P8_kmeans.csv'
result_path = "./results/"

#[DebugLevel]
debug = True # This is for printing the logs. If debug is true, we also print the logs in the DEBUG level. Otherwise, only the logs in INFO level is printed.

#[FeatureExtender]
n_terms = [2]
degree = 1



# max_features_dt = [auto,sqrt]
# max_depth_dt = [10, 20, 30]
# min_samples_leaf_dt = [1, 2, 4]
# min_samples_split_dt = [2, 5, 10]

n_estimators = '[20, 40, 60, 80]'
max_features_rf = '[auto, sqrt]'
max_depth_rf = '[10, 20, 30]'
min_samples_leaf_rf = '[5, 6, 7]'
min_samples_split_rf = '[2, 5, 10]'
bootstrap = '[True, False]'
kernel = '[rbf]'
c = '[10]'
epsilon = '[0.1]'
gamma = '[0.01]'
n_neighbors = '[5,10]'




run_num = 10


# variables:
# TR and TE variables based on which the samples should be splitted
image_nums_train_data = [5]
image_nums_test_data = [5]
core_nums_train_data = [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
core_nums_test_data = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]

# Feature selection related variables
select_features_vif = False
select_features_sfs = True
min_features = 1
max_features = -1
is_floating = False
fold_num = 5
regressor_name = "lr"

# Classifier related variables (Ridge)
ridge_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lasso = '[0.1,0.5]'

# Classifier related variables (DT)
max_features_dt = '[sqrt]'
max_depth_dt = '[7]'
min_samples_leaf_dt = '[4]'
min_samples_split_dt = '[5]'
to_be_inv_List = ['nContainers']

def read_inputs():
    """reads input, adds inverse of nContainers and drop the run col"""
    df = pd.read_csv(input_path)

    if target_column != 1:
        column_names = list(df)
        column_names[1], column_names[target_column] = column_names[target_column], column_names[1]
        df = df.reindex(columns=column_names)

    if input_name == 'kmeans':
        # df['inverse_nContainers'] = 1 / df['nContainers']
        df = df.drop(['run'], axis=1)

    return df

def scale_data(df):

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)
    return scaled_df, scaler



def split_data(seed, df, image_nums_train_data, image_nums_test_data, core_nums_train_data, core_nums_test_data):

    data_conf = {}
    data_conf["case"] = case
    data_conf["split"] = split
    data_conf["input_name"] = input_name
    #data_conf["sparkdl_run"] = sparkdl_run

    if input_name != "classifierselection":
        # Drop the constant columns
        df = df.loc[:, (df != df.iloc[0]).any()]
        df = shuffle(df, random_state=seed)

        # If dataSize column has different values
        if "dataSize" in df.columns:
            data_size_indices = pd.DataFrame(
                [[k, v.values] for k, v in df.groupby('dataSize').groups.items()], columns=['col', 'indices'])

            data_size_train_indices = \
            data_size_indices.loc[(data_size_indices['col'].isin(image_nums_train_data))]['indices']
            data_size_test_indices = \
            data_size_indices.loc[(data_size_indices['col'].isin(image_nums_test_data))]['indices']

            data_size_train_indices = np.concatenate(list(data_size_train_indices), axis=0)
            data_size_test_indices = np.concatenate(list(data_size_test_indices), axis=0)

        else:

            data_size_train_indices = range(0, df.shape[0])
            data_size_test_indices = range(0, df.shape[0])

        data_conf["image_nums_train_data"] = image_nums_train_data
        data_conf["image_nums_test_data"] = image_nums_test_data

        #if input_name in sparkdl_inputs:
            #core_num_indices = pd.DataFrame(
                #[[k, v.values] for k, v in df.groupby('nCores').groups.items()], columns=['col', 'indices'])

        core_num_indices = pd.DataFrame(
            [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
            columns=['col', 'indices'])


        # For interpolation and extrapolation, put all the cores to the test set.
        print(set(image_nums_train_data))
        print(set(image_nums_test_data))
        if set(image_nums_train_data) != set(image_nums_test_data):
            core_nums_test_data = core_nums_test_data + core_nums_train_data

        core_num_train_indices = \
        core_num_indices.loc[(core_num_indices['col'].isin(core_nums_train_data))]['indices']
        core_num_test_indices = \
        core_num_indices.loc[(core_num_indices['col'].isin(core_nums_test_data))]['indices']

        core_num_train_indices = np.concatenate(list(core_num_train_indices), axis=0)
        core_num_test_indices = np.concatenate(list(core_num_test_indices), axis=0)

        data_conf["core_nums_train_data"] = core_nums_train_data
        data_conf["core_nums_test_data"] = core_nums_test_data

        # Take the intersect of indices of datasize and core
        train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
        test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)

        # Before Scaling (This is for testing whether split based on datasize is OK.)
        # train_df = self.df.ix[self.data_size_train_indices]
        # test_df  = self.df.ix[self.data_size_test_indices]
        # train_df = self.df.ix[self.core_num_train_indices]
        # test_df  = self.df.ix[self.core_num_test_indices]
        # print(train_indices)
        # print(test_indices)

        # cores = self.df[["nCores"]]
        # self.df = self.df[["applicationCompletionTime","maxTask_S8","gap_value_3",\
        #                    "avgTask_S8","maxTask_S5","maxTask_S4","maxTask_S3",\
        #                    "gap_value_2","avgTask_S3","avgTask_S5","gap_value_1",\
        #                    "avgTask_S4","SHmax_S3","dataSize","inverse_nCoresTensorflow",\
        #                    "nTask_S8","maxTask_S2","maxTask_S0","avgTask_S2","SHmax_S4",\
        #                    "avgTask_S0","nCoresTensorflow","nCores","maxTask_S1",\
        #                    "inverse_nCores","avgTask_S1","SHavg_S4","Bmax_S4",\
        #                    "nTask_S2","SHavg_S3","Bavg_S4"]]

    # Classifier selection, the analysis are only on the train set
    if input_name == "classifierselection":
        # Drop the constant columns
        df = df.loc[:, (df != df.iloc[0]).any()]
        cores = df["nCores"]
        # Read

        data_conf["core_nums_train_data"] = core_nums_train_data
        data_conf["core_nums_test_data"] = []
        data_conf["image_nums_train_data"] = image_nums_train_data
        data_conf["image_nums_test_data"] = []

        train_indices = range(0, len(cores))
        test_indices = []

    # Scale the data.
    df, scaler = scale_data(df)
    train_df = df.ix[train_indices]
    test_df = df.ix[test_indices]
    train_labels = train_df.iloc[:, 0]
    train_features = train_df.iloc[:, 1:]

    test_labels = test_df.iloc[:, 0]
    test_features = test_df.iloc[:, 1:]

    # train_cores = cores.ix[train_indices]
    # test_cores = cores.ix[test_indices]
    # data_conf["train_cores"] = train_cores
    # data_conf["test_cores"] = test_cores
    # features_names[0] : applicationCompletionTime
    features_names = list(df.columns.values)[1:]
    data_conf["train_features_org"] = train_features.as_matrix()
    data_conf["test_features_org"] = test_features.as_matrix()
    # print(features_names)

    data_conf["test_without_apriori"] = False

    return train_features, train_labels, test_features, test_labels, features_names, scaler, data_conf


def add_all_comb(inv_df, inversed_cols_tr, output_column_idx, degree):
    """Given a dataframe, returns an extended df containing all combinations of columns except the ones that are
    inversed"""

    features_names = inv_df.columns.values
    df_dict = dict(inv_df)
    data_matrix = pd.DataFrame.as_matrix(inv_df)
    indices = list(range(data_matrix.shape[1]))

    # compute all possible combinations with replacement
    for j in range(2, degree + 1):
        combs = list(itertools.combinations_with_replacement(indices, j))
        # removes the combinations containing features and inversed of them
        remove_list_idx = []
        for ii in combs:
            for kk in inversed_cols_tr:
                if len(list(set.intersection(set(ii), set(kk)))) >= 2:
                    remove_list_idx.append(ii)
            if output_column_idx in ii:
                remove_list_idx.append(ii)
        for r in range(0,len(remove_list_idx)):
            combs.remove(remove_list_idx[r])
        # compute resulting column of the remaining combinations and add to the df
        for cc in combs:
            new_col = calculate_new_col(data_matrix, list(cc))
            new_feature_name = ''
            for i in range(len(cc)-1):
                new_feature_name = new_feature_name+features_names[cc[i]]+'_'
            new_feature_name = new_feature_name+features_names[cc[i+1]]
            df_dict[new_feature_name] = new_col

    ext_df = pd.DataFrame.from_dict(df_dict)
    return ext_df

def calculate_new_col(X, indices):
    """Given two indices and input matrix returns the multiplication of the columns corresponding columns"""
    index = 0
    new_col = X[:, indices[index]]
    for ii in list(range(1, len(indices))):
        new_col = np.multiply(new_col, X[:, indices[index + 1]])
        index += 1
    return new_col


def add_inverse_features(df, to_be_inv_List):
    """Given a dataframe and the name of columns that should be inversed, add the needed inversed columns and returns
    the resulting df and the indices of two reciprocals separately"""

    df_dict = dict(df)
    for c in to_be_inv_List:
        new_col = 1/np.array(df[c])
        new_feature_name = 'inverse_'+c
        df_dict[new_feature_name] = new_col

    inv_df = pd.DataFrame.from_dict(df_dict)

    # returns the indices of the columns that should be inversed and their inversed in one tuple
    inversing_cols = []
    for c in to_be_inv_List:
        cidx = inv_df.columns.get_loc(c)
        cinvidx = inv_df.columns.get_loc('inverse_'+c)
        inv_idxs = (cidx, cinvidx)
        inversing_cols.append(inv_idxs)
    return inv_df, inversing_cols

def calc_k_features(min_features, max_features, ext_feature_names):
    """calculate the range of number of features that sfs is allowed to select"""
    min_k_features = int(min_features)
    max_k_features = int(max_features)
    # Selecting from all features
    if max_k_features == -1:
        k_features = (min_k_features, len(ext_feature_names))
        # Selecting from the given range
    if max_k_features != -1:
        k_features = (min_k_features, max_k_features)
    return k_features

def read_inputs_sfs(ext_train_features, ext_test_features, train_labels, test_labels,
                    ext_feature_names, scaler, data_conf):

    ext_train_features = ext_train_features.as_matrix()
    ext_test_features = ext_test_features.as_matrix()
    train_labels = train_labels.as_matrix()
    test_labels = test_labels.as_matrix()
    feature_names = np.array(ext_feature_names)
    test_features_org = data_conf["test_features_org"]
    train_features_org = data_conf["train_features_org"]
    # del data_conf["test_features_org"]
    # del data_conf["train_features_org"]

    return train_features_org, test_features_org

def calcMSE(Y_hat, Y):
    MSE = np.mean((Y_hat - Y) ** 2)
    return MSE

def calcMAPE(Y_hat, Y):
    """given true and predicted values returns MAPE error"""
    Mapeerr = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    return Mapeerr


def Ridge_SFS_GridSearch(ridge_params, ext_train_features,train_labels,k_features,fold_num):

    X = pd.DataFrame.as_matrix(ext_train_features)
    Y = pd.DataFrame.as_matrix(train_labels)
    ext_feature_names = ext_train_features.columns.values

    alpha_v = ridge_params


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
        this_a = 'alpha = '+str(a)
        cv_info[this_a] = {}

        # building the sfs
        sfs = SFS(clone_estimator=True,
                  estimator = model,
                  k_features = k_features,
                  forward=True,
                  floating=False,
                  scoring='neg_mean_squared_error',
                  cv = fold_num,
                  n_jobs = -1)

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
        MSE = calcMSE(Y_hat, Y)
        param_overal_MSE.append(MSE)
        cv_info[this_a]['MSE_error'] = MSE


        # evaluate the MAPE error on the whole training data only using the selected features
        Mape_error = calcMAPE(Y_hat, Y)
        Mape_overal_error.append(Mape_error)
        cv_info[this_a]['MAPE_error'] = Mape_error
        print('alpha = ', a, '     MSE Error= ', MSE, '    MAPE Error = ', Mape_error, '    Ridge Coefs= ', ridge.coef_ ,'     Intercept = ', ridge.intercept_, '     SEL = ', ext_feature_names[list(sel_F_idx)]  )


    """################################################## Results #####################################################"""

    # select the best alpha based on obtained values
    MSE_index = param_overal_MSE.index(min(param_overal_MSE))

    # report the best alpha based on obtained values
    print('Least_MSE_Error_index = ', MSE_index, ' => Least_RSE_Error_alpha = ', alpha_v[MSE_index])
    Least_MSE_alpha = alpha_v[MSE_index]
    best_trained_model = Ridge(Least_MSE_alpha)
    best_trained_model.fit(X[:, sel_F[MSE_index]], Y)
    sel_idx = sel_F[MSE_index]

    return cv_info, Least_MSE_alpha, sel_idx, best_trained_model

def mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler):


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


    if y_pred_test != []:
        # Test error
        y_true_test = test_labels
        test_features_with_true = pd.DataFrame(test_features_org)
        test_features_with_pred = pd.DataFrame(test_features_org)

        y_true_test = pd.DataFrame(y_true_test)
        y_pred_test = pd.DataFrame(y_pred_test)

        y_pred_test.index = y_true_test.index


        test_features_with_true.insert(0, "y_true_test", y_true_test)
        test_features_with_pred.insert(0, "y_pred_test", y_pred_test)



        test_features_with_true.drop('y_pred_test', axis = 1, inplace=True)
        test_features_with_pred.drop('y_true_test', axis = 1, inplace=True)


        test_data_with_true = pd.DataFrame(scaler.inverse_transform(test_features_with_true.values))
        test_data_with_pred = pd.DataFrame(scaler.inverse_transform(test_features_with_pred.values))

        test_data_with_true_cols = ['y_true_test']
        test_data_with_pred_cols = ['y_pred_test']
        for elem in ext_feature_names:
            test_data_with_true_cols.append(elem)
            test_data_with_pred_cols.append(elem)


        test_data_with_true.columns = test_data_with_true_cols
        test_data_with_pred.columns = test_data_with_pred_cols
        test_data_with_true.index = y_true_test.index
        test_data_with_pred.index = y_true_test.index

        cores = test_data_with_true['nContainers'].unique().tolist()
        cores = list(map(lambda x: int(x), cores))
        if set(cores) == set(data_conf["core_nums_test_data"]):
            y_true_test_cores = cores

        cores = test_data_with_pred['nContainers'].unique().tolist()
        cores = list(map(lambda x: int(x), cores))
        if set(cores) == set(data_conf["core_nums_test_data"]):
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

        y_true_test = test_data_with_true.iloc[:, 0]
        y_pred_test = test_data_with_pred.iloc[:, 0]

        err_test = np.mean(np.abs((y_true_test - y_pred_test) / y_true_test)) * 100
    #if y_pred_test == []:
    #    err_test = -1

    # Train error
    y_true_train = train_labels
    train_features_with_true = pd.DataFrame(train_features_org)
    train_features_with_pred = pd.DataFrame(train_features_org)

    y_true_train = pd.DataFrame(y_true_train)
    y_pred_train = pd.DataFrame(y_pred_train)
    y_pred_train.index = y_true_train.index

    train_features_with_true.insert(0, "y_true_train", y_true_train)
    train_features_with_pred.insert(0, "y_pred_train", y_pred_train)

    train_features_with_true.drop('y_pred_train', axis=1, inplace=True)
    train_features_with_pred.drop('y_true_train', axis=1, inplace=True)

    train_data_with_true = pd.DataFrame(scaler.inverse_transform(train_features_with_true.values))
    train_data_with_pred = pd.DataFrame(scaler.inverse_transform(train_features_with_pred.values))



    train_data_with_true_cols = ['y_true_train']
    train_data_with_pred_cols = ['y_pred_train']
    for elem in ext_feature_names:
        train_data_with_true_cols.append(elem)
        train_data_with_pred_cols.append(elem)

    train_data_with_true.columns = train_data_with_true_cols
    train_data_with_pred.columns = train_data_with_pred_cols
    train_data_with_true.index = y_true_train.index
    train_data_with_pred.index = y_true_train.index

    cores = train_data_with_true['nContainers'].unique().tolist()
    cores = list(map(lambda x: int(x), cores))
    if set(cores) == set(data_conf["core_nums_train_data"]):
        y_true_train_cores = cores

    cores = train_data_with_pred['nContainers'].unique().tolist()
    cores = list(map(lambda x: int(x), cores))
    if set(cores) == set(data_conf["core_nums_train_data"]):
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



    y_true_train = train_data_with_true.iloc[:, 0]
    y_pred_train = train_data_with_pred.iloc[:, 0]

    err_train = np.mean(np.abs((y_true_train - y_pred_train) / y_true_train)) * 100

    return err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores

def sfs_and_grid_search(ridge_params, ext_train_features, train_labels, k_features, fold_num):
    if data_conf["test_without_apriori"] == False:

        cv_info, Least_MSE_alpha, sel_idx, best_trained_model = \
        Ridge_SFS_GridSearch(ridge_params, ext_train_features, train_labels, k_features, fold_num)

        # Since the data for classsifierselection is too small, we only calculate the train error

        X_train = pd.DataFrame.as_matrix(ext_train_features)
        Y_train = pd.DataFrame.as_matrix(train_labels)
        X_test = pd.DataFrame.as_matrix(ext_test_features)
        Y_test = pd.DataFrame.as_matrix(test_labels)

        if data_conf["input_name"] == "classifierselection":
            y_pred_test = []
        else:
            y_pred_test = best_trained_model.predict(X_test[:, sel_idx])

        y_pred_train = best_trained_model.predict(X_train[:, sel_idx])

        return cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test


def get_result_name(degree, select_features_sfs, k_features, is_floating):

    if degree == [] :
        result_name = "d=0_"
    if degree != [] :
        # n_terms = list(map(lambda x: str(x), n_terms))
        # n_terms = ','.join(n_terms)
        degree = str(degree)
        result_name = "d=" + degree + "_"
    # if select_features_vif == True :
    #    result_name += "vif_"
    # if select_features_vif == False :
    #    result_name += "no_vif_"
    if select_features_sfs == False :
        result_name += "baseline_results"
    if select_features_sfs == True and is_floating == False:
        result_name += "sfs_"
        result_name += str(k_features[0])+'_'
        result_name += str(k_features[1])
        # if max_k_features == -1:
        #    result_name += "all_features_results"
        #if max_k_features != -1 :
        #    result_name += str(self.max_k_features) + "_features_results"
    if select_features_sfs == True and is_floating == True :
        result_name += "sffs_"
        result_name += str(k_features[0]) + '_'
        result_name += str(k_features[1])
        # if self.max_k_features == -1 :
        #    self.result_name += "all_features_results"
        # if self.max_k_features != -1 :
        #   self.result_name += str(self.max_k_features) + "_features_results"
    #if self.data_conf["test_without_apriori"] == True:
    #    self.result_name += "_test_without_apriori"
    #if self.data_conf["fixed_features"] == True:
    #    self.result_name = "fixed_features"


    return result_name


def save_results(err_train, err_test, result_name, result_path, data_conf, cv_info, ridge_params, best_trained_model,
                               degree, Least_MSE_alpha):

    selected_feature_indices = list(sel_idx)
    best_params = Least_MSE_alpha

    results = data_conf
    results["regressor_name"] = 'lr'
    results["n_terms"] = degree
    results["selected_feature_names"] = cv_info['alpha = '+str(best_params)]['Selected_Features_Names']
    results["err_train"] = err_train
    results["err_test"] = err_test
    results["param_grid"] = ridge_params
    # results["best_estimator"] = gs.best_estimator_.steps
    results["best_estimator"] = best_trained_model._estimator_type
    # results["sfs_subsets"] = sfs.subsets_

    # split_no = data_conf["split"]
    case_no = data_conf["case"]

    result_path = os.path.join(result_path, result_name)
    if os.path.exists(result_path) == False:
        os.mkdir(result_path)

    return result_path, results

def plot_predicted_true(result_path, y_true_train, y_pred_train, y_true_test, y_pred_test):
    params_txt = 'best alpha: ' + str(run_info[-1]['best_param'])
    font = {'family': 'normal', 'size': 15}
    matplotlib.rc('font', **font)
    plot_path = os.path.join(result_path, "true_pred_plot")
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(y_pred_train, y_true_train, marker='o', s=300, facecolors='none', label="Train Set",
                color=colors[0])
    plt.scatter(y_pred_test, y_true_test, marker='^', s=300, facecolors='none', label="Test Set",
                color=colors[1])
    #if y_pred_test != []:
    min_val = min(min(y_pred_train), min(y_true_train), min(y_pred_test), min(y_true_test))
    max_val = max(max(y_pred_train), max(y_true_train), max(y_pred_test), max(y_true_test))
    #if y_pred_test == []:
    # min_val = min(min(y_pred_train), min(y_true_train))
    #max_val = max(max(y_pred_train), max(y_true_train))
    lines = plt.plot([min_val, max_val], [min_val, max_val], '-')
    plt.setp(lines, linewidth=0.9, color=colors[2])
    plt.title("Predicted vs True Values for " + regressor_name + "\n" + \
              data_conf["input_name"] + " " + str(data_conf["case"]) + " " + \
              str(data_conf["image_nums_train_data"]) + \
              str(data_conf["image_nums_test_data"]))
    plt.xlabel("Predicted values of applicationCompletionTime (ms)")
    plt.ylabel("True values of " + "\n" + "applicationCompletionTime (ms)")
    fig.text(.5, .01, params_txt, ha='center')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(prop={'size': 20})
    plt.savefig(plot_path + ".pdf")

def plot_cores_runtime(run_info, result_path, core_nums_train_data, core_nums_test_data , y_true_train, y_true_test):

    font = {'family': 'normal', 'size': 15}
    matplotlib.rc('font', **font)
    plot_path = os.path.join(result_path, "cores_runtime_plot")
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig = plt.figure(figsize=(9, 6))
    #if self.data_conf["fixed_features"] == False:


    params_txt = 'best alpha: ' + str(run_info[-1]['best_param'])
    regressor_name = 'Logistic Regression'

    core_num_indices = pd.DataFrame(
        [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
        columns=['col', 'indices'])
    plot_dict = {}
    # Training
    legcount1 = 0
    for Trcore in core_nums_train_data:

        #plot_dict[str(Trcore)] ={}
        # DF of samples having the core number equal to Trcore
        y_idx = core_num_indices.loc[core_num_indices['col'] == Trcore]['indices']

        # convert them to list
        y_idx_list = y_idx.iloc[0].tolist() # no need to iterate
        #y_tr_true = []
        #y_tr_pred = []

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

    #if self.data_conf["fixed_features"] == True:
    #    plt.scatter(self.data_conf["train_cores"], self.y_pred_train, marker='o', s=300, facecolors='none',
    #                label="Train Predicted Values", color=colors[1])
    #    plt.scatter(self.data_conf["train_cores"], self.y_true_train, marker='o', s=300, facecolors='none',
    #                label="Train True Values", color=colors[2])
    #    if self.y_pred_test != []:
    #        plt.scatter(self.data_conf["test_cores"], self.y_pred_test, marker='^', s=300, facecolors='none',
    #                    label="Test Predicted Values", color='C1')
    #        plt.scatter(self.data_conf["test_cores"], self.y_true_test, marker='^', s=300, facecolors='none',
    #                    label="Test True Values", color='C3')

    plt.title("Predicted and True Values for " + regressor_name + "\n" + \
              data_conf["input_name"] + " " + str(data_conf["case"]) + " " + \
              str(data_conf["image_nums_train_data"]) + \
              str(data_conf["image_nums_test_data"]))
    plt.xlabel("Number of cores")
    plt.ylabel("applicationCompletionTime (ms)")
    fig.text(.5, .01, params_txt, ha='center')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(prop={'size': 20})
    plt.savefig(plot_path + ".pdf")



def calcMSE(Y_hat, Y):
    MSE = np.mean((Y_hat - Y) ** 2)
    return MSE

def calcMAPE(Y_hat, Y):
    """given true and predicted values returns MAPE error"""
    Mapeerr = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    return Mapeerr



def plot_histogram(run_info, result_path):

    plot_path = os.path.join(result_path, "Features_Ferquency_Histogram_plot")

    """Plot the histogram of features selection frequency"""
    names_list = run_info[0]['ext_feature_names']
    name_count = []
    for i in range(len(names_list)):
        name_count.append(0)
    iternum = len(run_info)

    for i in range(iternum):
        for j in run_info[i]['Sel_features']:
            name_count[j] += 1

    font = {'family':'normal','size': 10}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig = plt.figure(figsize=(9,6))
    plt.bar(range(len(names_list)), name_count)
    plt.xticks(range(len(names_list)), names_list)
    plt.xticks(rotation = 90)
    plt.title('Histogram of features selection frequency in '+str(len(run_info))+' runs')
    # plt.show()
    plt.tight_layout()
    fig.savefig(plot_path + ".pdf")



def plot_MSE_Errors(run_info, result_path):

    plot_path = os.path.join(result_path, "MSE_Error_plot")
    MSE_list_TR = []
    MSE_list_TE = []
    for i in range(len(run_info)):
        y_true_train_val = run_info[i]['y_true_train']
        y_pred_train_val = run_info[i]['y_pred_train']
        msetr = calcMSE(y_pred_train_val, y_true_train_val)

        y_true_test_val = run_info[i]['y_true_test']
        y_pred_test_val = run_info[i]['y_pred_test']
        msete = calcMSE(y_pred_test_val, y_true_test_val)

        MSE_list_TR.append(msetr)
        MSE_list_TE.append(msete)

    font = {'family':'normal','size': 15}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig1 = plt.figure(figsize=(9,6))
    plt.plot(range(1, len(run_info)+1), MSE_list_TR, 'bs', range(1, len(run_info)+1), MSE_list_TE, 'r^')
    plt.xlabel('runs')
    plt.ylabel('MSE Error')
    plt.title('MSE Error in Training and Test Sets in '+str(len(run_info))+' runs')
    plt.xlim(1, len(MSE_list_TE))
    # plt.show()
    fig1.savefig(plot_path + ".pdf")


def plot_MAPE_Errors(run_info, result_path):

    plot_path = os.path.join(result_path, "MAPE_Error_plot")
    MAPE_list_TR = []
    MAPE_list_TE = []
    for i in range(len(run_info)):
        y_true_train_val = run_info[i]['y_true_train']
        y_pred_train_val = run_info[i]['y_pred_train']
        mapetr = calcMAPE(y_pred_train_val, y_true_train_val)

        y_true_test_val = run_info[i]['y_true_test']
        y_pred_test_val = run_info[i]['y_pred_test']
        mapete = calcMAPE(y_pred_test_val, y_true_test_val)

        MAPE_list_TR.append(mapetr)
        MAPE_list_TE.append(mapete)


    font = {'family':'normal','size': 15}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig2 = plt.figure(figsize=(9,6))
    plt.plot(range(1, len(run_info) + 1), MAPE_list_TR, 'bs', range(1, len(run_info) + 1), MAPE_list_TE, 'r^')
    plt.xlabel('runs')
    plt.ylabel('MAPE Error')
    plt.title('MAPE Error in Training and Test Sets in '+str(len(run_info))+' runs')
    plt.xlim(1, len(MAPE_list_TE))
    # plt.legend()
    # plt.show()
    fig2.savefig(plot_path + ".pdf")


def plot_Model_Size(run_info, result_path):

    plot_path = os.path.join(result_path, "Model_Size_Plot")
    model_size_list = []
    for i in range(len(run_info)):
        len(run_info[i]['Sel_features'])
        model_size_list.append(len(run_info[i]['Sel_features']))

    font = {'family':'normal','size': 15}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig3 = plt.figure(figsize=(9,6))
    plt.bar(range(1, len(run_info) + 1), model_size_list)
    plt.xlabel('runs')
    plt.ylabel('Model Size')
    plt.title('Number of Selected Features in '+str(len(run_info))+' runs')
    plt.xlim(1, len(model_size_list))
    plt.ylim(1, len(run_info[0]['ext_feature_names']))
    # plt.show()
    fig3.savefig(plot_path + ".pdf")


seed_v = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 1023]
df = read_inputs()
df, inversing_cols = add_inverse_features(df, to_be_inv_List)

run_info = []
for iter in range(run_num):
    result_path = "./results/"

    this_run = 'run_'+str(iter)
    print(this_run)

    run_info.append({})

    train_features, train_labels, test_features, test_labels, features_names, scaler, data_conf = \
        split_data(seed_v[iter], df, image_nums_train_data, image_nums_test_data, core_nums_train_data, core_nums_test_data)

    ext_train_features = add_all_comb(train_features, inversing_cols, 0, degree)
    ext_test_features = add_all_comb(test_features, inversing_cols, 0, degree)
    ext_feature_names = ext_train_features.columns.values

    run_info[iter]['ext_feature_names'] = ext_feature_names

    k_features = calc_k_features(min_features, max_features, ext_feature_names)


    cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test = \
        sfs_and_grid_search(ridge_params, ext_train_features, train_labels, k_features, fold_num)

    run_info[iter]['Sel_features'] = sel_idx
    run_info[iter]['Sel_features_names'] = ext_feature_names[list(sel_idx)]
    run_info[iter]['best_param'] = Least_MSE_alpha
    run_info[iter]['best_model'] = best_trained_model

    err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores = \
        mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler)

    run_info[iter]['y_true_train'] = y_true_train
    run_info[iter]['y_pred_train'] = y_pred_train
    run_info[iter]['y_true_test'] = y_true_test
    run_info[iter]['y_pred_test'] = y_pred_test


    result_name = get_result_name(degree, select_features_sfs, k_features, is_floating)

    result_path, results = save_results(err_train, err_test, result_name, result_path,
                               data_conf, cv_info, ridge_params, best_trained_model, degree, Least_MSE_alpha)



plot_predicted_true(result_path, y_true_train, y_pred_train, y_true_test, y_pred_test)
plot_cores_runtime(run_info, result_path, core_nums_train_data, core_nums_test_data , y_true_train, y_true_test)
plot_histogram(run_info, result_path)
plot_MSE_Errors(run_info, result_path)
plot_MAPE_Errors(run_info, result_path)
plot_Model_Size(run_info, result_path)


target = open(os.path.join(result_path, "run_info"), 'a')
target.write(str(run_info))
target.close()




