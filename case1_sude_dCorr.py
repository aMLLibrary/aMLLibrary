import sys
sys.modules[__name__].__dict__.clear()
import configparser, time, os
import ast
import pandas as pd
import numpy as np
import itertools
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
import math


# keep the starting time to compute the execution time
start = time.time()

# making the config file
config = configparser.ConfigParser()
config.sections()
config.read('params.ini')
config.sections()




#[General]
# Number of independent runs of the whole code with different shuffling
# some diagrams are plotted using the best run among these, the others collect all runs information
run_num = int(config['General']['run_num'])



# different input options:
#[DataPreparation]

# index of output column
target_column = int(config['DataPreparation']['target_column'])
input_name = config['DataPreparation']['input_name']
split = config['DataPreparation']['split']
case = config['DataPreparation']['case']
use_spark_info = config['DataPreparation'].getboolean('use_spark_info')
input_path = config['DataPreparation']['input_path']

# main folder to store diagrams and variables
# the main result path should exist
result_path = config['DataPreparation']['result_path']



#[DebugLevel]
# This is for printing the logs. If debug is true, we also print the logs in the DEBUG level. Otherwise,
# only the logs in INFO level is printed.
debug = config['DebugLevel'].getboolean('debug')



#[Splitting]
# How data will be splitted: just the samples that comply with the following specifications are placed
# in training and test data
image_nums_train_data = config['Splitting']['image_nums_train_data']
image_nums_train_data = [int(i) for i in ast.literal_eval(image_nums_train_data)]

image_nums_test_data = config['Splitting']['image_nums_test_data']
image_nums_test_data = [int(i) for i in ast.literal_eval(image_nums_test_data)]


core_nums_train_data = config['Splitting']['core_nums_train_data']
core_nums_train_data = [int(i) for i in ast.literal_eval(core_nums_train_data)]

core_nums_test_data = config['Splitting']['core_nums_test_data']
core_nums_test_data = [int(i) for i in ast.literal_eval(core_nums_test_data)]

seed_v = config['Splitting']['seed_vector']
seed_v = [int(i) for i in ast.literal_eval(seed_v)]


#[FeatureExtender]
# the degree of combinatorial terms computing as the extended features
degree = int(config['FeatureExtender']['degree'])

n_terms = config['FeatureExtender']['n_terms']
n_terms = [int(i) for i in ast.literal_eval(n_terms)]





# [rf]

# parameters for random forest
n_estimators = config['rf']['n_estimators']
n_estimators = [int(i) for i in ast.literal_eval(n_estimators)]
max_features_rf = config['rf']['max_features_rf']


max_depth_rf = config['rf']['max_depth_rf']
max_depth_rf = [int(i) for i in ast.literal_eval(max_depth_rf)]

min_samples_leaf_rf = config['rf']['min_samples_leaf_rf']
min_samples_leaf_rf = [int(i) for i in ast.literal_eval(min_samples_leaf_rf)]

min_samples_split_rf = config['rf']['min_samples_split_rf']
min_samples_split_rf = [int(i) for i in ast.literal_eval(min_samples_split_rf)]

bootstrap = config['rf']['bootstrap']
bootstrap = [i for i in ast.literal_eval(bootstrap)]

kernel = config['rf']['kernel']

c = config['rf']['c']
c = [int(i) for i in ast.literal_eval(c)]

epsilon = config['rf']['epsilon']
epsilon = [i for i in ast.literal_eval(epsilon)]

gamma = config['rf']['gamma']
gamma = [i for i in ast.literal_eval(gamma)]

n_neighbors = config['rf']['n_neighbors']
n_neighbors = [int(i) for i in ast.literal_eval(n_neighbors)]


#[FS]
# information about the feature selection method
select_features_vif = config['FS'].getboolean('select_features_vif')
select_features_sfs = config['FS'].getboolean('select_features_sfs')
min_features = int(config['FS']['min_features'])
max_features = int(config['FS']['max_features'])
is_floating = config['FS'].getboolean('is_floating')
fold_num = int(config['FS']['fold_num'])
Confidence_level = config['FS']['Confidence_level']
Confidence_level = ast.literal_eval(Confidence_level)



# [Regression]
# which regressor is going to used for the prediction
regressor_name = config['Regression']['regressor_name']


# [Ridge]
ridge_params = config['Ridge']['ridge_params']
ridge_params = [i for i in ast.literal_eval(ridge_params)]


# [Lasso]
lasso = config['Lasso']['lasso']
lasso = [i for i in ast.literal_eval(lasso)]



# [dt]
max_features_dt = config['dt']['max_features_dt']

max_depth_dt = config['dt']['max_depth_dt']
max_depth_dt = [int(i) for i in ast.literal_eval(max_depth_dt)]

min_samples_leaf_dt = config['dt']['min_samples_leaf_dt']
min_samples_leaf_dt = [int(i) for i in ast.literal_eval(min_samples_leaf_dt)]

min_samples_split_dt = config['dt']['min_samples_split_dt']
min_samples_split_dt = [int(i) for i in ast.literal_eval(min_samples_split_dt)]




# [Inverse]
# the name of to be inverted column in the dataset
to_be_inv_List = []
to_be_inv_List.append(config['Inverse']['to_be_inv_List'])



def read_inputs():
    """reads inputs and drop the run col and columns with constant values"""
    df = pd.read_csv(input_path)

    if target_column != 1:
        column_names = list(df)
        column_names[1], column_names[target_column] = column_names[target_column], column_names[1]
        df = df.reindex(columns=column_names)

    if input_name == 'kmeans':
        # df['inverse_nContainers'] = 1 / df['nContainers']
        df = df.drop(['run'], axis=1)

    if input_name != "classifierselection":

        # Drop the constant columns
        df = df.loc[:, (df != df.iloc[0]).any()]
    return df

def scale_data(df):

    """scale the dataframe"""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)
    return scaled_df, scaler



def split_data(seed, df, image_nums_train_data, image_nums_test_data, core_nums_train_data, core_nums_test_data):

    """split the original dataframe into Training Input (train_features), Training Output(train_labels),
    Test Input(test_features) and Test Output(test_labels)"""
    data_conf = {}
    data_conf["case"] = case
    data_conf["split"] = split
    data_conf["input_name"] = input_name
    #data_conf["sparkdl_run"] = sparkdl_run

    if input_name != "classifierselection":

        df = shuffle(df, random_state=seed)


        # If dataSize column has different values
        # finds out which are the datasizes in dataset to select corresponding indices for the TR and TE part
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

        # finds out which are the core numbers in dataset to select corresponding indices for the TR and TE part
        core_num_indices = pd.DataFrame(
            [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
            columns=['col', 'indices'])


        # For interpolation and extrapolation, put all the cores to the test set.
        print('image_nums_train_data: ', image_nums_train_data)
        print('image_nums_test_data: ', image_nums_test_data)
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


    # Scale the dataset, all together.
    # IMPORTANT: Since this functions does the scaling and we need the scaler be consistent with the test data, this
    # function should be applied on the extended dataset
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



def Ridge_SFS_GridSearch(ridge_params, train_features,train_labels, test_features, test_labels, k_features, fold_num):
    """given training and test input and output, required parameters to be searched among and the fold number, it
    first searches for the best features using SFS with CV, and using ONLY training set, then computes the MSE using
    only the SFS selected features and training set. Then select the parameter according to the least MSE error. Then
    it trains the model using that parameter, selected features and training data, then predicts the out of both
    training and test set"""

    X = pd.DataFrame.as_matrix(train_features)
    Y = pd.DataFrame.as_matrix(train_labels)
    ext_feature_names = train_features.columns.values

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


    # get the results:
    # select the best alpha based on obtained values
    MSE_index = param_overal_MSE.index(min(param_overal_MSE))

    # report the best alpha based on obtained values
    print('Least_MSE_Error_index = ', MSE_index, ' => Least_RSE_Error_alpha = ', alpha_v[MSE_index])
    Least_MSE_alpha = alpha_v[MSE_index]
    best_trained_model = Ridge(Least_MSE_alpha)
    best_trained_model.fit(X[:, sel_F[MSE_index]], Y)
    sel_idx = sel_F[MSE_index]

    # Since the data for classsifierselection is too small, we only calculate the train error

    X_train = pd.DataFrame.as_matrix(train_features)
    Y_train = pd.DataFrame.as_matrix(train_labels)
    X_test = pd.DataFrame.as_matrix(test_features)
    Y_test = pd.DataFrame.as_matrix(test_labels)

    if data_conf["input_name"] == "classifierselection":
        y_pred_test = []
    else:
        y_pred_test = best_trained_model.predict(X_test[:, sel_idx])

    y_pred_train = best_trained_model.predict(X_train[:, sel_idx])

    return cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test


def mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler):
    """This function concatenates TR input and TR output, and TE input and TE output (one time it considers real
    (scaled) TR and TE, then it considers predicted TR and TE output for concatenation),  once the data has the same
    number of columns when it was scaled, performs inverse scaling to compute the unscaled predicted output. Then MAPE
    error is computed using the unscaled data."""

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
        for elem in features_names:
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
    for elem in features_names:
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


def get_result_name(degree, select_features_sfs, k_features, is_floating):
    """Produces a meaningful name for creating a folder in order to save the diagrams and variables inside the
    result path"""
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
    result_name = result_name+'_' +str(run_num) +'_runs'

    Tr_size = '_Tr'
    for sz in image_nums_train_data:
        Tr_size = Tr_size+'_'+str(sz)

    Te_size = '_Te'
    for sz in image_nums_test_data:
        Te_size = Te_size + '_' + str(sz)

    result_name = result_name+Tr_size+Te_size

    return result_name


def save_results(err_train, err_test, result_name, result_path, data_conf, cv_info, ridge_params, best_trained_model,
                               degree, Least_MSE_alpha):
    """it saves the results in a variabl and makes the directory if not exist"""
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

def plot_predicted_true(result_path, run_info, best_run_idx):
    """Plots the predicted values of output, as opposed to the real values in TR and TE sets"""

    data_conf = run_info[best_run_idx]['data_conf']

    y_true_train = run_info[best_run_idx]['y_true_train']
    y_pred_train = run_info[best_run_idx]['y_pred_train']
    y_true_test = run_info[best_run_idx]['y_true_test']
    y_pred_test = run_info[best_run_idx]['y_pred_test']


    params_txt = 'best alpha: ' + str(run_info[best_run_idx]['best_param'])
    font = {'family': 'normal', 'size': 15}
    matplotlib.rc('font', **font)
    plot_path = os.path.join(result_path, "True_Pred_Plot")
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

def plot_cores_runtime(result_path, run_info, best_run_idx, core_nums_train_data, core_nums_test_data):
    """Plot the real and predicted values of output based on the values of cores number in the corresponding samples"""
    data_conf = run_info[best_run_idx]['data_conf']


    y_true_train = run_info[best_run_idx]['y_true_train']
    y_pred_train = run_info[best_run_idx]['y_pred_train']
    y_true_test = run_info[best_run_idx]['y_true_test']
    y_pred_test = run_info[best_run_idx]['y_pred_test']


    font = {'family': 'normal', 'size': 15}
    matplotlib.rc('font', **font)
    plot_path = os.path.join(result_path, "cores_runtime_plot")
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig = plt.figure(figsize=(9, 6))
    #if self.data_conf["fixed_features"] == False:


    params_txt = 'best alpha: ' + str(run_info[best_run_idx]['best_param'])
    regressor_name = 'Logistic Regression'

    core_num_indices = pd.DataFrame(
        [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
        columns=['col', 'indices'])
    # Training
    # This part is for making the legend
    legcount1 = 0
    for Trcore in core_nums_train_data:

        # DF of samples having the core number equal to Trcore
        y_idx = core_num_indices.loc[core_num_indices['col'] == Trcore]['indices']

        # convert them to list
        y_idx_list = y_idx.iloc[0].tolist() # no need to iterate
        #y_tr_true = []
        #y_tr_pred = []

        # scatter the points of different core numbers in TR set
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

    # This part is for making the legend
    legcount2 = 0
    for Tecore in core_nums_test_data:
        # DF of samples having the core number equal to Tecore
        y_idx_te = core_num_indices.loc[core_num_indices['col'] == Tecore]['indices']

        # convert them to list
        y_idx_te_list = y_idx_te.iloc[0].tolist() # no need to iterate

        # scatter the points of different core numbers in TE set

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
    """given true and predicted values in 2 vectors (scaled version), calculate the MSE error for the sole use in
    the cross validation and grid search"""
    MSE = np.mean((Y_hat - Y) ** 2)
    return MSE

def calcMAPE(Y_hat, Y):
    """given true and predicted values in 2 vectors (scaled version), calculate the MAPE error for the sole use in
    the cross validation and grid search"""
    Mapeerr = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    return Mapeerr


def select_best_run(run_info):
    """This function selects the best run among independent runs according to the least MAPE error on Test set"""
    Mape_list = []
    for i in range(len(run_info)):
        Mape_list.append(run_info[i]['MAPE_test'])

    best_run_idx = Mape_list.index(min(Mape_list))
    best_data_conf = run_info[best_run_idx]['data_conf']
    best_cv_info = run_info[best_run_idx]['cv_info']
    best_trained_model = run_info[best_run_idx]['best_model']
    best_Least_MSE_alpha = run_info[best_run_idx]['best_param']
    best_err_train = run_info[best_run_idx]['MAPE_train']
    best_err_test = run_info[best_run_idx]['MAPE_test']

    return best_run_idx, best_data_conf, best_cv_info, best_trained_model, best_Least_MSE_alpha, best_err_train, best_err_test


def plot_histogram(result_path, run_info, degree):
    """Plot the histogram of features selection frequency in independent runs"""

    plot_path = os.path.join(result_path, "Features_Ferquency_Histogram_plot")

    names_list = run_info[0]['ext_feature_names']
    name_count = []
    for i in range(len(names_list)):
        name_count.append(0)
    iternum = len(run_info)

    for i in range(iternum):
        for j in run_info[i]['Sel_features']:
            name_count[j] += 1

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




def plot_MSE_Errors(result_path, run_info):
    """Plot the MSE error of unscaled version in TR and TE set in independent runs"""

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


def plot_MAPE_Errors(result_path, run_info):
    """Plot the MAPE error of unscaled version in TR and TE set in independent runs"""

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


def plot_Model_Size(result_path, run_info):
    """Plot the selected model size in independent runs"""

    plot_path = os.path.join(result_path, "Model_Size_Plot")
    model_size_list = []
    for i in range(len(run_info)):
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


def Screening2(df, target_column_idx, Confidence_level):

    Y = df.iloc[:, target_column_idx]
    Y = pd.DataFrame.as_matrix(Y)

    output_name = df.columns.values[target_column_idx]
    X = df.loc[:, df.columns != output_name]
    X = pd.DataFrame.as_matrix(X)


    independence_list = []
    if X.shape[0] == Y.shape[0]:
        for f in range(X.shape[1]):
            independence_list.append(independentTest(X[:, f], Y, Confidence_level))

    irrelevant_col = [col for col, e in enumerate(independence_list) if e == 1]
    return irrelevant_col


def Screening(train_features, train_labels, Confidence_level):
    Y = train_labels
    Y = pd.DataFrame.as_matrix(Y)

    X = train_features
    X = pd.DataFrame.as_matrix(X)

    independence_list = []
    if X.shape[0] == Y.shape[0]:
        for f in range(X.shape[1]):
            independence_list.append(independentTest(X[:, f], Y, Confidence_level))

    irrelevant_col = [col for col, e in enumerate(independence_list) if e == 1]
    return irrelevant_col



def independentTest(x,y, Confidence_level):


    N = x.shape[0]
    x = x.reshape(N, 1)
    y = y.reshape(N, 1)

    xx = np.tile(x, [1,N])
    xxp = xx.transpose()
    diff_x = xx - xxp
    norm_x = np.sqrt(diff_x**2)

    temp1_x = norm_x.mean(0).reshape((1, N))
    temp2_x = np.tile(temp1_x, [N, 1])

    temp3_x = norm_x.mean(1).reshape((N, 1))
    temp4_x = np.tile(temp3_x, [1, N])

    X_MATRIX = norm_x - temp2_x - temp4_x + norm_x.mean()

    yy = np.tile(y, [1,N])
    yyp = yy.transpose()
    diff_y = yy - yyp
    norm_y = np.sqrt(diff_y**2)

    temp1_y = norm_y.mean(0).reshape((1, N))
    temp2_y = np.tile(temp1_y, [N, 1])

    temp3_y = norm_y.mean(1).reshape((N, 1))
    temp4_y = np.tile(temp3_y, [1, N])

    Y_MATRIX = norm_y - temp2_y - temp4_y + norm_y.mean()

    XYPair = np.multiply(X_MATRIX, Y_MATRIX)

    Vxy = XYPair.mean()

    alpha = 1 - Confidence_level
    T = N * Vxy / (norm_x.mean() * norm_y.mean())
    P = T <= (stats.norm.ppf(1 - alpha / 2)**2)

    return 1 if P else 0

def dCorRanking(train_features, train_labels):
    Y = train_labels
    Y = pd.DataFrame.as_matrix(Y)

    X = train_features
    X = pd.DataFrame.as_matrix(X)

    nancount = 0
    rel_ranking_list = []
    if X.shape[0] == Y.shape[0]:
        for f in range(X.shape[1]):
            rel_ranking_list.append(rel_rank(X[:, f], Y))
            if math.isnan(rel_rank(X[:, f], Y)):
                nancount += 1
    print(nancount)
    return rel_ranking_list


def rel_rank(x, y):

    N = x.shape[0]
    x = x.reshape(N, 1)
    y = y.reshape(N, 1)

    xx = np.tile(x, [1,N])
    xxp = xx.transpose()
    diff_x = xx - xxp
    norm_x = np.sqrt(diff_x**2)

    temp1_x = norm_x.mean(0).reshape((1, N))
    temp2_x = np.tile(temp1_x, [N, 1])

    temp3_x = norm_x.mean(1).reshape((N, 1))
    temp4_x = np.tile(temp3_x, [1, N])

    X_MATRIX = norm_x - temp2_x - temp4_x + norm_x.mean()

    yy = np.tile(y, [1,N])
    yyp = yy.transpose()
    diff_y = yy - yyp
    norm_y = np.sqrt(diff_y**2)

    temp1_y = norm_y.mean(0).reshape((1, N))
    temp2_y = np.tile(temp1_y, [N, 1])

    temp3_y = norm_y.mean(1).reshape((N, 1))
    temp4_y = np.tile(temp3_y, [1, N])

    Y_MATRIX = norm_y - temp2_y - temp4_y + norm_y.mean()

    XYPair = np.multiply(X_MATRIX, Y_MATRIX)

    Vxy = XYPair.mean()

    # alpha = 1 - Confidence_level
    T = N * Vxy / (norm_x.mean() * norm_y.mean())
    # P = T <= (stats.norm.ppf(1 - alpha / 2)**2)
    if math.isnan(T):
        print('norm x mean: ', norm_x.mean())
        print('norm y mean: ', norm_y.mean())

    return T




# Set the seed vector for performing the shuffling in different runs
# seed_v = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 1023]

# read the csv data into a pandas DataFrame
df = read_inputs()

# invert the columns specifiying in inverting columns
df, inversing_cols = add_inverse_features(df, to_be_inv_List)

# irrelevant_features = Screening2(df, 0, 0.999999)


# extend the features
ext_df = add_all_comb(df, inversing_cols, 0, degree)


# The list for keeping independent runs information
run_info = []
for iter in range(run_num):

    this_run = 'run_'+str(iter)
    print(this_run)

    run_info.append({})

    # shuffle the samples and split the data
    train_features, train_labels, test_features, test_labels, features_names, scaler, data_conf = \
        split_data(seed_v[iter], ext_df, image_nums_train_data, image_nums_test_data, core_nums_train_data, core_nums_test_data)

    run_info[iter]['ext_feature_names'] = features_names
    run_info[iter]['data_conf'] = data_conf

    # computes the range of final model size
    k_features = calc_k_features(min_features, max_features, features_names)

    print('selecting features in range ', k_features, ':')


    # screening:
    irrelevant_features = Screening(train_features, train_labels, 0.999999)
    print(len(irrelevant_features), ' irrelevant features')

    #run_info[iter]['irrelevant_features'] = irrelevant_features
    irrelevant_features_names = [features_names[i] for i in irrelevant_features]
    #relevant_features_idx = list(set(range(len(features_names)))-set(irrelevant_features))

    #relevant_features_names = [features_names[i] for i in relevant_features_idx]

    #ranking_list = dCorRanking(train_features, train_labels)


    # train_features = train_features.drop(train_features.columns[irrelevant_features], axis=1)
    # test_features = test_features.drop(test_features.columns[irrelevant_features], axis=1)

    # train_features = train_features.loc[:, relevant_features_names]
    # test_features = test_features.loc[:, relevant_features_names]





    # Grid search
    cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test = \
        Ridge_SFS_GridSearch(ridge_params, train_features, train_labels, test_features, test_labels, k_features, fold_num)


    run_info[iter]['cv_info'] = cv_info
    run_info[iter]['Sel_features'] = list(sel_idx)
    run_info[iter]['Sel_features_names'] = [features_names[i] for i in sel_idx]
    run_info[iter]['best_param'] = Least_MSE_alpha
    run_info[iter]['best_model'] = best_trained_model

    # compute MAPE training and test error for uscaled data
    err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores = \
        mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler)

    run_info[iter]['MAPE_train'] = err_train
    run_info[iter]['MAPE_test'] = err_test
    run_info[iter]['y_true_train'] = y_true_train
    run_info[iter]['y_pred_train'] = y_pred_train
    run_info[iter]['y_true_test'] = y_true_test
    run_info[iter]['y_pred_test'] = y_pred_test


# computes the best run based on minimum MAPE on test set
best_run_idx, best_data_conf, best_cv_info, best_trained_model, best_Least_MSE_alpha, best_err_train, best_err_test = \
    select_best_run(run_info)

# get the result folder name
result_name = get_result_name(degree, select_features_sfs, k_features, is_floating)

# create the path and save variables
result_path, results = save_results(best_err_train, best_err_test, result_name, result_path,
                                    best_data_conf, best_cv_info, ridge_params, best_trained_model, degree, best_Least_MSE_alpha)


# plot the diagrams
plot_predicted_true(result_path, run_info, best_run_idx)
plot_cores_runtime(result_path, run_info, best_run_idx, core_nums_train_data, core_nums_test_data)
plot_histogram(result_path, run_info, degree)
plot_MSE_Errors(result_path, run_info)
plot_MAPE_Errors(result_path, run_info)
plot_Model_Size(result_path, run_info)

# save run information variable for later use
target = open(os.path.join(result_path, "run_info"), 'a')
target.write(str(run_info))
target.close()

end = time.time()

# computes and report the execution time
execution_time = str(end-start)
print("Execution Time : " + execution_time)




