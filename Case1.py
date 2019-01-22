import sys
sys.modules[__name__].__dict__.clear()

import pandas as pd
import numpy as np
import math
import itertools
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import json

# different input options:

#input_path = 'P8_kmeans.csv'
#input_path = 'dumydata2.csv'
#input_path = 'yourfile.csv'
#input_path = 'dumydata.csv'
#input_path = "newdummy.csv"
input_path = "newd.csv"


# variables:
# TR and TE variables based on which the samples should be splitted
image_nums_train_data = [5]
image_nums_test_data = [5]
core_nums_train_data = [6,10,14,18,22,26,30,34,38,42,46]
core_nums_test_data = [8,12,16,20,24,28,32,36,40,44,48]

# Feature selection related variables
select_features_vif = False
select_features_sfs = True
min_features = 2
max_features = 2
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

test_without_apriori = False

# preprocessing related variables (DT)
inversing = True
inverseColNameList = ['nContainers']
extension = True
degree = 2



def calculate_new_col(X, indices):
    """Given two indices and input matrix returns the multiplication of the columns corresponding columns"""
    index = 0
    new_col = X[:, indices[index]]
    for ii in list(range(1, len(indices))):
        new_col = np.multiply(new_col, X[:, indices[index + 1]])
        index += 1
    return new_col


def add_all_comb(inv_train_df, inversed_cols_tr, degree):
    """Given a dataframe, returns an extended df containing all combinations of columns except the ones that are
    inversed"""

    features_names = inv_train_df.columns.values
    df_dict = dict(inv_train_df)
    data_matrix = pd.DataFrame.as_matrix(inv_train_df)
    indices = list(range(data_matrix.shape[1]))

    # compute all possible combinations with replacement
    for j in range(2, degree + 1):
        combs = list(itertools.combinations_with_replacement(indices, j))
        # removes the combinations containing features and inversed of them
        for ii in combs:
            for kk in inversed_cols_tr:
                if len(list(set.intersection(set(ii), set(kk)))) >= 2:
                    combs.remove(ii)

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


def add_inverse_features(df, colNameList):
    """Given a dataframe and the name of columns that should be inversed, add the needed inversed columns and returns
    the resulting df and the indices of two reciprocals separately"""

    df_dict = dict(df)
    for c in colNameList:
        new_col = 1/np.array(df[c])
        new_feature_name = 'inverse_'+c
        df_dict[new_feature_name] = new_col

    inv_df = pd.DataFrame.from_dict(df_dict)

    # returns the indices of the columns that should be inversed and their inversed in one tuple
    inversing_cols = []
    for c in colNameList:
        cidx = inv_df.columns.get_loc(c)
        cinvidx = inv_df.columns.get_loc('inverse_'+c)
        inv_idxs = (cidx, cinvidx)
        inversing_cols.append(inv_idxs)
    return inv_df, inversing_cols

def myNorm(df, col_name_list):
    """Normalizes the df all columns values between 0 and 1"""

    df1 = df
    df_dict = dict(df1)
    feature_name = col_name_list
    data_matrix = pd.DataFrame.as_matrix(df1)

    # find the min and max of the whole matrix
    dfmin = data_matrix.min()
    dfmax = data_matrix.max()
    scale_factor = dfmax - dfmin

    # scale all columns based on the obtained scale_factor
    for f in feature_name:
        df1[f] = np.array(df1[f]) - dfmin
        df1[f] = df1[f] / scale_factor
    return df1, scale_factor


def mean_absolute_percentage_error(y_pred_test, test_labels, y_pred_train, train_labels):
    """given true and predicted values returns MAPE error for training and test set"""

    if y_pred_test != []:
        # Test error
        y_true_test = test_labels
        test_features_with_true = pd.DataFrame(test_features_org)
        test_features_with_pred = pd.DataFrame(test_features_org)

        y_true_test = pd.DataFrame(y_true_test)
        y_pred_test = pd.DataFrame(y_pred_test)

        test_features_with_true.insert(0, "y_true_test", y_true_test)
        test_features_with_pred.insert(0, "y_pred_test", y_pred_test)

        test_data_with_true = pd.DataFrame(scaler.inverse_transform(test_features_with_true.values))
        test_data_with_pred = pd.DataFrame(scaler.inverse_transform(test_features_with_pred.values))

        for col in test_data_with_true:
            cores = test_data_with_true[col].unique().tolist()
            cores = list(map(lambda x: int(x), cores))
            if set(cores) == set(core_nums_test_data):
                    y_true_test_cores = test_data_with_true[col].tolist()
        for col in test_data_with_pred:
            cores = test_data_with_pred[col].unique().tolist()
            cores = list(map(lambda x: int(x), cores))
            if set(cores) == set(core_nums_test_data):
                y_pred_test_cores = test_data_with_pred[col].tolist()
        y_true_test = test_data_with_true.iloc[:, 0]
        y_pred_test = test_data_with_pred.iloc[:, 0]

        err_test = np.mean(np.abs((y_true_test - y_pred_test) / y_true_test)) * 100
    if y_pred_test == []:
        err_test = -1

    # Train error
    y_true_train = train_labels
    train_features_with_true = pd.DataFrame(train_features_org)
    train_features_with_pred = pd.DataFrame(train_features_org)

    y_true_train = pd.DataFrame(y_true_train)
    y_pred_train = pd.DataFrame(y_pred_train)

    train_features_with_true.insert(0, "y_true_train", y_true_train)
    train_features_with_pred.insert(0, "y_pred_train", y_pred_train)

    train_data_with_true = pd.DataFrame(scaler.inverse_transform(train_features_with_true.values))
    train_data_with_pred = pd.DataFrame(scaler.inverse_transform(train_features_with_pred.values))

    for col in train_data_with_true:
        cores = train_data_with_true[col].unique().tolist()
        cores = list(map(lambda x: int(x), cores))
        if set(cores) == set(core_nums_train_data):
            y_true_train_cores = train_data_with_true[col].tolist()
    for col in train_data_with_pred:
        cores = train_data_with_pred[col].unique().tolist()
        cores = list(map(lambda x: int(x), cores))
        if set(cores) == set(core_nums_train_data):
            y_pred_train_cores = train_data_with_pred[col].tolist()

    y_true_train = train_data_with_true.iloc[:, 0]
    y_pred_train = train_data_with_pred.iloc[:, 0]

    err_train = np.mean(np.abs((y_true_train - y_pred_train) / y_true_train)) * 100
    return err_test, err_train

def calcMAPE(Y_hat_test, Y_test):
    """given true and predicted values returns MAPE error"""
    Mapeerr = np.mean(np.abs((Y_test - Y_hat_test) / Y_test)) * 100
    return Mapeerr

def scale_data(df):
    """scale and normalized the data using standard scaler and returns the scaled data, the scaler object, the mean
    value and std of the output column for later use"""

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)

    # obtain the mean and std of the standard scaler using the formula z = (x-u)/s
    # if u is the mean and s is the std, z is the scaled version of the x


    output_scaler_mean = scaler.mean_
    output_scaler_std = (df.values[0][0] - output_scaler_mean[0])/ scaled_array[0][0]

    # return the mean and std for later use
    # return scaled_df, scaler, output_scaler_mean[0], output_scaler_std
    return scaled_df, scaler, df.mean(), df.std()



# read the data from .csv file:
df = pd.read_csv(input_path)



# drop run column
tempcols = ["run","gg"]
df = df.drop(tempcols, axis=1)

# compute the matrix and name of columns as features
data_matrix = pd.DataFrame.as_matrix(df)

# save the output
output_df = df['applicationCompletionTime']


# remove constant valued columns
df = df.loc[:, (df != df.iloc[0]).any()]


# randomize the samples
seed = 1234
#df = shuffle(df, random_state = seed)


# Dictionary keeping the information about the input and training and test samples
data_conf = {}
data_conf["case"] = "same datasize in TR and TE_even cores in TR, odds in TE"
data_conf["input_name"] = "K_means"

# Separate the training and test sets based on the datasize
# case1:
# Training and test samples have the same datasizes:

# locating the indices of traing and test samples based on datasize and core numbers:
"""################################################ Datasize indices #######################################"""
if "dataSize" in df.columns:
    data_size_indices = pd.DataFrame([[k, v.values] for k, v in df.groupby('dataSize').groups.items()],
                                          columns=['col', 'indices'])

    data_size_train_indices = \
    data_size_indices.loc[(data_size_indices['col'].isin(image_nums_train_data))]['indices']
    data_size_test_indices = \
    data_size_indices.loc[(data_size_indices['col'].isin(image_nums_test_data))]['indices']

    data_size_train_indices = np.concatenate(list(data_size_train_indices), axis=0)
    data_size_test_indices = np.concatenate(list(data_size_test_indices), axis=0)

else:

    data_size_train_indices = range(0, df.shape[0])
    data_size_test_indices = range(0, df.shape[0])

# save the info about training and test datasize in the data dictionary
data_conf["image_nums_train_data"] = image_nums_train_data
data_conf["image_nums_test_data"] = image_nums_test_data

"""################################################ Core number indices #######################################"""

core_num_indices = pd.DataFrame([[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
                                     columns=['col', 'indices'])


# For interpolation and extrapolation, put all the cores to the test set.
if set(image_nums_train_data) != set(image_nums_test_data):
    core_nums_test_data = core_nums_test_data + core_nums_train_data

core_num_train_indices = \
    core_num_indices.loc[(core_num_indices['col'].isin(core_nums_train_data))]['indices']
core_num_test_indices = \
    core_num_indices.loc[(core_num_indices['col'].isin(core_nums_test_data))]['indices']

core_num_train_indices = np.concatenate(list(core_num_train_indices), axis=0)
core_num_test_indices = np.concatenate(list(core_num_test_indices), axis=0)

# save the info about training and test core numbers in the data dictionary
data_conf["core_nums_train_data"] = core_nums_train_data
data_conf["core_nums_test_data"] = core_nums_test_data

################## get together all the TR and TE indices:
train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)


"""############################################## Normalization ###############################################"""
######## scale the data
scaled_df, scaler, scaler_mean_df, scaler_std_df = scale_data(df)


"""################################################ Inversing #################################################"""

# adding the inverse of nContainers columns to the data
if inversing == True:
    inv_df, inversed_cols_tr = add_inverse_features(scaled_df, inverseColNameList)


# check the feature names after inversing
inv_feature_names = inv_df.columns.values


"""################################################ Extension #################################################"""

# Extend the df by adding combinations of features as new features to the df
if extension == True:

    ext_train_features_df = add_all_comb(inv_train_df, inversed_cols_tr, degree)
    ext_test_features_df = add_all_comb(inv_test_df, inversed_cols_te, degree)


# check the feature names after extension
ext_feature_names = ext_train_features_df.columns.values

data_conf["ext_feature_names"] = ext_feature_names.tolist()








"""################################################ Splitting #################################################"""


# splitting the data
train_df = scaled_df.ix[train_indices]
test_df = scaled_df.ix[test_indices]
train_labels = train_df.iloc[:, 0]
train_features = train_df.iloc[:, 1:]
test_labels = test_df.iloc[:, 0]
test_features = test_df.iloc[:, 1:]






"""##########################  SFS and Hyper Params tunning using grid search and CV ##########################"""

# computing the range of accepted number of features: (k_features)
if select_features_sfs == True:

    min_k_features = int(min_features)
    max_k_features = int(max_features)
    # Selecting from all features
    if max_k_features == -1:
        k_features = (min_k_features, ext_train_features_df.shape[1])
        # Selecting from the given range
    if max_k_features != -1:
        k_features = (min_k_features, max_k_features)



# finding the best model using Kfold-CV and just the traing set

fold_num = fold_num

# determining the size of data
feature_num = ext_train_features_df.shape[1]
sample_num = ext_train_features_df.shape[0]

# vector of parameters to be search through
alpha_v = ridge_params


# input and output preparation to use in CV iterations
X = pd.DataFrame.as_matrix(ext_train_features_df)
Y = pd.DataFrame.as_matrix(train_labels)


# list of mean value obtained scores of the selected parameter (alpha) :
param_overal_scores = []

# list of mean value RSE error for different alpha values:
param_overal_RSE = []

# list of mean value of obtained SFS scores of different alpha values:
sfs_overal_scores = []

# list of mean value of MAPE in different alpha values:
Mape_overal_error = []

# Dictionary keeping information about all scores and values and selected features in all iterations for all params
cv_info = {}

# Selected features for each alpha
sel_F = []

# Selected features names for each alpha
sel_F_names= []


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

    # fit the sfs on training part and evaluate the score on test part of this fold
    sfs = sfs.fit(X, Y)
    sel_F_idx = sfs.k_feature_idx_
    sel_F.append(sel_F_idx)
    cv_info[this_a]['Selected_Features'] = list(sel_F_idx)

    sel_F_names.append(ext_feature_names[list(sel_F_idx)].tolist())
    cv_info[this_a]['Selected_Features_Names'] = ext_feature_names[list(sel_F_idx)].tolist()

    # keep the sfs score for this alpha
    sfs_overal_scores.append(sfs.k_score_)
    cv_info[this_a]['sfs_score'] = sfs.k_score_

    # fit the ridge model with the selected features and evaluate the ridge score
    # This score is one the whole training data only
    ridge.fit(X[:, sfs.k_feature_idx_], Y)
    ridge_score = ridge.score(X[:, sfs.k_feature_idx_], Y)
    param_overal_scores.append(ridge_score)
    cv_info[this_a]['ridge_score'] = ridge_score

    # evaluate the RSE error on the whole training data only using the selected features
    Y_hat = ridge.predict(X[:, sfs.k_feature_idx_])
    RSE = math.sqrt(sum((Y_hat - Y) ** 2))
    param_overal_RSE.append(RSE)
    cv_info[this_a]['RSE_error'] = RSE

    # evaluate the MAPE error on the whole training data only using the selected features
    Mape_error = calcMAPE(Y_hat, Y)
    Mape_overal_error.append(Mape_error)
    cv_info[this_a]['MAPE_error'] = Mape_error
    print('alpha = ', a, '    sfs_scores = ', sfs.k_score_, '    ridge_score = ', ridge_score, '    RSE Error= ', RSE,
          '    MAPE Error = ', Mape_error)


"""################################################## Results #####################################################"""

# select the best alpha based on obtained values
SFS_index = sfs_overal_scores.index(max(sfs_overal_scores))
alpha_score_index = param_overal_scores.index(max(param_overal_scores))
RSE_index = param_overal_RSE.index(min(param_overal_RSE))
MAPE_index = Mape_overal_error.index(min(Mape_overal_error))


# report the best alpha based on obtained values
print('Best_sfs_score_index = ', SFS_index, ' => Best_sfs_score_alpha = ', alpha_v[SFS_index])
print('Best_ridge_score_index = ', alpha_score_index, ' => Best_ridge_score_alpha = ', alpha_v[alpha_score_index])
print('Least_RSE_Error_index = ', RSE_index, ' => Least_RSE_Error_alpha = ', alpha_v[RSE_index])
print('Least_MAPE_Error_index = ', MAPE_index, ' => Least_MAPE_Error_alpha = ', alpha_v[MAPE_index])


# prepare the data for final error on test dataset based on different
X_train = pd.DataFrame.as_matrix(ext_train_features_df)
Y_train = pd.DataFrame.as_matrix(train_labels)
X_test = pd.DataFrame.as_matrix(ext_test_features_df)
Y_test = pd.DataFrame.as_matrix(test_labels)


print('......................................Results based on Ridge score index...............................')
'''############################################ Ridge score index ####################################################'''

# alpha score computations for the model consisting of the selected features:
print('Best alpha by selecting model having highest ridge score in grid search = ', alpha_v[alpha_score_index])
print('Selected features based on classifier maximum score: ', sel_F[alpha_score_index])
print('Selected features names based on classifier maximum score: ', sel_F_names[alpha_score_index])
print('Model size on classifier maximum score: ', len(sel_F[alpha_score_index]))


# predict training and test data using the model suggested by best ridge score
best_ridge_score_model = Ridge(alpha_v[alpha_score_index])
best_ridge_score_model.fit(X_train[:, sel_F[alpha_score_index]], Y_train)
Y_hat_training = best_ridge_score_model.predict(X_train[:, sel_F[alpha_score_index]])
Y_hat_test = best_ridge_score_model.predict(X_test[:, sel_F[alpha_score_index]])

# compute the RSE error
Errortraining = math.sqrt(sum((Y_hat_training-Y_train)**2))
realErrortraining = Errortraining*output_scaler_std

Errortest = math.sqrt(sum((Y_hat_test-Y_test)**2))
realErrortest = Errortest*output_scaler_std

# report the RSE and MAPE error
print('RSE Error (Training) = ', realErrortraining)
print('RSE Error (Test) = ', realErrortest)
print('MAPE Error (Training) = ', calcMAPE(Y_hat_training, Y_train))
print('MAPE Error (Test) = ', calcMAPE(Y_hat_test, Y_test))


# save info about model in the data variable
data_conf['Ridge_Score'] = {}
data_conf['Ridge_Score']['alpha'] = alpha_v[alpha_score_index]
data_conf['Ridge_Score']['Sel_features'] = list(sel_F[alpha_score_index])
data_conf['Ridge_Score']['Sel_features_name'] = sel_F_names[alpha_score_index]
data_conf['Ridge_Score']['Model_size'] = len(sel_F[alpha_score_index])
data_conf['Ridge_Score']['RSE_Error_TR'] = realErrortraining
data_conf['Ridge_Score']['RSE_Error_TE'] = realErrortest
data_conf['Ridge_Score']['MAPE_Error_Training'] = calcMAPE(Y_hat_training, Y_train)
data_conf['Ridge_Score']['MAPE_Error_Test'] = calcMAPE(Y_hat_test, Y_test)


print('.......................................Results based on SFS score index.................................')
'''############################################ SFS score index ####################################################'''

# alpha score computations for the model consisting of the selected features:
print('Best alpha by selecting model having highest SFS score in grid search = ', alpha_v[SFS_index])
print('Selected features based on SFS maximum score: ', sel_F[SFS_index])
print('Selected features names based on SFS maximum score: ', sel_F_names[SFS_index])
print('Model size on SFS maximum score: ', len(sel_F[SFS_index]))


# predict training and test data using the model suggested by best sfs score
best_sfs_score_model = Ridge(alpha_v[SFS_index])
best_sfs_score_model.fit(X_train[:, sel_F[SFS_index]], Y_train)
Y_hat_training = best_sfs_score_model.predict(X_train[:, sel_F[SFS_index]])
Y_hat_test = best_sfs_score_model.predict(X_test[:, sel_F[SFS_index]])

# compute the RSE error
Errortraining = math.sqrt(sum((Y_hat_training-Y_train)**2))
realErrortraining = Errortraining*output_scaler_std

Errortest = math.sqrt(sum((Y_hat_test-Y_test)**2))
realErrortest = Errortest*output_scaler_std

# report the RSE and MAPE error
print('RSE Error (Training) = ', realErrortraining)
print('RSE Error (Test) = ', realErrortest)
print('MAPE Error (Training) = ', calcMAPE(Y_hat_training, Y_train))
print('MAPE Error (Test) = ', calcMAPE(Y_hat_test, Y_test))


# save info about model in the data variable
data_conf['SFS_Score'] = {}
data_conf['SFS_Score']['alpha'] = alpha_v[alpha_score_index]
data_conf['SFS_Score']['Sel_features'] = list(sel_F[alpha_score_index])
data_conf['SFS_Score']['Sel_features_name'] = sel_F_names[alpha_score_index]
data_conf['SFS_Score']['Model_size'] = len(sel_F[alpha_score_index])
data_conf['SFS_Score']['RSE_Error_TR'] = realErrortraining
data_conf['SFS_Score']['RSE_Error_TE'] = realErrortest
data_conf['SFS_Score']['MAPE_Error_Training'] = calcMAPE(Y_hat_training, Y_train)
data_conf['SFS_Score']['MAPE_Error_Test'] = calcMAPE(Y_hat_test, Y_test)



'''############################################ RSE ####################################################'''
print('.......................................Results based on RSE.............................................')
# RSE error computations for the model consisting of the selected features:
print('Best alpha by selecting model having minimum RSE error in the grid search = ', alpha_v[RSE_index])
print('Selected features based on minimum RSE error: ', sel_F[RSE_index])
print('Selected features names based on minimum RSE error: ', sel_F_names[RSE_index])
print('Model size based on minimum RSE error: ', len(sel_F[RSE_index]))


# predict training and test data using the model suggested by least RSE error
least_RSE_model = Ridge(alpha_v[RSE_index])
least_RSE_model.fit(X_train[:, sel_F[RSE_index]], Y_train)

Y_hat_training = least_RSE_model.predict(X_train[:, sel_F[RSE_index]])
Y_hat_test = least_RSE_model.predict(X_test[:, sel_F[RSE_index]])

# compute the RSE error
Errortraining = math.sqrt(sum((Y_hat_training-Y_train)**2))
realErrortraining = Errortraining*output_scaler_std

Errortest = math.sqrt(sum((Y_hat_test-Y_test)**2))
realErrortest = Errortest*output_scaler_std



# report the RSE and MAPE error
print('RSE Error (Training)= ', realErrortraining)
print('RSE Error (Test) = ', realErrortest)
print('MAPE Error (Training) = ', calcMAPE(Y_hat_training, Y_train))
print('MAPE Error (Test) = ', calcMAPE(Y_hat_test, Y_test))


# save info about model in the data variable
data_conf['Min_RSE'] = {}
data_conf['Min_RSE']['alpha'] = alpha_v[RSE_index]
data_conf['Min_RSE']['Sel_features'] = list(sel_F[RSE_index])
data_conf['Min_RSE']['Sel_features_name'] = sel_F_names[RSE_index]
data_conf['Min_RSE']['Model_size'] = len(sel_F[RSE_index])
data_conf['Min_RSE']['RSE_Error_TR'] = realErrortraining
data_conf['Min_RSE']['RSE_Error_TE'] = realErrortest
data_conf['Min_RSE']['MAPE_Error_Training'] = calcMAPE(Y_hat_training, Y_train)
data_conf['Min_RSE']['MAPE_Error_Test'] = calcMAPE(Y_hat_test, Y_test)


'''############################################ MAPE ####################################################'''
print('....................................Results based on MAPE............................................')
# RSE error computations for the model consisting of the selected features:
print('Best alpha by selecting model having minimum MAPE error in the grid search = ', alpha_v[MAPE_index])
print('Selected features based on minimum MAPE error: ', sel_F[MAPE_index])
print('Selected features names based on minimum MAPE error: ', sel_F_names[MAPE_index])
print('Model size based on minimum MAPE error: ', len(sel_F[MAPE_index]))


# predict training and test data using the model suggested by least MAPE error
least_MAPE_model = Ridge(alpha_v[MAPE_index])
least_MAPE_model.fit(X_train[:, sel_F[MAPE_index]], Y_train)

Y_hat_training = least_MAPE_model.predict(X_train[:, sel_F[MAPE_index]])
Y_hat_test = least_MAPE_model.predict(X_test[:, sel_F[MAPE_index]])

# compute the RSE error
Errortraining = math.sqrt(sum((Y_hat_training-Y_train)**2))
realErrortraining = Errortraining*output_scaler_std

Errortest = math.sqrt(sum((Y_hat_test-Y_test)**2))
realErrortest = Errortest*output_scaler_std


# report the RSE and MAPE error
print('RSE Error (Training)= ', realErrortraining)
print('RSE Error (Test) = ', realErrortest)
print('MAPE Error (Training) = ', calcMAPE(Y_hat_training, Y_train))
print('MAPE Error (Test) = ', calcMAPE(Y_hat_test, Y_test))


# save info about model in the data variable
data_conf['Min_MAPE'] = {}
data_conf['Min_MAPE']['alpha'] = alpha_v[MAPE_index]
data_conf['Min_MAPE']['Sel_features'] = list(sel_F[MAPE_index])
data_conf['Min_MAPE']['Sel_features_name'] = sel_F_names[MAPE_index]
data_conf['Min_MAPE']['Model_size'] = len(sel_F[MAPE_index])
data_conf['Min_MAPE']['RSE_Error_TR'] = realErrortraining
data_conf['Min_MAPE']['RSE_Error_TE'] = realErrortest
data_conf['Min_MAPE']['MAPE_Error_Training'] = calcMAPE(Y_hat_training, Y_train)
data_conf['Min_MAPE']['MAPE_Error_Test'] = calcMAPE(Y_hat_test, Y_test)
print('.........................................................................................')


# save the data in a file
with open('results.json', 'w') as fp:
    json.dump(data_conf, fp)
