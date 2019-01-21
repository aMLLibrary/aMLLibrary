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


# different input options:

#input_path = 'P8_kmeans.csv'
#input_path = 'dumydata2.csv'
#input_path = 'yourfile.csv'
input_path = 'dumydata.csv'


# variables:
image_nums_train_data = [5]
image_nums_test_data = [5]
core_nums_train_data = [6,10,14,18,22,26,30,34,38,42,46]
core_nums_test_data = [8,12,16,20,24,28,32,36,40,44,48]


select_features_vif = False
select_features_sfs = True
min_features = 1
max_features = 10
is_floating = False
fold_num = 5
FSfold_num = 2
LOOCV = False
regressor_name = "lr"
degree = 2

ridge_params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
lasso = '[0.1,0.5]'

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

inversing = True
inverseColNameList = ['nContainers']
extension = True



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
    for j in range(2, degree + 1):
        combs = list(itertools.combinations_with_replacement(indices, j))
        # removes the combinations containing features and inversed of them
        for ii in combs:
            for kk in inversed_cols_tr:
                if len(list(set.intersection(set(ii), set(kk)))) >= 2:
                    combs.remove(ii)

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
    dfmin = data_matrix.min()
    dfmax = data_matrix.max()
    scale_factor = dfmax - dfmin
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
    output_scaler_mean = scaler.mean_
    output_scaler_std = (df.values[0][0] - output_scaler_mean[0])/ scaled_array[0][0]
    return scaled_df, scaler, output_scaler_mean[0], output_scaler_std


# read the data from .csv file:
df = pd.read_csv(input_path)


# drop run column
tempcols = ["run"]
df = df.drop(tempcols, axis=1)

# compute the matrix and name of columns as features
data_matrix = pd.DataFrame.as_matrix(df)
features_names = list(df.columns.values)

# save the output
output_df = df['applicationCompletionTime']


# remove constant valued columns
df = df.loc[:, (df != df.iloc[0]).any()]


# randomize the samples
seed = 1234
df = shuffle(df, random_state = seed)


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

data_conf["core_nums_train_data"] = core_nums_train_data
data_conf["core_nums_test_data"] = core_nums_test_data

################## get together all the TR and TE indices:
train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)


"""############################################## Normalization ###############################################"""
######## scale the data
scaled_df, scaler, output_scaler_mean, output_scaler_std = scale_data(df)


"""################################################ Splitting #################################################"""


train_df = scaled_df.ix[train_indices]
test_df = scaled_df.ix[test_indices]
train_labels = train_df.iloc[:, 0]
train_features = train_df.iloc[:, 1:]
test_labels = test_df.iloc[:, 0]
test_features = test_df.iloc[:, 1:]


data_conf["reduced_features_names"] = list(train_df.columns.values)[1:]
data_conf["train_features_org"] = train_features.as_matrix()
data_conf["test_features_org"] = test_features.as_matrix()



"""################################################ Inversing #################################################"""

# adding the inverse of nContainers columns to the data
if inversing == True:
    inv_train_df, inversed_cols_tr = add_inverse_features(train_features, inverseColNameList)
    inv_test_df, inversed_cols_te = add_inverse_features(test_features, inverseColNameList)


# check the feature names after inversing
inv_feature_names = inv_train_df.columns.values


"""################################################ Extension #################################################"""

# Extend the df by adding combinations of features as new features to the df
if extension == True:

    ext_train_features_df = add_all_comb(inv_train_df, inversed_cols_tr, degree)
    ext_test_features_df = add_all_comb(inv_test_df, inversed_cols_te, degree)


# check the feature names after extension
ext_feature_names = ext_train_features_df.columns.values



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


k_fold = KFold(n_splits=fold_num, shuffle=False, random_state=None)

# input and output preparation to use in CV iterations
X = pd.DataFrame.as_matrix(ext_train_features_df)
Y = pd.DataFrame.as_matrix(train_labels)


# list of mean value obtained scores of the selected parameter (alpha) :
param_overal_scores = []

# list of mean value RSE error for different alpha values:
param_overal_error = []

# list of mean value of obtained SFS scores of different alpha values:
sfs_overal_scores = []

# list of mean value of MAPE in different alpha values:
Mape_overal_error = []

# Dictionary keeping information about all scores and values and selected features in all iterations for all params
cv_info = {}

for a in alpha_v:
    print('......')
    print('alpha = ', a)
    # list of ridge score of CV test data for different folds in current alpha
    score_list = []

    # list of RSE error of CV test data for different folds in current alpha
    error_list = []

    # list of MAPE error of CV test data for different folds in current alpha
    Mape_error_list = []

    # list of SFS error of CV test data for different folds in current alpha
    sfs_score_list = []

    # building the models
    ridge = Ridge(a)
    model = Ridge(a)
    this_a = 'alpha = '+str(a)
    cv_info[this_a] = {}

    for k, (train, test) in enumerate(k_fold.split(X, Y)):
        cv_info[this_a][str(k)] = {}
        # building the sfs
        sfs = SFS(clone_estimator=True,
                  estimator=model,
                  k_features=k_features,
                  forward=True,
                  floating=False,
                  scoring='neg_mean_squared_error',
                  cv=0,
                  n_jobs=-1)

        # fit the sfs on training part and evaluate the score on test part of this fold
        sfs = sfs.fit(X[train, :], Y[train])
        sel_F_idx = sfs.k_feature_idx_
        sfs_score_list.append(sfs.k_score_)
        cv_info[this_a][str(k)]['Selected_features_idx'] = sel_F_idx
        cv_info[this_a][str(k)]['sfs_scores'] = sfs.k_score_


        # fit the ridge model on training part and evaluate the ridge score on test part of this fold
        # Rows and columns selection should be done in different steps:
        xTRtemp = X[:, sfs.k_feature_idx_]
        ridge.fit(xTRtemp[train, :], Y[train])

        xTEtemp = X[:, sfs.k_feature_idx_]
        ridge_score = ridge.score(xTEtemp[test, :], Y[test])

        cv_info[this_a][str(k)]['ridge_score'] = ridge_score
        score_list.append(ridge_score)

        # evaluate the RSE error on test part of this fold
        Y_hat = ridge.predict(xTEtemp[test, :])
        sserror = math.sqrt(sum((Y_hat-Y[test])**2))
        cv_info[this_a][str(k)]['RSE'] = sserror
        error_list.append(sserror)

        # evaluate the MAPE error on test part of this fold
        Mape_error = calcMAPE(Y_hat, Y[test])
        cv_info[this_a][str(k)]['MAPE'] = Mape_error
        Mape_error_list.append(Mape_error)

        # report the progress in console
        print('fold = ', k, '    sfs_scores = ', sfs.k_score_, '    ridge_score = ', ridge_score, '    RSE = ', sserror,  '    MAPE = ', Mape_error)

    # computes the mean values of this alpha and save them in the dictionary
    param_overal_scores.append(sum(score_list)/len(score_list))
    param_overal_error.append(sum(error_list)/len(error_list))
    sfs_overal_scores.append(sum(sfs_score_list)/len(sfs_score_list))
    Mape_overal_error.append(sum(Mape_error_list)/len(Mape_error_list))

    cv_info[this_a]['mean_model_score'] = sum(score_list)/len(score_list)
    cv_info[this_a]['mean_sfs_score'] = sum(sfs_score_list)/len(sfs_score_list)
    cv_info[this_a]['mean_error'] = sum(error_list)/len(error_list)
    cv_info[this_a]['mean_MAPE_error'] = sum(Mape_error_list)/len(Mape_error_list)


"""################################################## Results #####################################################"""

# select the best alpha based on obtained values
RSE_index = param_overal_error.index(min(param_overal_error))
alpha_score_index = param_overal_scores.index(max(param_overal_scores))
MAPE_index = Mape_overal_error.index(min(Mape_overal_error))

# report the best alpha based on obtained values
print('RSE_index = ', RSE_index)
print('Best_alpha_score_index = ', alpha_score_index)
print('MAPE_index = ', MAPE_index)


# prepare the data for final error on test dataset
X_train = pd.DataFrame.as_matrix(ext_train_features_df)
Y_train = pd.DataFrame.as_matrix(train_labels)
X_test = pd.DataFrame.as_matrix(ext_test_features_df)
Y_test = pd.DataFrame.as_matrix(test_labels)

print('..........................')
# RSE error computations for the model consisting of the selected features:
my_best_alpha = 'alpha = '+str(alpha_v[RSE_index])
my_best_sel = []
for i in range (fold_num):
    fold_sel = cv_info[my_best_alpha][str(i)]['Selected_features_idx']
    my_best_sel = list(set.union(set(my_best_sel), set(fold_sel)))
print('Best sel based on minimum RSE error: ', my_best_sel)


best_model_myError = Ridge(alpha_v[RSE_index])
best_model_myError.fit(X_train[:, my_best_sel], Y_train)
Y_hat_test = best_model_myError.predict(X_test[:, my_best_sel])
print('RSE best alpha = ', alpha_v[RSE_index])
Error = math.sqrt(sum((Y_hat_test-Y_test)**2))
realError = Error*output_scaler_std
print('RSE Error = ', realError)
print('MAPE Error = ', calcMAPE(Y_hat_test, Y_test))
print('..........................')


# alpha score computations for the model consisting of the selected features:
python_best_alpha = 'alpha = '+str(alpha_v[alpha_score_index])
python_best_sel = []
for i in range (fold_num):
    fold_sel = cv_info[python_best_alpha][str(i)]['Selected_features_idx']
    python_best_sel = list(set.union(set(python_best_sel), set(fold_sel)))
print('Best sel based on classifier maximum score: ', python_best_sel)



best_model_pythonError = Ridge(alpha_v[alpha_score_index])
best_model_pythonError.fit(X_train[:, python_best_sel], Y_train)
Y_hat_test = best_model_pythonError.predict(X_test[:, python_best_sel])
print('Classifier maximum score alpha = ', alpha_v[alpha_score_index])
Error = math.sqrt(sum((Y_hat_test-Y_test)**2))
realError = Error*output_scaler_std
print('RSE Error = ', realError)
print('MAPE Error = ', calcMAPE(Y_hat_test, Y_test))
print('..........................')

# MAPE computations:
mape_best_alpha = 'alpha = '+str(alpha_v[MAPE_index])
mape_best_sel = []
for i in range (fold_num):
    fold_sel = cv_info[mape_best_alpha][str(i)]['Selected_features_idx']
    mape_best_sel = list(set.union(set(mape_best_sel), set(fold_sel)))
print('Best sel based on minimum MAPE error: ', mape_best_sel)


best_model_MAPE = Ridge(alpha_v[MAPE_index])
best_model_MAPE.fit(X_train[:, mape_best_sel], Y_train)
Y_hat_test = best_model_MAPE.predict(X_test[:, mape_best_sel])
print('MAPE best alpha = ', alpha_v[MAPE_index])
Error = math.sqrt(sum((Y_hat_test-Y_test)**2))
realError = Error*output_scaler_std
print('RSE Error = ', realError)
print('MAPE Error = ', calcMAPE(Y_hat_test, Y_test))
print('..........................')


# report the grid search information
print(cv_info)


