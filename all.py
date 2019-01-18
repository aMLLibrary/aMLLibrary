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
from sklearn import feature_selection as FS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS



# variables:
input_path = 'P8_kmeans.csv'
#input_path = 'dumydata.csv'
#input_path = 'yourfile.csv'


image_nums_train_data = [5]
image_nums_test_data = [5]
core_nums_train_data = [6,10,14,18,22,26,30,34,38,42,46]
core_nums_test_data = [8,12,16,20,24,28,32,36,40,44,48]


select_features_vif = False
select_features_sfs = True
min_features = 1
max_features = -1
is_floating = False
fold_num = 5
FSfold_num = 2
LOOCV = True
regressor_name = "lr"
degree = 3

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
input_normalization = True
output_normalization = False
extension = True



def calculate_new_col(X, indices):
    index = 0
    new_col = X[:, indices[index]]
    for ii in list(range(1, len(indices))):
        new_col = np.multiply(new_col, X[:, indices[index + 1]])
        index += 1
    return new_col


def add_all_comb(df, degree):

    df_dict = dict(df)
    data_matrix = pd.DataFrame.as_matrix(df)
    indices = list(range(data_matrix.shape[1]))
    for j in range(2, degree + 1):
        combs = list(itertools.combinations_with_replacement(indices, j))
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
    df_dict = dict(df)
    for c in colNameList:
        new_col = 1/np.array(df[c])
        new_feature_name = 'inverse_'+c
        df_dict[new_feature_name] = new_col

    inv_df = pd.DataFrame.from_dict(df_dict)
    return inv_df

def myNorm(df, col_name_list):
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


def scale_data(df):
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)
    return scaled_df, scaler


df = pd.read_csv(input_path)
# drop run column
tempcols = ["run"]
df = df.drop(tempcols, axis=1)
#if input_path != 'P8_kmeans.csv':
#    df = df.drop("Unnamed: 6", axis=1)

data_matrix = pd.DataFrame.as_matrix(df)
features_names = list(df.columns.values)


output_df = df['applicationCompletionTime']

# removes the columns of run and application completion time
cols = ["applicationCompletionTime"]
df = df.drop(cols, axis=1)



# Separate the training and test sets based on the datasize
# case1:
# ooni ke datasizashoon ye joore ro bayad negah dare va kollan baghiye sample ha ro forget kone:

# remove zero valued columns
df = df.loc[:, (df != df.iloc[0]).any()]


# TODO: remove const value


# randomize the samples
seed = 1234
df = shuffle(df, random_state = seed)


data_conf = {}
data_conf["case"] = "same datasize in TR and TE_even cores in TR, odds in TE"
data_conf["input_name"] = "K_means"




################ Datasize indices:
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

############################################################### split the data #######################################

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


#train_df = df.ix[train_indices]
#test_df = df.ix[test_indices]
#train_labels = train_df.iloc[:, 0]
#train_features = train_df.iloc[:, 1:]
#test_labels = test_df.iloc[:, 0]
#test_features = test_df.iloc[:, 1:]

train_df = df.ix[train_indices]
test_df = df.ix[test_indices]
train_labels = output_df.ix[train_indices]
train_features = train_df
test_labels = output_df.ix[test_indices]
test_features = test_df



data_conf["reduced_features_names"] = list(train_df.columns.values)[1:]
data_conf["train_features_org"] = train_features.as_matrix()
data_conf["test_features_org"] = test_features.as_matrix()





############################################################### Inversing #######################################



# use the inverse of n_core instead of nContainers
if inversing == True:
    df = add_inverse_features(df, inverseColNameList)

#FIXME: remove the dropping
# if we wanna remove the features that we got inverse from, just drop them:

    df = df.drop(inverseColNameList, axis=1)

############################################################### Normalization #######################################

# FIXME: Standard scaling
######## scale the data
# df, scaler = scale_data(df)

###### My normalization:
if input_normalization == True:
    df, scaling_factor = myNorm(df, list(df.columns.values))

### scaling the output:
if output_normalization == True:

    data_matrix_out = pd.DataFrame.as_matrix(output_df)
    dfmin_out = data_matrix_out.min()
    dfmax_out = data_matrix_out.max()
    scale_factor_out = dfmax_out - dfmin_out
    output_df = np.array(output_df - dfmin_out)
    output_df = pd.DataFrame(output_df/scale_factor_out)


############################################################### Extension #######################################

if extension == True:
    df = add_all_comb(df, degree)


################################################################# SFS ############################################

if select_features_sfs == True:

    min_k_features = int(min_features)
    max_k_features = int(max_features)
    # Selecting from all features
    if max_k_features == -1:
        k_features = (min_k_features, df.shape[1])
        # Selecting from the given range
    if max_k_features != -1:
        k_features = (min_k_features, max_k_features)


X = pd.DataFrame.as_matrix(df.ix[train_indices])
Y = pd.DataFrame.as_matrix(output_df[train_indices])

model = LinearRegression()

sfs = SFS(model,
          k_features=k_features,
          forward=True,
          floating=False,
          scoring='neg_mean_squared_error',
          cv=FSfold_num,
          n_jobs=-1)
sfs = sfs.fit(X, Y)
# scoring='accuracy'

print('\nSequential Forward Selection (k=3):')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)



# find the best model
# k_fold cross validation: (using just the training set)
fold_num = fold_num
if LOOCV == True:
    fold_num = train_features.shape[0]

feature_num = train_features.shape[1]
sample_num = train_features.shape[0]
alpha_v = ridge_params


k_fold= KFold(n_splits=fold_num, shuffle=False, random_state=None)

X = pd.DataFrame.as_matrix(train_features)
Y = pd.DataFrame.as_matrix(train_labels)
param_overal_scores = []
param_overal_error = []
for a in alpha_v:
    score_list = []
    error_list = []
    ridge = Ridge(a)
    for k, (train, test) in enumerate(k_fold.split(X, Y)):
        ridge.fit(X[train, :], Y[train])
        score_list.append(ridge.score(X[test, :], Y[test]))
        Y_hat = ridge.predict(X[test, :])
        sserror = math.sqrt(sum((Y_hat-Y[test])**2))
        error_list.append(sserror)
    param_overal_scores.append(sum(score_list)/len(score_list))
    param_overal_error.append(sum(error_list)/len(error_list))
    min_index = param_overal_error.index(min(param_overal_error))
    max_index = param_overal_scores.index(max(param_overal_scores))

print('min_index = ', min_index)
print('max_index = ', max_index)

best_model_myError = Ridge(alpha_v[min_index])
best_model_pythonError = Ridge(alpha_v[max_index])


X_test = pd.DataFrame.as_matrix(test_features)
Y_test = pd.DataFrame.as_matrix(test_labels)

# Error in test data:
# My model error
best_model_myError.fit(X, Y)
Y_hat_test = best_model_myError.predict(X_test)
print('Y_hat_test = ')
print(Y_hat_test)
print('Y_hat_test = ')
print(Y_hat_test.shape)
print(Y_test)
print('Y_test_size = ')
print(Y_test.shape)


print('My alpha = ', alpha_v[min_index])
Error = math.sqrt(sum((Y_hat_test-Y_test)**2))

if output_normalization == True:
    realError = Error*scale_factor_out
else:
    realError = Error

print('My Error = ', realError)


# Python model error
best_model_pythonError.fit(X, Y)
Y_hat_test = best_model_pythonError.predict(X_test)
print('Python alpha = ', alpha_v[max_index])
Error = math.sqrt(sum((Y_hat_test-Y_test)**2))

if output_normalization == True:
    realError = Error*scale_factor_out
else:
    realError = Error

print('Python Error = ', realError)



