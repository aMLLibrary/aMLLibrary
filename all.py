import pandas as pd
import numpy as np
import random
import math
import itertools
import ast
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from custom_sfs import CSequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# variables:
input_path = 'yourfile.csv'
#input_path = 'yourfile.csv'

core_nums_train_data = [6, 10, 14, 18, 24, 28, 32, 36, 40, 44]
core_nums_test_data = [8, 12, 16, 22, 26, 30, 34, 38, 42]

select_features_vif = False
select_features_sfs = True
min_features = 1
max_features = -1
is_floating = False
fold_num = 5
regressor_name = "lr"


ridge_params = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
lasso = '[0.1,0.5]'

max_features_dt = '[sqrt]'
max_depth_dt = '[7]'
min_samples_leaf_dt = '[4]'
min_samples_split_dt = '[5]'

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
degree = '[3]'
gamma = '[0.01]'
n_neighbors = '[5,10]'

test_without_apriori = False

#scaler = 0
scaler = StandardScaler()



def add_all_comb(features_names, data_matrix, degree):

    indices = list(range(data_matrix.shape[1]))
    for j in range(2, degree + 1):
        combs = list(itertools.combinations_with_replacement(indices, j))

        for cc in combs:
            temp = 1
            for i in range(len(cc)):
                temp = np.multiply(temp,data_matrix[:, cc[i]])
            new_feature_value = temp.reshape((data_matrix.shape[0], 1))
            #new_feature_value=np.multiply(data_matrix[:, cc[0]],data_matrix[:, cc[1]])
            data_matrix = np.append(data_matrix , new_feature_value, axis=1)
            new_feature_name = ''
            for i in range(len(cc)-1):
                new_feature_name = new_feature_name+features_names[cc[i]]+'_'
            new_feature_name = new_feature_name+features_names[cc[i+1]]
            features_names.append(new_feature_name)
            # new_features_dict[new_feature_name] = list(new_feature_value.reshape(1,data_matrix.shape[0]))

    return features_names, data_matrix
    # return new_features_dict



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




def grid_search(pipeline,params_grid):
    gs = GridSearchCV(estimator=pipeline,
                        param_grid=params_grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=1,
                        cv=fold_num,
                        refit=True)
    return gs

def sfs_and_grid_search():
    if test_without_apriori == False:

        if "ridge" in regressor_name or "lasso" in regressor_name:
            regressor, param_grid = get_lr()
        if "svr" in regressor_name:
            regressor, param_grid = get_svr()
        if "knn" in regressor_name:
            regressor, param_grid = get_knn()
        if "dt" in regressor_name:
            regressor, param_grid = get_dt()
        # if "rf" in self.regressor_name:
            # regressor, param_grid = self.get_rf()

        # self.logger.info(self.regressor_name)
        # self.logger.info(param_grid)
        sfs = sfs_model(regressor)
        pipe = Pipeline([('sfs', sfs)])
        gs = grid_search(pipe, param_grid)
        gs = gs.fit(ext_train_features, train_labels)

        # SFS checking with sample data : set n_features to k_features in sfs_model
        # X, y, coef = make_regression(n_samples=20, n_features=6, noise=0.1, coef=True)
        # gs = gs.fit(X, y)
        # sfs = gs.best_estimator_.steps[0][1]
        # print(coef)
        # print(sfs.estimator.coef_)
        # fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
        # plt.title('Sequential Forward Selection (w. StdDev)')
        # plt.grid()
        # plt.show()

        # Since the data for classsifierselection is too small, we only calculate the train error
        # if self.data_conf["input_name"] == "classifierselection":
         #   y_pred_test = []
        #else:
        #    y_pred_test = gs.predict(self.ext_test_features)

        y_pred_test = gs.predict(ext_test_features)

        y_pred_train = gs.predict(ext_train_features)

        err_test, err_train = mean_absolute_percentage_error(y_pred_test, y_pred_train)

        result_path = self.save_results(gs, param_grid, err_train, err_test)
        filename = os.path.join(result_path, 'best_model.sav')
        pickle.dump(gs, open(filename, 'wb'))

        if self.data_conf["test_without_apriori"] == True:
            self.get_result_name()
            result_path = os.path.join(".", )
            result_path = os.path.join(".", "results", "sparkdl",
                                       self.data_conf["input_name"],
                                       self.data_conf["sparkdl_run"])
            # DELETE THIS LATER
            result_path = os.path.join(result_path, self.result_name)
            result_path = os.path.join(result_path, (self.data_conf["split"] + "_" +
                                                     self.data_conf["case"] + "_" +
                                                     self.regressor_name))
            filename = os.path.join(result_path, 'best_model.sav').replace("_test_without_apriori", "")
            gs_loaded = pickle.load(open(filename, 'rb'))
            y_pred_test = gs_loaded.predict(self.ext_test_features)
            y_pred_train = gs_loaded.predict(self.ext_train_features)
            err_test, err_train = self.mean_absolute_percentage_error(y_pred_test, y_pred_train)
            result_path = self.save_results(gs_loaded, "-", err_train, err_test)

    return result_path


def get_lr():

    lasso_alpha = lasso
    lasso_alpha = [float(i) for i in ast.literal_eval(lasso_alpha)]

    ridge_alpha = ridge
    ridge_alpha = [float(i) for i in ast.literal_eval(ridge_alpha)]

    param_grid_lr = []
    if regressor_name == "lasso":
        lr = Lasso()
        param_grid_lr.append({'sfs__estimator__alpha': lasso_alpha})
    elif regressor_name == "ridge":
        lr = Ridge()
        param_grid_lr.append({'sfs__estimator__alpha': ridge_alpha })

    return lr, param_grid_lr

def get_svr():

    kernel = str(kernel)
    kernel = kernel[1:-1].split(",")

    C = c
    C = [float(i) for i in ast.literal_eval(C)]

    epsilon = epsilon
    epsilon = [float(i) for i in ast.literal_eval(epsilon)]

    degree = degree
    degree = [int(i) for i in ast.literal_eval(degree)]

    gamma = gamma
    gamma = [float(i) for i in ast.literal_eval(gamma)]
    # gamma = str(self.conf.get('FeatureSelector','gamma'))
    # gamma = gamma[1:-1].split(",")

    svr = SVR()
    param_grid_svr = []

    for kk in kernel:
        params = {}
        params['sfs__estimator__C'] = C
        params['sfs__estimator__epsilon'] = epsilon
        params['sfs__estimator__kernel'] = [kk]
        if kk == "poly" and degree != []:
            params['sfs__estimator__degree'] = degree
        if kk == "rbf" and gamma != []:
            params['sfs__estimator__gamma'] = gamma
        param_grid_svr.append(params)

    return svr, param_grid_svr

def get_knn():

    n_neighbors = n_neighbors
    n_neighbors = [int(i) for i in ast.literal_eval(n_neighbors)]

    knn = KNeighborsRegressor()
    param_grid_knn = [{'sfs__estimator__n_neighbors': n_neighbors}]

    return knn, param_grid_knn


def get_dt():

    max_depth = max_depth_dt
    max_depth = [int(i) for i in ast.literal_eval(max_depth)]

    min_samples_leaf = min_samples_leaf_dt
    min_samples_leaf = [int(i) for i in ast.literal_eval(min_samples_leaf)]

    min_samples_split = min_samples_split_dt
    min_samples_split = [int(i) for i in ast.literal_eval(min_samples_split)]

    max_features = max_features_dt
    max_features = max_features[1:-1].split(",")

    dt = DecisionTreeRegressor()
    param_grid_dt = [{'sfs__estimator__max_depth': max_depth,
                        'sfs__estimator__min_samples_leaf': min_samples_leaf,
                        'sfs__estimator__min_samples_split': min_samples_split,
                        'sfs__estimator__max_features': max_features}]

    return dt, param_grid_dt

def get_rf():

    n_estimators = n_estimators
    n_estimators = [int(i) for i in ast.literal_eval(n_estimators)]

    max_features = max_features_rf
    max_features = max_features[1:-1].split(",")

    max_depth = max_depth_rf
    max_depth = [int(i) for i in ast.literal_eval(max_depth)]

    min_samples_leaf = min_samples_leaf_rf
    min_samples_leaf = [int(i) for i in ast.literal_eval(min_samples_leaf)]

    min_samples_split = min_samples_split_rf
    min_samples_split = [int(i) for i in ast.literal_eval(min_samples_split)]

    bootstrap = bootstrap
    bootstrap = [bool(i) for i in ast.literal_eval(bootstrap)]

    rf = RandomForestRegressor()
    param_grid_rf = [{'sfs__estimator__n_estimators': n_estimators,
                          'sfs__estimator__max_features': max_features,
                          'sfs__estimator__max_depth': max_depth,
                          'sfs__estimator__min_samples_leaf': min_samples_leaf,
                          'sfs__estimator__min_samples_split': min_samples_split,
                          'sfs__estimator__bootstrap': bootstrap}]

    return rf, param_grid_rf



def sfs_model(model):
    sfs = CSequentialFeatureSelector(clone_estimator=True,
                                    estimator=model,
                                    k_features=k_features,
                                    forward=True,
                                    floating=is_floating,
                                    scoring='neg_mean_squared_error',
                                    cv=fold_num)
    return sfs

def scale_data(df):
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)
    return scaled_df, scaler


df = pd.read_csv(input_path)

# drop run column
tempcols = ["run"]
df = df.drop(tempcols, axis=1)
if input_path == 'yourfile.csv':
    df = df.drop("Unnamed: 6", axis=1)

data_matrix = pd.DataFrame.as_matrix(df)
features_names = list(df.columns.values)
column_names = list(df)
df_dict = dict(df)
csv_col_name = list(df.columns.values)

output = df_dict['applicationCompletionTime']

# removes the columns of run and application completion time
cols = ["applicationCompletionTime"]
original_input = df.drop(cols, axis=1)
original_input_mat = pd.DataFrame.as_matrix(original_input)
original_col_name = list(original_input.columns.values)
original_features_names = features_names[2:]

# Separate the training and test sets based on the datasize
# case1:
# ooni ke datasizashoon ye joore ro bayad negah dare va kollan baghiye sample ha ro forget kone:

# remove zero valued columns
df = df.loc[:, (df != df.iloc[0]).any()]
seed = 1234
df = shuffle(df, random_state = seed)


data_conf = {}
data_conf["case"] = "same datasize in TR and TE_even cores in TR, odds in TE"
data_conf["input_name"] = "K_means"

image_nums_train_data = [5]
image_nums_test_data = [5]
core_nums_train_data = [6,10,14,18,22,26,30,34,38,42,46]
core_nums_test_data = [8,12,16,20,24,28,32,36,40,44,48]


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

################ core indices:

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



######## sacle the data
df, scaler = scale_data(df)


train_df = df.ix[train_indices]
test_df = df.ix[test_indices]
train_labels = train_df.iloc[:, 0]
train_features = train_df.iloc[:, 1:]
test_labels = test_df.iloc[:, 0]
test_features = test_df.iloc[:, 1:]

features_names = list(df.columns.values)[1:]

data_conf["reduced_features_names"] = list(train_df.columns.values)[1:]
data_conf["train_features_org"] = train_features.as_matrix()
data_conf["test_features_org"] = test_features.as_matrix()




# use the inverse of n_core instead of nContainers
# for i in range(input_mat.shape[0]):
#    input_mat[i][57] = 1 / input_mat[i][57]
# final_feature_name[57] = 'inverse_nContainers'



# Feature extension:




# find the best model
# k_fold cross validation: (using just the training set)

fold_num = fold_num
fold_num = train_features.shape[0]

feature_num = train_features.shape[1]
sample_num = train_features.shape[0]
alpha_v = ridge_params



k_fold= KFold(n_splits=fold_num, shuffle=False, random_state=None)

X = pd.DataFrame.as_matrix(train_features)
Y = pd.DataFrame.as_matrix(train_labels)
ridge_overal_scores = []
error_list = []
for a in alpha_v:
    score_list = []
    ridge = Ridge(a)
    for k, (train, test) in enumerate(k_fold.split(X, Y)):
        ridge.fit(X[train, :], Y[train])
        score_list.append(ridge.score(X[test, :], Y[test]))
        Y_hat = ridge.predict(X[test, :])
        sserror = math.sqrt(sum((Y_hat-Y[test])**2))
    ridge_overal_scores.append(sum(score_list)/len(score_list))
    error_list.append(sserror)
    min_index = error_list.index(min(error_list))
    max_index = ridge_overal_scores.index(max(ridge_overal_scores))

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
print('My alpha = ', alpha_v[min_index])
Error = math.sqrt(sum((Y_hat_test-Y_test)**2))
print('My Error = ', Error)


# Python model error
best_model_pythonError.fit(X, Y)
Y_hat_test = best_model_pythonError.predict(X_test)
print('Python alpha = ', alpha_v[max_index])
Error = math.sqrt(sum((Y_hat_test-Y_test)**2))
print('Python Error = ', Error)


