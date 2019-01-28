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
image_nums_train_data = [5]
image_nums_test_data = [5]
case = 'case1'
core_nums_train_data = [6, 10, 14, 18, 24, 28, 32, 36, 40, 44]
core_nums_test_data = [8, 12, 16, 22, 26, 30, 34, 38, 42]
use_spark_info = False


#[DebugLevel]
debug = True # This is for printing the logs. If debug is true, we also print the logs in the DEBUG level. Otherwise, only the logs in INFO level is printed.

#[FeatureExtender]
n_terms = [2]

#[FeatureSelector]
select_features_vif = False
select_features_sfs = True
min_features = 1
max_features = -1
is_floating = False
fold_num = 10
regressor_name = 'dt'
# regressor_name = ridge
ridge = [0.1,0.5]
lasso = [0.1,0.5]

max_features_dt = '[sqrt]'
max_depth_dt = [7]
min_samples_leaf_dt = [4]
min_samples_split_dt = [5]

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








input_path = 'P8_kmeans.csv'
#input_path = 'dumydata2.csv'
#input_path = 'yourfile.csv'
#input_path = 'dumydata.csv'
#input_path = "newdummy.csv"
#input_path = "newd.csv"
#input_path = "testcase4.csv"


result_path = "./results/"

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






test_without_apriori = False

# preprocessing related variables (DT)
inversing = True
inverseColNameList = ['nContainers']
extension = False
degree = 2



def calculate_new_col(X, indices):
    """Given two indices and input matrix returns the multiplication of the columns corresponding columns"""
    index = 0
    new_col = X[:, indices[index]]
    for ii in list(range(1, len(indices))):
        new_col = np.multiply(new_col, X[:, indices[index + 1]])
        index += 1
    return new_col


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

def calcMAPE(Y_hat, Y):
    """given true and predicted values returns MAPE error"""
    Mapeerr = np.mean(np.abs((Y - Y_hat) / Y)) * 100
    return Mapeerr

def scale_data(df):
    """scale and normalized the data using standard scaler and returns the scaled data, the scaler object, the mean
    value and std of the output column for later use"""

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)

    # obtain the mean and std of the standard scaler using the formula z = (x-u)/s
    # if u is the mean and s is the std, z is the scaled version of the x

    #output_scaler_mean = scaler.mean_
    #output_scaler_std = (df.values[0][0] - output_scaler_mean[0])/ scaled_array[0][0]

    # return the mean and std for later use
    # return scaled_df, scaler, output_scaler_mean[0], output_scaler_std
    # return scaled_df, scaler, df.mean(), df.std()
    return scaled_df, scaler

def denormalizeCol(test_labels, y_mean , y_std ):
    denorm_test_labels = (test_labels*y_std)+y_mean
    return denorm_test_labels

def calcMSE(Y_hat, Y):
    MSE = np.mean((Y_hat - Y) ** 2)
    return MSE



def plot_histogram(results, result_path):

    plot_path = os.path.join(result_path, "Features_Ferquency_Histogram_plot")

    """Plot the histogram of features selection frequency"""
    names_list = results[0]['ext_feature_names']
    name_count = []
    for i in range(len(names_list)):
        name_count.append(0)
    iternum = len(results)

    for i in range(iternum):
        for j in results[i]['Results']['Sel_features']:
            name_count[j] += 1

    font = {'family':'normal','size': 10}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig = plt.figure(figsize=(10,5))
    plt.bar(range(len(names_list)), name_count)
    plt.xticks(range(len(names_list)), names_list)
    plt.xticks(rotation = 90)
    plt.title(results[0]['case']+' - Histogram of features selection frequency')
    plt.show()
    plt.tight_layout()
    fig.savefig(plot_path + ".pdf")





def plot_MSE_Errors(results, result_path):

    plot_path = os.path.join(result_path, "MSE_Error_plot")
    MSE_list_TR = []
    MSE_list_TE = []
    for i in range(len(results)):
        MSE_list_TR.append(results[i]['Results']['MSE_Error_TR'])
        MSE_list_TE.append(results[i]['Results']['MSE_Error_TE'])

    font = {'family':'normal','size': 15}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig1 = plt.figure(figsize=(9,6))
    plt.plot(range(1, len(results)+1), MSE_list_TR, 'bs', range(1, len(results)+1), MSE_list_TE, 'r^')
    plt.xlabel('runs')
    plt.ylabel('MSE Error')
    plt.title('MSE Error in Training and Test Sets')
    plt.xlim(1, len(MSE_list_TE))
    plt.show()
    fig1.savefig(plot_path + ".pdf")


def plot_MAPE_Errors(results, result_path):

    plot_path = os.path.join(result_path, "MAPE_Error_plot")
    MAPE_list_TR = []
    MAPE_list_TE = []
    for i in range(len(results)):
        MAPE_list_TR.append(results[i]['Results']['MAPE_Error_Training'])
        MAPE_list_TE.append(results[i]['Results']['MAPE_Error_Test'])
    font = {'family':'normal','size': 15}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig2 = plt.figure(figsize=(9,6))
    plt.plot(range(1, len(results) + 1), MAPE_list_TR, 'bs', range(1, len(results) + 1), MAPE_list_TE, 'r^')
    plt.xlabel('runs')
    plt.ylabel('MAPE Error')
    plt.title('MAPE Error in Training and Test Sets')
    plt.xlim(1, len(MAPE_list_TE))
    plt.legend()
    plt.show()
    fig2.savefig(plot_path + ".pdf")


def plot_Model_Size(results, result_path):

    plot_path = os.path.join(result_path, "Model_Size_Plot")
    model_size_list = []
    for i in range(len(results)):
        model_size_list.append(results[i]['Results']['Model_size'])
    font = {'family':'normal','size': 15}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig3 = plt.figure(figsize=(9,6))
    plt.bar(range(1, len(results) + 1), model_size_list)
    plt.xlabel('runs')
    plt.ylabel('Model Size')
    plt.title('Number of Selected Features in Runs')
    plt.xlim(1, len(model_size_list))
    plt.ylim(1, len(results[0]['ext_feature_names']))
    plt.show()
    fig3.savefig(plot_path + ".pdf")




def plot_predicted_true(results, result_path):

    plot_path = os.path.join(result_path,"true_pred_plot")

    # find the best run
    MSE_list_TE = []
    for i in range(len(results)):
        MSE_list_TE.append(results[i]['Results']['MSE_Error_TE'])
    best_run = MSE_list_TE.index(min(MSE_list_TE))

    data_conf = results[best_run]
    y_true_train = results[best_run]['Results']['Y_training_real']
    y_pred_train = results[best_run]['Results']['Y_hat_training_real']
    y_true_test = results[best_run]['Results']['Y_test_real']
    y_pred_test = results[best_run]['Results']['Y_hat_test_real']

    font = {'family':'normal','size': 15}
    matplotlib.rc('font', **font)
    colors = cm.rainbow(np.linspace(0, 0.5, 3))
    fig4 = plt.figure(figsize=(9,6))
    plt.scatter(y_pred_train, y_true_train, marker='o',s=300,facecolors='none',label="Train Set",color=colors[0])
    plt.scatter(y_pred_test, y_true_test, marker='^',s=300,facecolors='none',label="Test Set",color=colors[1])
    #if y_pred_test.values != []:
    min_val = min(min(y_pred_train),min(y_true_train),min(y_pred_test),min(y_true_test))
    max_val = max(max(y_pred_train),max(y_true_train),max(y_pred_test),max(y_true_test))
    #if y_pred_test.values == []:
    min_val = min(min(y_pred_train),min(y_true_train))
    max_val = max(max(y_pred_train),max(y_true_train))
    lines = plt.plot([min_val, max_val], [min_val, max_val], '-')
    plt.setp(lines, linewidth=0.9, color=colors[2])
    plt.title("Predicted vs True Values for " +  regressor_name +"\n" + \
                data_conf["input_name"]  + " " + str(data_conf["case"])  + " " + \
                str(data_conf["image_nums_train_data"]) + \
                str(data_conf["image_nums_test_data"]) )
    plt.xlabel("Predicted values of applicationCompletionTime (ms)")
    plt.ylabel("True values of " + "\n" + "applicationCompletionTime (ms)")
    fig4.text(.5, .01, 'alpha = '+ str(results[best_run]['Results']['alpha']), ha='center')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(prop={'size': 20})
    plt.show()
    fig4.savefig(plot_path + ".pdf")


def Ridge_SFS_GridSearch(alpha_v, X,Y,k_features,fold_num):
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
        print('alpha = ', a, '     MSE Error= ', MSE, '    MAPE Error = ', Mape_error, '    Ridge Coefs= ', ridge.coef_ ,'    Intercept = ', ridge.intercept_, '    SEl = ', ext_feature_names[list(sel_F_idx)]  )


    """################################################## Results #####################################################"""

    # select the best alpha based on obtained values
    MSE_index = param_overal_MSE.index(min(param_overal_MSE))
    #MAPE_index = Mape_overal_error.index(min(Mape_overal_error))


    # report the best alpha based on obtained values
    print('Least_MSE_Error_index = ', MSE_index, ' => Least_RSE_Error_alpha = ', alpha_v[MSE_index])
    #print('Least_MAPE_Error_index = ', MAPE_index, ' => Least_MAPE_Error_alpha = ', alpha_v[MAPE_index])
    Least_MSE_alpha = alpha_v[MSE_index]
    #MeastMAPEalpha = alpha_v[MAPE_index]
    best_trained_model = Ridge(Least_MSE_alpha)
    best_trained_model.fit(X[:, sel_F[MSE_index]], Y)


    return cv_info, Least_MSE_alpha, best_trained_model




def locateTRTEindices(df, image_nums_train_data, image_nums_test_data, core_nums_train_data, core_nums_test_data):
    """ locating the indices of training and test samples based on datasize and core numbers:"""
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



    ################## get together all the TR and TE indices:
    train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
    test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)

    return train_indices, test_indices


def split_data(ext_df):
    train_df = ext_df.ix[train_indices]
    test_df = ext_df.ix[test_indices]

    train_labels = train_df.iloc[:, 0]
    train_features = train_df.iloc[:, 1:]
    test_labels = test_df.iloc[:, 0]
    test_features = test_df.iloc[:, 1:]
    return train_features, train_labels, test_features, test_labels

def calc_k_features(min_features, max_features):

    """calculate the range of number of features that sfs is allowed to select"""
    min_k_features = int(min_features)
    max_k_features = int(max_features)
    # Selecting from all features
    if max_k_features == -1:
        k_features = (min_k_features, train_features.shape[1])
        # Selecting from the given range
    if max_k_features != -1:
        k_features = (min_k_features, max_k_features)
    return k_features




# Dictionary keeping the information about the input and training and test samples
data_conf = {}
data_conf["case"] = "same datasize in TR and TE_even cores in TR, odds in TE"
data_conf["input_name"] = "K_means"

# Separate the training and test sets based on the datasize
# case1:
# Training and test samples have the same datasizes:
"""################################################ Datasize indices #######################################"""

# save the info about training and test datasize in the data dictionary
data_conf["image_nums_train_data"] = image_nums_train_data
data_conf["image_nums_test_data"] = image_nums_test_data


# save the info about training and test core numbers in the data dictionary
data_conf["core_nums_train_data"] = core_nums_train_data
data_conf["core_nums_test_data"] = core_nums_test_data


# read the data from .csv file:
df = pd.read_csv(input_path)
sample_num = df.shape[0]



# drop run column
tempcols = ["run"]
df = df.drop(tempcols, axis=1)

# compute the matrix and name of columns as features
data_matrix = pd.DataFrame.as_matrix(df)


# remove constant valued columns
df = df.loc[:, (df != df.iloc[0]).any()]


results = []
for iter in range(run_num):

    # randomize the samples
    seed = 1234
    df = shuffle(df, random_state = seed)

    # locating the indices of training and test samples based on datasize and core numbers:
    train_indices, test_indices = locateTRTEindices(df, image_nums_train_data, image_nums_test_data,
                                                    core_nums_train_data, core_nums_test_data)

    data_conf["train_indices"] = train_indices
    data_conf["test_indices"] = test_indices

    """################################################ Inversing #################################################"""

    # adding the inverse of nContainers columns to the data
    if inversing == True:
        inv_df, inversed_cols_tr = add_inverse_features(df, inverseColNameList)


    # check the feature names after inversing
    inv_feature_names = inv_df.columns.values



    """################################################ Extension #################################################"""


    # save the output
    #output_df = inv_df['applicationCompletionTime']
    #inv_df = inv_df.drop(['applicationCompletionTime'], axis=1)



    # Extend the df by adding combinations of features as new features to the df
    if extension == True:

        ext_df = add_all_comb(inv_df, inversed_cols_tr, 0, degree)
        ext_feature_names = ext_df.columns.values[1:]

    if extension == False:
        ext_df = inv_df
        ext_feature_names = ext_df.columns.values[1:]

    # check the feature names after extension

    data_conf["ext_feature_names"] = ext_feature_names.tolist()


    """################################################ Splitting  ##############################################"""


    train_features, train_labels, test_features, test_labels = split_data(ext_df)

    """############################################## Normalization ###############################################"""

    # scale the data
    scaled_train_labels, train_labels_scaler = scale_data(pd.DataFrame(train_labels))
    scaled_train_features, train_features_scaler = scale_data(train_features)
    scaled_test_labels, test_labels_scaler = scale_data(pd.DataFrame(test_labels))
    scaled_test_features, test_features_scaler = scale_data(test_features)

    """##########################  SFS and Hyper Params tunning using grid search and CV ##########################"""

    # computing the range of accepted number of features: (k_features)
    if select_features_sfs == True:
        k_features = calc_k_features(min_features, max_features)


    # input and output preparation to use in CV iterations

    X = pd.DataFrame.as_matrix(scaled_train_features)
    Y = pd.DataFrame.as_matrix(scaled_train_labels)





    # finding the best model using Kfold-CV and just the training set
    cv_info, Least_MSE_alpha, best_trained_model = Ridge_SFS_GridSearch(ridge_params, X, Y, k_features, fold_num)


    # determining the size of data
    # feature_num = X.shape[1]


    # Compute the error for training and test data
    # Prepare the data for final error on test dataset based on different (scaled version)


    X_train = pd.DataFrame.as_matrix(scaled_train_features)
    Y_train = pd.DataFrame.as_matrix(scaled_train_labels)
    X_test = pd.DataFrame.as_matrix(scaled_test_features)
    Y_test = pd.DataFrame.as_matrix(scaled_test_labels)



    '''############################################ MSE ####################################################'''
    print('.................................... Results based on MSE ...........................................')
    # RSE error computations for the model consisting of the selected features:
    print('Best alpha by selecting model having minimum MSE error in the grid search = ', Least_MSE_alpha)
    print('Selected features based on minimum MSE error: ', cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features'])
    print('Selected features names based on minimum MSE error: ', cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features_Names'])
    print('Model size based on minimum MSE error: ', len(cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features']))

    final_sel_idx = cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features']
    final_sel_names = cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features_Names']



    # predict training and test data using the model suggested by least RSE error

    # least_MSE_model = Ridge(alpha_v[MSE_index])
    # least_MSE_model.fit(X_train[:, sel_F[MSE_index]], Y_train)



    # These Y_TR set is predicted using the final model returned by GridSearch:
    Y_hat_training_scaled = best_trained_model.predict(X_train[:, final_sel_idx])

    # scale back the data to compute the MSE and MAPE
    Y_hat_training_real = train_labels_scaler.inverse_transform(Y_hat_training_scaled)
    Y_training_real = train_labels_scaler.inverse_transform(scaled_train_labels)

    MSE_training_real = calcMSE(Y_hat_training_real, Y_training_real)
    MAPE_training_real = calcMAPE(Y_hat_training_real, Y_training_real)
    print('MSE Error (Training)= ', MSE_training_real)
    print('MAPE Error (Training) = ', MAPE_training_real)




    # These Y_TE set is trained using the final model returned by GridSearch:
    Y_hat_test_scaled = best_trained_model.predict(X_test[:, final_sel_idx])

    # scale back the data to compute the MSE and MAPE
    Y_hat_test_real = test_labels_scaler.inverse_transform(Y_hat_test_scaled)
    Y_test_real = test_labels_scaler.inverse_transform(scaled_test_labels)

    MSE_test_real = calcMSE(Y_hat_test_real, Y_test_real)
    MAPE_test_real = calcMAPE(Y_hat_test_real, Y_test_real)
    print('MSE Error (Test) = ', MSE_test_real)
    print('MAPE Error (Test) = ', MAPE_test_real)




    # save info about model in the data variable

    data_conf['Results'] = {}
    data_conf['Results']['alpha'] = Least_MSE_alpha
    data_conf['Results']['Sel_features'] = cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features']
    data_conf['Results']['Sel_features_name'] = cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features_Names']
    data_conf['Results']['Model_size'] = len(cv_info['alpha = '+str(Least_MSE_alpha)]['Selected_Features'])



    data_conf['Results']['MSE_Error_TR'] = MSE_training_real
    data_conf['Results']['MSE_Error_TE'] = MSE_test_real
    data_conf['Results']['MAPE_Error_Training'] =  MAPE_training_real
    data_conf['Results']['MAPE_Error_Test'] = MAPE_test_real
    data_conf['Results']['Y_training_real'] = Y_training_real.tolist()
    data_conf['Results']['Y_hat_training_real'] = Y_hat_training_real.tolist()
    data_conf['Results']['Y_test_real'] = Y_test_real.tolist()
    data_conf['Results']['Y_hat_test_real'] = Y_hat_test_real.tolist()

    results.append(data_conf)






# save the data in a file
#with open('results.json', 'w') as fp:
#    json.dump(data_conf, fp)


plot_histogram(results, result_path)
plot_MSE_Errors(results, result_path)
plot_MAPE_Errors(results, result_path)
plot_Model_Size(results, result_path)
plot_predicted_true(results, result_path)



finalresults = {}
finalresults["case1"] = results

f = open('resultslist.pckl', 'wb')
pickle.dump(results, f)
f.close()


#f = open('resultslist.pckl', 'rb')
#savedresults = pickle.load(f)
#f.close()



# save the data

with open('output.txt', 'wt') as out:
    pprint(finalresults, stream=out)

with open('output2.txt', 'wt') as out:
    pprint(results, stream=out)


