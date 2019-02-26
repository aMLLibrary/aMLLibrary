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
import xgboost as xgb
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import FeatureSelection
import Regression
import PreliminaryDataProcessing
import DataPreprocessing
import Splitting
import Normalization


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
        self.feature_selection = FeatureSelection.FeatureSelection()
        self.regression = Regression.Regression()
        self.results = Results()
        self.preliminary_data_processing = PreliminaryDataProcessing.PreliminaryDataProcessing("P8_kmeans.csv")
        self.data_preprocessing = DataPreprocessing.DataPreprocessing()
        self.data_splitting = Splitting.Splitting()
        self.normalization = Normalization.Normalization()

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
        self.parameters['FS']['XGBoost'] = bool(ast.literal_eval(self.conf.get('FS', 'XGBoost')))

        self.parameters['FS']['min_features'] = int(self.conf['FS']['min_features'])
        self.parameters['FS']['max_features'] = int(self.conf['FS']['max_features'])
        self.parameters['FS']['is_floating'] = bool(ast.literal_eval(self.conf.get('FS', 'is_floating')))
        self.parameters['FS']['fold_num'] = int(self.conf['FS']['fold_num'])
        self.parameters['FS']['Confidence_level'] = self.conf['FS']['Confidence_level']
        self.parameters['FS']['clipping_no'] = int(self.conf['FS']['clipping_no'])
        self.parameters['FS']['degree'] = int(self.conf['FS']['degree'])
        self.parameters['FS']['SFS_Ridge_param_list'] = self.conf.get('FS', 'SFS_Ridge_param_list')
        self.parameters['FS']['SFS_Ridge_param_list'] = [i for i in ast.literal_eval(self.parameters['FS']['SFS_Ridge_param_list'])]

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
        start = time.time()

        self.logger.info("Start of the algorithm")
        # performs reading data, drops irrelevant columns
        df = self.preliminary_data_processing.process(self.parameters)

        # performs inverting of the columns and adds combinatorial terms to the df
        ext_df = self.data_preprocessing.process(df, self.parameters)

        matlab_var = pd.DataFrame(data = 0, index = range(66), columns= ['test'])

        # performs the algorithm multiple time and each time changes the seed to shuffle
        for iter in range(self.run_num):

            this_run = 'run_' + str(iter)
            print('==================================================================================================')
            print(this_run)

            # the list containing all the information about runs of algorithm, for each run a dictionary is added
            # to the list
            self.run_info.append({})

            # performs data splitting and returns splitted data
            train_features, train_labels, test_features, test_labels = \
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


            print('err_train:', err_train)
            print('err_test:', err_test)
            # matlab_var['iter_'+str(iter)+'_y_true_train'] = list(y_true_train)
            # matlab_var['iter_'+str(iter)+'_y_pred_train'] = list(y_pred_train)
            # matlab_var['iter_'+str(iter)+'_y_true_test'] = list(y_true_test)
            # matlab_var['iter_'+str(iter)+'_y_pred_test'] = list(y_pred_test)

            # save the run_info variable as string in a temporary file in the result folder
            self.results.save_temporary_results(self.run_info)

        # saves the best run results and necessary plots in the defined folder in result directory
        self.results.process(ext_df, self.run_info, self.parameters)

        # matlab_var = matlab_var.drop(['test'], axis=1)
        # matlab_var.to_csv('matlab_var.csv', sep='\t')

        end = time.time()
        execution_time = str(end-start)
        print("Execution Time : " + execution_time)


class Task(object):
    def __init__(self):
        self.inputDF = None  # Check with Marco this is a DF, I would create an empty DF
        self.outputDF = None


class DataPrepration(Task):
    """This is the main class defining the pipeline of machine learning task"""

    def __init__(self):
        Task.__init__(self)


class DataAnalysis(Task):
    """This is class defining machine learning task"""

    def __init__(self):
        Task.__init__(self)
        self.parameters = {}


class Results(Task):
    """This is class defining machine learning task"""

    def __init__(self):
        Task.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, ext_df, run_info, parameters):

        self.logger.info("Preparing results and plots: ")
        # retrieve needed variables
        if parameters['FS']['select_features_sfs']:
            k_features = parameters['FS']['k_features']

        # find the best run among elements of run_info according to the minimum MAPE error
        best_run = self.select_best_run(run_info)

        # create the result name based on desired input parameters
        result_name = self.get_result_name(parameters)

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

    def get_result_name(self, parameters):
        """makes a name for saving the results and plots based on the current input parameters"""

        # retrieve necessary information to built the name of result folder
        degree = parameters['FS']['degree']
        select_features_sfs = parameters['FS']['select_features_sfs']
        select_features_vif = parameters['FS']['select_features_vif']
        XGBoost = parameters['FS']['XGBoost']

        is_floating = parameters['FS']['is_floating']
        run_num = parameters['General']['run_num']
        image_nums_train_data = parameters['Splitting']['image_nums_train_data']
        image_nums_test_data = parameters['Splitting']['image_nums_test_data']
        degree = str(degree)

        # concatenate parameters in a meaningful way
        result_name = "d=" + degree + "_"

        if select_features_vif:
            result_name += "vif_"

        if select_features_sfs:
            k_features = parameters['FS']['k_features']

            if not is_floating:
                result_name += "sfs_"
                result_name += str(k_features[0]) + '_'
                result_name += str(k_features[1])

            if is_floating:
                result_name += "sffs_"
                result_name += str(k_features[0]) + '_'
                result_name += str(k_features[1])

        if XGBoost:
            result_name += "XGB_"

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
        fs = FeatureSelection.FeatureSelection()

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
        fs = FeatureSelection.FeatureSelection()

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

        plot_path = os.path.join(result_path, "XGBoost_Feature_importance")
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

