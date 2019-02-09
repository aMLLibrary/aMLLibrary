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


class SequenceDataProcessing(object):
    def __init__(self):
        self.conf = cp.ConfigParser()
        self.parameters = {}
        self.get_parameters()
        self.seed_v = []
        self.run_info = []
        self.data_config = {}
        self.run_num = self.parameters['General']['run_num']
        self.result_path = self.parameters['DataPreparation']['result_path']
        self.preliminary_data_processing = PreliminaryDataProcessing("P8_kmeans.csv")
        self.data_preprocessing = DataPreprocessing()
        self.data_splitting = Splitting()
        self.normalization = Normalization()
        # self.seed_v = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 1023]
        self.seed_v = self.parameters['Splitting']['seed_vector']
        self.scaler = None
        self.feature_selection = FeatureSelection()
        self.regression = Regression()
        self.results = Results()

    def get_parameters(self):
        """Gets the parameters from the config file named parameters.ini and put them into a dictionary
        named parameters"""
        self.conf.read('params.ini')
        self.parameters['General'] = {}
        self.parameters['General']['run_num'] = int(self.conf['General']['run_num'])
        self.parameters['DataPreparation'] = {}
        self.parameters['DataPreparation']['result_path'] = self.conf['DataPreparation']['result_path']
        self.parameters['Splitting'] = {}
        self.parameters['Splitting']['seed_vector'] = self.conf['Splitting']['seed_vector']
        self.parameters['Splitting']['seed_vector'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['seed_vector'])]


    def process(self):

        # df = read_inputs()
        temp = self.preliminary_data_processing.process(None)

        # splitting_df = self.data_splitting.make_splitting_df(temp, self.data_splitting.criterion_col_list)

        temp = self.data_preprocessing.process(temp)


        for iter in range(self.run_num):
            self.result_path = "./results/"
            this_run = 'run_' + str(iter)
            print(this_run)
            self.run_info.append({})

        #    Should be replaced with basic functions of splitting:
        #    temp = self.data_splitting.process(temp, splitting_df, self.seed_v[iter])

            train_features, train_labels, test_features, test_labels, features_names, self.scaler \
                = self.data_splitting.process(temp, self.seed_v[iter])
            print(self.seed_v[iter])
            self.run_info[iter]['ext_feature_names'] = features_names

            cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test = \
                    self.feature_selection.process(train_features, train_labels, test_features, test_labels, features_names)

            self.run_info[iter]['cv_info'] = cv_info
            self.run_info[iter]['Sel_features'] = list(sel_idx)
            self.run_info[iter]['Sel_features_names'] = [features_names[i] for i in sel_idx]
            self.run_info[iter]['best_param'] = Least_MSE_alpha
            self.run_info[iter]['best_model'] = best_trained_model

            err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores = \
            self.regression.process(y_pred_test, y_pred_train, test_features, test_labels, train_features,
            train_labels, self.scaler, features_names)


            self.run_info[iter]['MAPE_train'] = err_train
            self.run_info[iter]['MAPE_test'] = err_test
            self.run_info[iter]['y_true_train'] = y_true_train
            self.run_info[iter]['y_pred_train'] = y_pred_train
            self.run_info[iter]['y_true_test'] = y_true_test
            self.run_info[iter]['y_pred_test'] = y_pred_test
            self.run_info[iter]['data_conf'] = self.data_config
            self.results.save_temporary_results(self.run_info)
            print('tamoom shod iter')
        print(self.parameters)
        self.results.process(self.run_info)





        #run_info = []


        #for iter in range(run_num):
        #     result_path = "./results/"
        #
        #     this_run = 'run_' + str(iter)
        #     print(this_run)
        #
        #     run_info.append({})
        #
        #     ext_df = add_all_comb(df, inversing_cols, 0, degree)
        #
        #     train_features, train_labels, test_features, test_labels, features_names, scaler, data_conf = \
        #         split_data(seed_v[iter], ext_df, image_nums_train_data, image_nums_test_data, core_nums_train_data,
        #                    core_nums_test_data)
        #
        #     run_info[iter]['ext_feature_names'] = features_names
        #     run_info[iter]['data_conf'] = data_conf
        #
        #     k_features = calc_k_features(min_features, max_features, features_names)
        #
        #     print('selecting features in range ', k_features, ':')
        #
        #     cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test = \
        #         Ridge_SFS_GridSearch(ridge_params, train_features, train_labels, test_features, test_labels, k_features,
        #                              fold_num)
        #
        #     run_info[iter]['cv_info'] = cv_info
        #     run_info[iter]['Sel_features'] = list(sel_idx)
        #     run_info[iter]['Sel_features_names'] = [features_names[i] for i in sel_idx]
        #     run_info[iter]['best_param'] = Least_MSE_alpha
        #     run_info[iter]['best_model'] = best_trained_model
        #
        #     err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores = \
        #         mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features,
        #                                        train_labels, scaler)
        #
        #     run_info[iter]['MAPE_train'] = err_train
        #     run_info[iter]['MAPE_test'] = err_test
        #     run_info[iter]['y_true_train'] = y_true_train
        #     run_info[iter]['y_pred_train'] = y_pred_train
        #     run_info[iter]['y_true_test'] = y_true_test
        #     run_info[iter]['y_pred_test'] = y_pred_test
        #
        #     target = open(os.path.join('./results/', "temp_run_info"), 'a')
        #     target.write(str(run_info))
        #     target.close()
        #
        # best_run_idx, best_data_conf, best_cv_info, best_trained_model, best_Least_MSE_alpha, best_err_train, best_err_test = \
        #     select_best_run(run_info)
        #
        # result_name = get_result_name(degree, select_features_sfs, k_features, is_floating)
        #
        # result_path, results = save_results(best_err_train, best_err_test, result_name, result_path,
        #                                     best_data_conf, best_cv_info, ridge_params, best_trained_model, degree,
        #                                     best_Least_MSE_alpha)
        #
        # plot_predicted_true(result_path, run_info, best_run_idx)
        # plot_cores_runtime(result_path, run_info, best_run_idx, core_nums_train_data, core_nums_test_data)
        # plot_histogram(result_path, run_info, degree)
        # plot_MSE_Errors(result_path, run_info)
        # plot_MAPE_Errors(result_path, run_info)
        # plot_Model_Size(result_path, run_info)
        #
        # target = open(os.path.join(result_path, "run_info"), 'a')
        # target.write(str(run_info))
        # target.close()

class Task(object):
    def __init__(self):
        self.inputDF = None  # Check with Marco this is a DF, I would create an empy DF
        self.outputDF = None


class DataPrepration(Task):
    """This is the main class defining the pipeline of machine learning task"""

    def __init__(self):
        Task.__init__(self)
        self.parameters = {}
        self.case = ""
        self.split = ""

        self.df = []
        self.data_size_indices = []
        self.core_num_indices = []

        self.data_size_train_indices = []
        self.data_size_test_indices = []

        self.core_num_train_indices = []
        self.core_num_test_indices = []
        self.seed_v = []
        self.features_name = []



class Normalization(DataPrepration):

    def __init__(self):
        DataPrepration.__init__(self)
        # self.scaler = StandardScaler()

    # def process(self, inputDF):
    #     print('process of normalization')
    #     self.inputDF = inputDF
    #     self.outputDF, self.scaler = self.scale_data(self.inputDF)
    #
    #     return self.outputDF, self.scaler

    # def scale_data(self, df):
    #     """scale the dataframe"""
    #     scaled_array = self.scaler.fit_transform(df.values)
    #     scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
    #     return scaled_df, self.scaler


class PreliminaryDataProcessing(DataPrepration):

    def __init__(self, input_file):
        DataPrepration.__init__(self)
        self.scale = StandardScaler()
        self.raw_data = []
        self.input_path = input_file

    def process(self, inputDF):
        """Get the csv file and change it to data frame as output"""
        self.outputDF = pd.read_csv(self.input_path)

        # drop the run column
        self.outputDF = self.outputDF.drop(['run'], axis=1)

        # drop constant columns
        self.outputDF = self.outputDF.loc[:, (self.outputDF != self.outputDF.iloc[0]).any()]

        return self.outputDF


class DataPreprocessing(DataPrepration):

    def __init__(self):
        DataPrepration.__init__(self)

        self.conf = cp.ConfigParser()
        # self.to_be_inv_List = ['nContainers']
        self.parameters = {}
        self.get_parameters()
        self.to_be_inv_List = self.parameters['Inverse']['to_be_inv_List']
        self.inversing_cols = None
        self.degree = self.parameters['FeatureExtender']['degree']


    def process(self, inputDF):
        self.inputDF = inputDF
        self.outputDF, self.inversing_cols = self.add_inverse_features(self.inputDF, self.to_be_inv_List)
        self.outputDF = self.add_all_comb(self.outputDF, self.inversing_cols, 0, self.degree)

        return self.outputDF

    def get_parameters(self):
        """Gets the parameters from the config file named parameters.ini and put them into a dictionary
        named parameters"""
        self.conf.read('params.ini')
        self.parameters['Inverse'] = {}
        self.parameters['Inverse']['to_be_inv_List'] = [str(self.conf['Inverse']['to_be_inv_List'])]
        self.parameters['FeatureExtender'] = {}
        self.parameters['FeatureExtender']['degree'] = int(self.conf['FeatureExtender']['degree'])


    def add_inverse_features(self, df, to_be_inv_List):
        """Given a dataframe and the name of columns that should be inversed, add the needed inversed columns and returns
        the resulting df and the indices of two reciprocals separately"""

        df_dict = dict(df)
        for c in to_be_inv_List:
            new_col = 1 / np.array(df[c])
            new_feature_name = 'inverse_' + c
            df_dict[new_feature_name] = new_col

        inv_df = pd.DataFrame.from_dict(df_dict)

        # returns the indices of the columns that should be inversed and their inversed in one tuple
        inversing_cols = []
        for c in to_be_inv_List:
            cidx = inv_df.columns.get_loc(c)
            cinvidx = inv_df.columns.get_loc('inverse_' + c)
            inv_idxs = (cidx, cinvidx)
            inversing_cols.append(inv_idxs)
        return inv_df, inversing_cols


    def add_all_comb(self, inv_df, inversed_cols_tr, output_column_idx, degree):
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
                new_col = self.calculate_new_col(data_matrix, list(cc))
                new_feature_name = ''
                for i in range(len(cc)-1):
                    new_feature_name = new_feature_name+features_names[cc[i]]+'_'
                new_feature_name = new_feature_name+features_names[cc[i+1]]
                df_dict[new_feature_name] = new_col

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

    def __init__(self):
        DataPrepration.__init__(self)
        self.conf = cp.ConfigParser()
        self.parameters = {}
        self.get_parameters()
        self.case = self.parameters['DataPreparation']['case']
        self.split = self.parameters['DataPreparation']['split']
        self.training_indices = []
        self.test_indices = []
        # self.data_size_indices = []
        # self.core_num_indices = []
        self.input_name = self.parameters['DataPreparation']['input_name']

        #self.data_size_train_indices = self.parameters['Splitting']['image_nums_train_data']
        #self.data_size_test_indices = self.parameters['Splitting']['image_nums_test_data']

        self.core_nums_train_data = self.parameters['Splitting']['core_nums_train_data']
        self.core_nums_test_data = self.parameters['Splitting']['core_nums_test_data']

        self.image_nums_train_data = self.parameters['Splitting']['image_nums_train_data']
        self.image_nums_test_data = self.parameters['Splitting']['image_nums_test_data']

        self.criterion_col_list = self.parameters['Splitting']['criterion_col_list']

        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.features_names = []

        self.scaler = StandardScaler()


    def process(self, inputDF, seed):

        self.inputDF = inputDF
        self.train_features, self.train_labels, self.test_features, self.test_labels, self.features_names, self.scaler = \
            self.split_data(seed, self.inputDF, )


        return self.train_features, self.train_labels, self.test_features, self.test_labels, self.features_names, self.scaler


        #print(self.splitting_df)
        #print(self.outputDF)

        # locate training and test indices
        # self.training_indices, self.test_indices = self.getTRTEindices(self.outputDF)
        # print(self.training_indices)
        # print(self.test_indices)

        #self.outputDF = self.split_data(seed, self.inputDF, self)

        #train_features, train_labels, test_features, test_labels, features_names, scaler, data_conf



    def get_parameters(self):
        """Gets the parameters from the config file named parameters.ini and put them into a dictionary
        named parameters"""
        self.conf.read('params.ini')
        self.parameters['Splitting'] = {}

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

        self.parameters['DataPreparation'] = {}
        self.parameters['DataPreparation']['input_name'] = self.conf.get('DataPreparation', 'input_name')
        self.parameters['DataPreparation']['split'] = self.conf.get('DataPreparation', 'split')
        self.parameters['DataPreparation']['case'] = self.conf.get('DataPreparation', 'case')



    def make_splitting_df (self, df, criterion_col_list):
        criterion_col_list = ['nContainers', 'dataSize']
        splitting_df = pd.DataFrame(index=range(df.shape[0]))
        splitting_df['original_index'] = df.index
        for col in criterion_col_list:
            splitting_df[col] = df[col]

        return splitting_df


    def shuffleSamples(self, df, splitting_df, seed):
        df , splitting_df= shuffle(df, splitting_df, random_state = seed)
        return df, splitting_df

    # def getTRTEindices(self, df):
    #
    #
    #     if "dataSize" in df.columns:
    #
    #         data_size_indices = pd.DataFrame(
    #             [[k, v.values] for k, v in df.groupby('dataSize').groups.items()], columns=['col', 'indices'])
    #
    #         data_size_train_indices = \
    #             data_size_indices.loc[(data_size_indices['col'].isin(self.image_nums_train_data))]['indices']
    #         data_size_test_indices = \
    #             data_size_indices.loc[(data_size_indices['col'].isin(self.image_nums_test_data))]['indices']
    #
    #         data_size_train_indices = np.concatenate(list(data_size_train_indices), axis=0)
    #         data_size_test_indices = np.concatenate(list(data_size_test_indices), axis=0)
    #
    #     # else:
    #     #
    #     #     data_size_train_indices = range(0, df.shape[0])
    #     #     data_size_test_indices = range(0, df.shape[0])
    #     #
    #     # data_conf["image_nums_train_data"] = image_nums_train_data
    #     # data_conf["image_nums_test_data"] = image_nums_test_data
    #     #
    #     # # if input_name in sparkdl_inputs:
    #     # # core_num_indices = pd.DataFrame(
    #     # # [[k, v.values] for k, v in df.groupby('nCores').groups.items()], columns=['col', 'indices'])
    #
    #     core_num_indices = pd.DataFrame(
    #         [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
    #         columns=['col', 'indices'])
    #
    #     # For interpolation and extrapolation, put all the cores to the test set.
    #     print('image_nums_train_data: ', self.image_nums_train_data)
    #     print('image_nums_test_data: ', self.image_nums_test_data)
    #     if set(self.image_nums_train_data) != set(self.image_nums_test_data):
    #         self.core_nums_test_data = self.core_nums_test_data + self.core_nums_train_data
    #
    #     core_num_train_indices = \
    #         core_num_indices.loc[(core_num_indices['col'].isin(self.core_nums_train_data))]['indices']
    #     core_num_test_indices = \
    #         core_num_indices.loc[(core_num_indices['col'].isin(self.core_nums_test_data))]['indices']
    #
    #     core_num_train_indices = np.concatenate(list(core_num_train_indices), axis=0)
    #     core_num_test_indices = np.concatenate(list(core_num_test_indices), axis=0)
    #
    #     #data_conf["core_nums_train_data"] = core_nums_train_data
    #     #data_conf["core_nums_test_data"] = core_nums_test_data
    #
    #     # Take the intersect of indices of datasize and core
    #     train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
    #     test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)
    #
    #     return train_indices, test_indices

    def scale_data(self, df):
        """scale the dataframe"""
        scaled_array = self.scaler.fit_transform(df.values)
        scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
        return scaled_df, self.scaler


    def split_data(self, seed, df):

        """split the original dataframe into Training Input (train_features), Training Output(train_labels),
        Test Input(test_features) and Test Output(test_labels)"""
        self.data_conf = {}
        self.data_conf["case"] = self.parameters['DataPreparation']['case']
        self.data_conf["split"] = self.parameters['DataPreparation']['split']
        self.data_conf["input_name"] = self.parameters['DataPreparation']['input_name']
        #self.data_conf["sparkdl_run"] = self.parameters['DataPreparation']['split']

        if self.input_name != "classifierselection":

            df = shuffle(df, random_state=seed)

            # If dataSize column has different values
            if "dataSize" in df.columns:
                data_size_indices = pd.DataFrame(
                    [[k, v.values] for k, v in df.groupby('dataSize').groups.items()], columns=['col', 'indices'])

                data_size_train_indices = \
                    data_size_indices.loc[(data_size_indices['col'].isin(self.image_nums_train_data))]['indices']
                data_size_test_indices = \
                    data_size_indices.loc[(data_size_indices['col'].isin(self.image_nums_test_data))]['indices']

                data_size_train_indices = np.concatenate(list(data_size_train_indices), axis=0)
                data_size_test_indices = np.concatenate(list(data_size_test_indices), axis=0)

            else:

                data_size_train_indices = range(0, df.shape[0])
                data_size_test_indices = range(0, df.shape[0])

            # data_conf["image_nums_train_data"] = image_nums_train_data
            # data_conf["image_nums_test_data"] = image_nums_test_data

            # if input_name in sparkdl_inputs:
            # core_num_indices = pd.DataFrame(
            # [[k, v.values] for k, v in df.groupby('nCores').groups.items()], columns=['col', 'indices'])

            core_num_indices = pd.DataFrame(
                [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
                columns=['col', 'indices'])

            # For interpolation and extrapolation, put all the cores to the test set.
            print('image_nums_train_data: ', self.image_nums_train_data)
            print('image_nums_test_data: ', self.image_nums_test_data)
            # if set(self.image_nums_train_data) != set(self.image_nums_test_data):
            #     core_nums_test_data = core_nums_test_data + core_nums_train_data

            core_num_train_indices = \
                core_num_indices.loc[(core_num_indices['col'].isin(self.core_nums_train_data))]['indices']
            core_num_test_indices = \
                core_num_indices.loc[(core_num_indices['col'].isin(self.core_nums_test_data))]['indices']

            core_num_train_indices = np.concatenate(list(core_num_train_indices), axis=0)
            core_num_test_indices = np.concatenate(list(core_num_test_indices), axis=0)

            self.data_conf["core_nums_train_data"] = self.core_nums_train_data
            self.data_conf["core_nums_test_data"] = self.core_nums_test_data

            # Take the intersect of indices of datasize and core
            self.train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
            self.test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)

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
        if self.input_name == "classifierselection":
            # Drop the constant columns
            df = df.loc[:, (df != df.iloc[0]).any()]
            cores = df["nCores"]
            # Read

            # data_conf["core_nums_train_data"] = core_nums_train_data
            # data_conf["core_nums_test_data"] = []
            # data_conf["image_nums_train_data"] = image_nums_train_data
            # data_conf["image_nums_test_data"] = []

            train_indices = range(0, len(cores))
            test_indices = []

        # Scale the data.
        df, self.scaler = self.scale_data(df)
        train_df = df.ix[self.train_indices]
        test_df = df.ix[self.test_indices]
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
        # data_conf["train_features_org"] = train_features.as_matrix()
        # data_conf["test_features_org"] = test_features.as_matrix()
        # # print(features_names)
        #
        # data_conf["test_without_apriori"] = False

        #return train_features, train_labels, test_features, test_labels, features_names, self.scaler, data_conf
        return train_features, train_labels, test_features, test_labels, features_names, self.scaler


class FeatureSelection(DataPrepration):

    def __init__(self):
        DataPrepration.__init__(self)
        self.conf = cp.ConfigParser()
        self.parameters = {}
        self.get_parameters()
        self.select_features_vif = self.parameters['FS']['select_features_vif']
        self.select_features_sfs = self.parameters['FS']['select_features_sfs']
        self.min_features = self.parameters['FS']['min_features']
        self.max_features = self.parameters['FS']['max_features']
        self.is_floating = self.parameters['FS']['is_floating']
        self.fold_num = self.parameters['FS']['fold_num']
        self.Confidence_level = self.parameters['FS']['Confidence_level']
        self.clipping_no = self.parameters['FS']['clipping_no']
        self.ridge_params = self.parameters['Ridge']['ridge_params']


    def get_parameters(self):
        """Gets the parameters from the config file named parameters.ini and put them into a dictionary
        named parameters"""
        self.conf.read('params.ini')
        self.parameters['FS'] = {}

        self.parameters['FS']['select_features_vif'] = bool(self.conf['FS']['select_features_vif'])
        self.parameters['FS']['select_features_sfs'] = bool(self.conf['FS']['select_features_sfs'])
        self.parameters['FS']['min_features'] = int(self.conf['FS']['min_features'])
        self.parameters['FS']['max_features'] = int(self.conf['FS']['max_features'])
        self.parameters['FS']['is_floating'] = bool(self.conf['FS']['is_floating'])
        self.parameters['FS']['fold_num'] = int(self.conf['FS']['fold_num'])
        self.parameters['FS']['Confidence_level'] = self.conf['FS']['Confidence_level']
        self.parameters['FS']['clipping_no'] = int(self.conf['FS']['clipping_no'])
        self.parameters['Ridge'] = {}
        self.parameters['Ridge']['ridge_params'] = self.conf['Ridge']['ridge_params']
        self.parameters['Ridge']['ridge_params'] = [i for i in ast.literal_eval(self.parameters['Ridge']['ridge_params'])]


    def process(self, train_features, train_labels, test_features, test_labels, features_names):

        k_features = self.calc_k_features(features_names)

        cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test =\
            self.Ridge_SFS_GridSearch(train_features, train_labels, test_features, test_labels, k_features)


        return cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test

    def Ridge_SFS_GridSearch(self, train_features, train_labels, test_features, test_labels, k_features):

        X = pd.DataFrame.as_matrix(train_features)
        Y = pd.DataFrame.as_matrix(train_labels)
        ext_feature_names = train_features.columns.values

        alpha_v = self.ridge_params

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
                      cv=self.fold_num,
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

        # Since the data for classsifierselection is too small, we only calculate the train error

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

        return cv_info, Least_MSE_alpha, sel_idx, best_trained_model, y_pred_train, y_pred_test

    def calc_k_features(self, features_names):
        """calculate the range of number of features that sfs is allowed to select"""

        # Selecting from all features
        if self.max_features == -1:
            k_features = (self.min_features, len(features_names))
            # Selecting from the given range
        if self.max_features != -1:
            k_features = (self.min_features, self.max_features)
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
        # self.parameters = {}
        # self.get_parameters()


    def process(self, y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler, features_names):

        err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores = \
            self.mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features,
                                                train_labels, scaler, features_names)

        return err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores

    def mean_absolute_percentage_error(self, y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler, features_names):


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

        y_true_train = train_data_with_true.iloc[:, 0]
        y_pred_train = train_data_with_pred.iloc[:, 0]

        err_train = np.mean(np.abs((y_true_train - y_pred_train) / y_true_train)) * 100

        return err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores

class Results(Task):
    """This is class defining machine learning task"""

    def __init__(self):
        Task.__init__(self)

    def process(self, run_info):
        best_run_idx, best_data_conf, best_cv_info, best_trained_model, best_Least_MSE_alpha, best_err_train, best_err_test = \
                self.select_best_run(run_info)
        # print(self.parameters)

        # result_name = self.get_result_name(degree, select_features_sfs, k_features, is_floating)
        #
        # result_path, results = save_results(best_err_train, best_err_test, result_name, result_path,
        #                                     best_data_conf, best_cv_info, ridge_params, best_trained_model, degree,
        #                                     best_Least_MSE_alpha)

        return

    def select_best_run(self, run_info):
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


    def get_result_name(self, degree, select_features_sfs, k_features, is_floating):

        if degree == []:
            result_name = "d=0_"
        if degree != []:
            # n_terms = list(map(lambda x: str(x), n_terms))
            # n_terms = ','.join(n_terms)
            degree = str(degree)
            result_name = "d=" + degree + "_"
        # if select_features_vif == True :
        #    result_name += "vif_"
        # if select_features_vif == False :
        #    result_name += "no_vif_"
        if select_features_sfs == False:
            result_name += "baseline_results"
        if select_features_sfs == True and is_floating == False:
            result_name += "sfs_"
            result_name += str(k_features[0]) + '_'
            result_name += str(k_features[1])
            # if max_k_features == -1:
            #    result_name += "all_features_results"
            # if max_k_features != -1 :
            #    result_name += str(self.max_k_features) + "_features_results"
        if select_features_sfs == True and is_floating == True:
            result_name += "sffs_"
            result_name += str(k_features[0]) + '_'
            result_name += str(k_features[1])
            # if self.max_k_features == -1 :
            #    self.result_name += "all_features_results"
            # if self.max_k_features != -1 :
            #   self.result_name += str(self.max_k_features) + "_features_results"
        # if self.data_conf["test_without_apriori"] == True:
        #    self.result_name += "_test_without_apriori"
        # if self.data_conf["fixed_features"] == True:
        #    self.result_name = "fixed_features"
        result_name = result_name + '_' + str(run_num) + '_runs'

        Tr_size = '_Tr'
        for sz in image_nums_train_data:
            Tr_size = Tr_size + '_' + str(sz)

        Te_size = '_Te'
        for sz in image_nums_test_data:
            Te_size = Te_size + '_' + str(sz)

        result_name = result_name + Tr_size + Te_size

        return result_name

    def save_temporary_results(self, run_info):
        target = open(os.path.join('./results/', "temp_run_info"), 'a')
        target.write(str(run_info))
        target.close()

