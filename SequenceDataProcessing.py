from sklearn.preprocessing import StandardScaler
import pandas as pd
import configparser as cp
import numpy as np
import ast
import itertools


class SequenceDataProcessing(object):
    def __init__(self):
        # self.steps = []
        self.seed_v = []
        self.run_info = []
        self.run_num = 10
        self.result_path = "./results/"
        self.preliminary_data_processing = PreliminaryDataProcessing("P8_kmeans.csv")
        self.data_preprocessing = DataPreprocessing()


    def process(self):

        seed_v = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 1023]

        # df = read_inputs()
        temp = self.preliminary_data_processing.process(None)

        # df, inversing_cols = add_inverse_features(df, to_be_inv_List)
        temp, inversing_cols = self.data_preprocessing.process(temp)

        # ext_df = add_all_comb(df, inversing_cols, 0, degree)
        temp = self.data_preprocessing.add_all_comb(temp, inversing_cols, 0, 2)
        print(temp)


        for iter in range(self.run_num):
            self.result_path = "./results/"
            this_run = 'run_' + str(iter)
            print(this_run)
            self.run_info.append({})


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
        self.scale = StandardScaler()


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
        self.to_be_inv_List = ['nContainers']

    def process(self, inputDF):
        self.inputDF = inputDF
        self.outputDF, self.inversing_cols = self.add_inverse_features(self.inputDF, self.to_be_inv_List)
        return self.outputDF, self.inversing_cols

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





class Splitting(DataPrepration):

    def __init__(self):
        DataPrepration.__init__(self)

        self.conf = cp.ConfigParser()
        self.parameters = {}
        self.get_parameters()
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



    def get_parameters(self):
        """Gets the parameters from the config file named parameters.ini and put them into a dictionary
        named parameters"""
        self.conf.read('params.ini')
        self.parameters['Splitting'] = {}

        self.parameters['Splitting']['image_nums_train_data'] = self.conf['Splitting']['image_nums_train_data']
        self.parameters['Splitting']['image_nums_train_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['image_nums_train_data'])]
        self.data_size_train_indices = self.parameters['Splitting']['image_nums_train_data']


        self.parameters['Splitting']['image_nums_test_data'] = self.conf.get('Splitting', 'image_nums_test_data')
        self.parameters['Splitting']['image_nums_test_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['image_nums_test_data'])]
        self.data_size_test_indices = self.parameters['Splitting']['image_nums_test_data']


        self.parameters['Splitting']['core_nums_train_data'] = self.conf.get('Splitting', 'core_nums_train_data')
        self.parameters['Splitting']['core_nums_train_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['core_nums_train_data'])]
        self.core_num_train_indices = self.parameters['Splitting']['core_nums_train_data']


        self.parameters['Splitting']['core_nums_test_data'] = self.conf.get('Splitting', 'core_nums_test_data')
        self.parameters['Splitting']['core_nums_test_data'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['core_nums_test_data'])]
        self.core_num_test_indices = self.parameters['Splitting']['core_nums_test_data']


        self.parameters['Splitting']['seed_vector'] = self.conf.get('Splitting', 'seed_vector')
        self.parameters['Splitting']['seed_vector'] = [int(i) for i in ast.literal_eval(self.parameters['Splitting']['seed_vector'])]
        self.seed_v = self.parameters['Splitting']['seed_vector']



        print(self.parameters)

    #def shuffleSamples(self):
    #    self.df = shuffle(self.df, random_state=seed)

    def split_data(self, seed):

        image_nums_train_data = (self.conf.get('DataPreparation', 'image_nums_train_data'))
        image_nums_train_data = [int(i) for i in ast.literal_eval(image_nums_train_data)]

        image_nums_test_data = (self.conf.get('DataPreparation', 'image_nums_test_data'))
        image_nums_test_data = [int(i) for i in ast.literal_eval(image_nums_test_data)]


class FeatureSelection(DataPrepration):

    def __init__(self):
        DataPrepration.__init__(self)



class DataAnalysis(Task):
    """This is class defining machine learning task"""

    def __init__(self):
        Task.__init__(self)
        self.parameters = {}


class Regression(DataAnalysis):

    def __init__(self):
        DataAnalysis.__init__(self)