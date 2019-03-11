import sys
import glob
import pandas as pd
import numpy as np
import math
import ast
import os
import logging
import configparser as cp
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import DataConversionWarning

INV_PREFIX = 'Inv_'
warnings.filterwarnings(action = 'ignore', category = DataConversionWarning)

def is_column_line(str):
    try:
        float(str)
        return False
    except ValueError:
        return True

class DataPreparation(object):
    def __init__(self):
        self.conf = cp.ConfigParser()
        self.conf.optionxform = str
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.input_name = ""
        self.case  =  ""
        self.split = ""

        self.df = []
        self.all_csv_df = []
        self.data_size_indices = []
        self.core_num_indices  = []

        self.data_size_train_indices = []
        self.data_size_test_indices = []

        self.core_num_train_indices = []
        self.core_num_test_indices = []

        self.sparkdl_inputs = ["runbest","classifierselection","hyperparamopt"]
        self.tpcds_inputs   = ["query19","query20","query21","query26","query40","query32","query52"]
        self.tf_deepspeech_inputs   = ["tf_deepspeech"]


    def read_inputs(self, config_path, input_path, gaps_path):
        
        self.conf.read(config_path)
        debug = ast.literal_eval(self.conf.get('DebugLevel', 'debug'))
        logging.basicConfig(level = logging.DEBUG) if debug else logging.basicConfig(level = logging.INFO)
        target_column       = int(self.conf.get('DataPreparation', 'target_column'))
        self.input_name     = self.conf.get('DataPreparation', 'input_name')
        self.case           = self.conf.get('DataPreparation', 'case')
        self.split          = self.conf.get('DataPreparation', 'split')
        self.test_size = float(self.conf.get('DataPreparation', 'test_size'))
        self.remove_fractional = ast.literal_eval(self.conf.get('DataPreparation', 'remove_fractional'))

        self.normalize_data = ast.literal_eval(self.conf.get('DataPreparation', 'normalize_data'))
        self.seed = int(self.conf.get('DataPreparation','seed'))
        self.extrapolate = self.conf.get('tf_deepspeech', 'extrapolate')
        if len(self.extrapolate) != 0:
            self.extrapolate = ast.literal_eval(self.extrapolate)

        self.feature_extrapolate = None
        self.train_extrapolate = None
        self.test_extrapolate = None
        if self.conf.has_option('tf_deepspeech','feature_extrapolate') \
                and self.conf.has_option('tf_deepspeech','train_extrapolate') \
                and self.conf.has_option('tf_deepspeech','test_extrapolate'):
            self.feature_extrapolate = ast.literal_eval(self.conf.get('tf_deepspeech','feature_extrapolate'))
            self.train_extrapolate = ast.literal_eval(self.conf.get('tf_deepspeech','train_extrapolate'))
            self.test_extrapolate = ast.literal_eval(self.conf.get('tf_deepspeech','test_extrapolate'))


        self.logger.debug('Current CSV: ' + str(input_path))
        self.df = pd.read_csv(input_path)

        #PROVA LOAD TUTTI I DATI
        # self.load_all_timestamp()

        #PROVA Raggruppamento dati con limite
        # self.reduce_groups_data()

        if self.df.empty:
            with open(os.path.join(os.getcwd(), self.input_name + '_output'), 'w') as fw:
                res = "\nused_features_number =  0 \nmape_test =  0 \nmodel = NO MODEL"
                fw.write(res)
            self.logger.info('EMPTY DATAFRAME')
            sys.exit()

        if self.remove_fractional:
            self.df = self.df.loc[self.df['epochs(18)']%1 == 0].reset_index(drop=True)

        self.logger.debug('Current Dataframe: \n' + str(self.df.head()))
        self.df = shuffle(self.df, random_state = self.seed)
        self.logger.debug('Shuffle Dataframe: \n' + str(self.df.head()))
        self.logger.debug('Index before Reset: ' + str(self.df.index))
        self.df.reset_index(drop=True,inplace=True)
        self.logger.debug('Dataframe after Reset: \n' + str(self.df.head()))
        self.logger.debug('Index after Reset: ' + str(self.df.index))

        #Two different ways for extrapolation
        if len(self.extrapolate) != 0:
            #In this way there is a defined threshold for splitting test and train
            self.logger.debug('Extrapolation Enabled: ' + str(self.extrapolate))
            train_filter, test_filter = self.filter_feature(self.df, self.extrapolate)
            self.df = self.df.loc[(train_filter | test_filter),:]
            self.train_filter, self.test_filter = self.filter_feature(self.df, self.extrapolate)
            self.logger.debug('Train Dataframe should be: \n' + str(self.df.loc[train_filter,:].head()))
            self.logger.debug('Test Dataframe should be: \n' + str(self.df.loc[test_filter,:].head()))
        elif not self.feature_extrapolate is None\
             and not self.train_extrapolate is None\
             and not self.test_extrapolate is None:
            self.logger.debug('Extrapolate : ' + str(self.feature_extrapolate))
            self.logger.debug('Train : ' + str(self.train_extrapolate))
            self.logger.debug('Test : ' + str(self.test_extrapolate))
            #In this way two different set for train and test need to be fully specified
            train_filter, test_filter = self.filter_feature_values(self.df, self.feature_extrapolate, self.train_extrapolate, self.test_extrapolate)
            self.df = self.df.loc[(train_filter | test_filter), :]
            self.train_filter, self.test_filter = self.filter_feature_values(self.df, self.feature_extrapolate, self.train_extrapolate, self.test_extrapolate)
            self.logger.debug('Train Dataframe should be: \n' + str(self.df.loc[train_filter, :].head()))
            self.logger.debug('Test Dataframe should be: \n' + str(self.df.loc[test_filter, :].head()))

        self.all_csv_df = self.df.copy()

        if target_column != 1:
            column_names = list(self.df)
            self.logger.debug('Initial columns: ' + str(column_names))
            column_names[1], column_names[target_column] = column_names[target_column], column_names[1]
            self.df = self.df.reindex(columns = column_names)
            self.logger.debug('New columns(target moved) : ' + str(list(self.df.columns)))

        new = pd.DataFrame()
        target_name = self.df.columns[1]
        new[target_name] = self.df[target_name]

        features_conf = self.conf.get('tf_deepspeech', 'features')
        list_features = []
        if len(features_conf) != 0:
            list_features = ast.literal_eval(features_conf)
        for el in list_features:
            new[el] = self.df[el]

        self.include_inverse = ast.literal_eval(self.conf.get('tf_deepspeech','include_inverse'))
        if self.include_inverse:
            for c in new.columns[1:]:
                new_column = INV_PREFIX + str(c)
                new[new_column] = 1 / self.df[c]

        self.df = new
        self.logger.debug('DF at End data preparation : \n' + str(self.df.head()))
        self.logger.info("Data Preparation - tf_deepspeech")

    def reduce_groups_data(self):
        # Now datas are reduced, and will be taken at most the minimum of runs available for each signature
        groups = self.complete_df.groupby(
            ['GPU number(9)', 'batch size(13)', 'Computed Iterations Number(16)', 'CPU threads(19)', 'GFlops(7)'])
        groups = self.complete_df.groupby(
            ['GPU number(9)', 'batch size(13)', 'Computed Iterations Number(16)', 'CPU threads(19)'])
        max_elements = np.min(groups.size())
        self.reduced_df = pd.DataFrame()
        for (c, g) in groups:
            tempdf = shuffle(g, random_state=self.seed)
            self.reduced_df = self.reduced_df.append(tempdf.iloc[:max_elements - 1], ignore_index=True)
        self.df = self.reduced_df
        self.logger.debug('Reduced Dataframe containts: ' + str(self.df.shape[0]) + ' rows')

    def load_all_timestamp(self):
        # Starting from summary csv, all the iterations are loaded into df
        self.complete_df = pd.DataFrame(columns=self.df.columns)
        all_timestamps = self.df['starting timestamp(0)']
        array_unique_timestamps = all_timestamps.unique()
        self.logger.debug('Unique timestamps in initial csv file: ' + str(len(array_unique_timestamps)))
        files_to_consider = []
        mappings = {
            'epochs(18)': 'Epoch',
            'Real Iterations Number(15)': 'Iteration',
            'data time(27)': 'DataTime',
            'training time(28)': 'TrainingTime'
        }
        for timestamp in array_unique_timestamps:
            script_path = os.path.realpath(__file__)
            # file = glob.glob(os.path.join(os.path.dirname(input_path), '**/*' + str(timestamp) + '*'), recursive=True)[
            #     0]
            if len(glob.glob(os.path.join(os.path.dirname(script_path), '**/*' + str(timestamp) + '*'),
                             recursive=True)) != 1:
                print(str(timestamp) + ' not found or duplicate')
                sys.exit()
            file = glob.glob(os.path.join(os.path.dirname(script_path), '**/*' + str(timestamp) + '*'), recursive=True)[
                0]
            temp_new = pd.read_csv(file)
            temp_new = temp_new.loc[temp_new['Phase'] == 'Training']
            temp_new_filtered = pd.DataFrame()
            for e in temp_new['Epoch'].unique():
                if e == 0:
                    to_add = temp_new.loc[temp_new['Epoch'] == e].iloc[20:, :]
                    temp_new_filtered = temp_new_filtered.append(to_add, ignore_index=True)
                else:
                    to_add = temp_new.loc[temp_new['Epoch'] == e].iloc[1:, :]
                    temp_new_filtered = temp_new_filtered.append(to_add, ignore_index=True)
                # temp_new.loc[temp_new['Epoch'] == e]
            temp_new = temp_new_filtered
            row = self.df.loc[self.df['starting timestamp(0)'] == timestamp].iloc[0, :]
            for k, v in row.items():
                if k not in mappings.keys():
                    temp_new[k] = v
            for m in mappings.keys():
                temp_new[m] = temp_new[mappings[m]]
            if temp_new['data time(27)'][0] != 'nan' and not temp_new['data time(27)'].isnull().values.any():
                temp_new['Target Column(34)'] = temp_new['data time(27)'] + temp_new['training time(28)']
            else:
                temp_new['Target Column(34)'] = temp_new['training time(28)']
            self.complete_df = self.complete_df.append(temp_new[self.df.columns], ignore_index=True)
            # files_to_consider.append(glob.glob(os.path.join(os.path.dirname(input_path), '**/*' + str(timestamp) + '*'), recursive=True)[0])
        self.df = self.complete_df
        self.logger.debug('Loading all csv complete, df contains: ' + str(self.df.shape[0]) + ' rows')

    def scale_data(self, df):
        self.logger.debug('Scaling Data')
        scaled_array = self.scaler.fit_transform(df.values)
        scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)
        # self.save_data(scaled_df)
        self.logger.debug('Scaled DF : \n' + str(scaled_df.head()))
        return scaled_df,self.scaler

    def save_data(self, dataframe):
        result_path = os.path.join(".", "results", self.conf.get('DataPreparation',"input_name"),'scaled_df.csv')
        dataframe.to_csv(result_path, index= False)
        return

    def extend_generate(self, config_path):
        from feature_extender import FeatureExtender
        #Remove constant features
        self.df = self.df.loc[:, (self.df != self.df.iloc[0]).any()]


        new_df = pd.DataFrame()

        list_initial_features = list(self.df.columns[1:])
        initial_features_df = self.df.iloc[:, 1:]

        extender = FeatureExtender()
        extender.read_inputs(config_path, self.all_csv_df)
        generated_features = extender.generate()
        print('Features Generated: ' + str(list(generated_features.columns)))

        if not initial_features_df.empty:
            extender = FeatureExtender()
            extender.read_inputs(config_path, initial_features_df)
            features_and_ext_df, list_features_and_ext = extender.extend()
            filter_extended_features = list(np.invert(
                                        features_and_ext_df.columns.isin(list_initial_features)))
            only_extended_features_df = features_and_ext_df.loc[:, filter_extended_features]
            #only_extended_features_df.reset_index(drop=True, inplace=True)

            generated_not_in_ext_df = generated_features.loc[:, list(np.invert(
                generated_features.columns.isin(only_extended_features_df.columns)))]

            new_df = pd.concat([new_df, only_extended_features_df], axis=1)
        else:
            generated_not_in_ext_df = generated_features

        new_df = pd.concat([new_df, generated_not_in_ext_df], axis=1)

        if not initial_features_df.empty:
            features_not_in_generated_df = initial_features_df.loc[:, list(np.invert(
                                    initial_features_df.columns.isin(generated_features.columns)))]
            # new_df = pd.concat([new_df, features_not_in_generated_df], axis=1)
            new_df = pd.concat([features_not_in_generated_df, new_df], axis=1)

        # Add target
        new_df.insert(0, self.df.columns[0], self.df.iloc[:, 0])

        self.df = new_df


        return

    def split_data(self, seed):

        data_conf = {}
        data_conf["case"] = self.case
        data_conf["split"] = self.split
        data_conf["input_name"] = self.input_name


        # Drop the constant columns
        self.df = self.df.loc[:, (self.df != self.df.iloc[0]).any()]
        # self.df = shuffle(self.df, random_state = seed)
        #if len(self.extrapolate) != 0:
        #    train_filter, test_filter = self.filter_feature(self.df, self.extrapolate)
        #    self.df = self.df.loc[(train_filter | test_filter),:]
        #    train_filter, test_filter = self.filter_feature(self.df, self.extrapolate)


        # Scale the data.
        if self.normalize_data:
            self.df,scaler = self.scale_data(self.df)
        else:
            scaler = None

        if len(self.extrapolate) == 0 and self.feature_extrapolate is None:
            if self.test_size != 1:
                self.train_df, self.test_df = train_test_split(self.df, test_size=self.test_size, random_state=seed)
            elif self.test_size ==1:
                self.train_df = self.df
                self.test_df = self.df
            else:
                print('Error, accepted test_size<=1')
                sys.exit(1)
        else:
            self.logger.debug('Extrapolation')
            self.train_df, self.test_df = self.df.loc[self.train_filter,:], self.df.loc[self.test_filter,:]
            self.logger.debug('Actual train : \n' + str(self.train_df.head()))
            self.logger.debug('Actual test : \n' + str(self.test_df.head()))

        train_labels   = self.train_df.iloc[:,0]
        train_features = self.train_df.iloc[:,1:]

        test_labels   = self.test_df.iloc[:,0]
        test_features = self.test_df.iloc[:,1:]

        # train_cores = cores.ix[train_indices]
        # test_cores = cores.ix[test_indices]
        # data_conf["train_cores"] = train_cores
        # data_conf["test_cores"] = test_cores
        # features_names[0] : applicationCompletionTime
        features_names = list(self.df.columns.values)[1:]
        data_conf["train_features_org"] = train_features.as_matrix()
        data_conf["test_features_org"]  = test_features.as_matrix()
        # print(features_names)

        return train_features,train_labels,test_features,test_labels,features_names,scaler,data_conf

    def filter_feature(self, dataframe, extrapolate_params):
        if isinstance(extrapolate_params, list):
            train_filter = pd.Series(np.ones(dataframe.shape[0], dtype=bool))
            test_filter = pd.Series(np.ones(dataframe.shape[0], dtype=bool))
            for tup in extrapolate_params:
                feature_name, max_train, min_test = tup
                train_filter &= dataframe[feature_name] <= max_train
                test_filter &= dataframe[feature_name] > min_test
        else:
            feature_name, max_train, min_test = extrapolate_params
            train_filter = dataframe[feature_name] <= max_train
            test_filter = dataframe[feature_name] > min_test
        return train_filter, test_filter

    def filter_feature_values(self, dataframe, feature_name, train_values, test_values):
        train_filter = pd.Series([el in train_values for el in dataframe[feature_name]])
        test_filter = pd.Series([el in test_values for el in dataframe[feature_name]])
        return train_filter, test_filter



