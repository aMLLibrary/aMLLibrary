from SequenceDataProcessing import *
import logging


class Regression(DataAnalysis):

    def __init__(self):
        DataAnalysis.__init__(self)
        self.conf = cp.ConfigParser()
        self.logger = logging.getLogger(__name__)

    def process(self, y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, parameters, run_info):
        """computes the MAPE error of the real predication by first scaling the predicted values"""

        # retrieve the same scaler used for normalization to perform inverse transform
        scaler = run_info[-1]['scaler']

        # retrieve all the feature names
        features_names = parameters['Features']['Extended_feature_names']

        self.logger.info("Computing MAPE: ")
        # compute MAPE error for not scaled data
        err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores = \
            self.mean_absolute_percentage_error(y_pred_test, y_pred_train, test_features, test_labels, train_features,
                                                train_labels, scaler, features_names)

        test_var = pd.DataFrame(data = 0, index = range(len(y_true_train)) ,columns=['y_true_train','y_pred_train', 'y_true_test', 'y_pred_test'])

        test_var['y_true_train'] = y_true_train
        test_var['y_pred_train'] = y_pred_train
        test_var['y_true_test'] = y_true_test
        test_var['y_pred_test'] = y_pred_test

        print('err_train:', err_train)
        print('err_test:', err_test)

        test_var.to_csv(index=False)

        # populate run_info with results
        run_info[-1]['MAPE_test'] = err_test
        run_info[-1]['MAPE_train'] = err_train
        run_info[-1]['y_true_train'] = y_true_train
        run_info[-1]['y_pred_train'] = y_pred_train
        run_info[-1]['y_true_test'] = y_true_test
        run_info[-1]['y_pred_test'] = y_pred_test
        run_info[-1]['y_true_train_cores'] = y_true_train_cores
        run_info[-1]['y_pred_train_cores'] = y_pred_train_cores
        run_info[-1]['y_true_test_cores'] = y_true_test_cores
        run_info[-1]['y_pred_test_cores'] = y_pred_test_cores

        return err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test

    def mean_absolute_percentage_error(self, y_pred_test, y_pred_train, test_features, test_labels, train_features, train_labels, scaler, features_names):
        """computess MAPE error in real data by first scaling back the data (denormalize)"""

        # create the Data the same size and concatenate once real scaled output and once predicted output to make
        # data ready for inverse transform
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


        # add the scaled prediction data and real value to 2 different dataframe
        if y_pred_test != []:

            # Test error
            y_true_test = test_labels
            test_features_with_true = pd.DataFrame(test_features_org)
            test_features_with_pred = pd.DataFrame(test_features_org)

            # make the DataFrame out of array
            y_true_test = pd.DataFrame(y_true_test)
            y_pred_test = pd.DataFrame(y_pred_test)

            # make the index of test data alligned with training set
            y_pred_test.index = y_true_test.index

            # concatenate output columns
            test_features_with_true.insert(0, "y_true_test", y_true_test)
            test_features_with_pred.insert(0, "y_pred_test", y_pred_test)

            # make sure that each data has only predicted OR true value
            test_features_with_true.drop('y_pred_test', axis = 1, inplace=True)
            test_features_with_pred.drop('y_true_test', axis = 1, inplace=True)

            # do the inverse transform
            test_data_with_true = pd.DataFrame(scaler.inverse_transform(test_features_with_true.values))
            test_data_with_pred = pd.DataFrame(scaler.inverse_transform(test_features_with_pred.values))

            # label the columns names as it was before and add true and predicted value as column name
            test_data_with_true_cols = ['y_true_test']
            test_data_with_pred_cols = ['y_pred_test']
            for elem in features_names:
                test_data_with_true_cols.append(elem)
                test_data_with_pred_cols.append(elem)

            # make the indexing correct as it was before
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

            # select just the output to compute MAPE
            y_true_test = test_data_with_true.iloc[:, 0]
            y_pred_test = test_data_with_pred.iloc[:, 0]

            err_test = np.mean(np.abs((y_true_test - y_pred_test) / y_true_test)) * 100
        #if y_pred_test == []:
        #    err_test = -1

        # Train error
        y_true_train = train_labels

        # make the DataFrame out of matrix of training input
        train_features_with_true = pd.DataFrame(train_features_org)
        train_features_with_pred = pd.DataFrame(train_features_org)

        # make the DataFrame out of matrix of training output
        y_true_train = pd.DataFrame(y_true_train)
        y_pred_train = pd.DataFrame(y_pred_train)
        y_pred_train.index = y_true_train.index


        # concatenate output columns
        train_features_with_true.insert(0, "y_true_train", y_true_train)
        train_features_with_pred.insert(0, "y_pred_train", y_pred_train)

        # make sure DataFrame has only true OR predicted value
        train_features_with_true.drop('y_pred_train', axis=1, inplace=True)
        train_features_with_pred.drop('y_true_train', axis=1, inplace=True)

        # do the inverse transform
        train_data_with_true = pd.DataFrame(scaler.inverse_transform(train_features_with_true.values))
        train_data_with_pred = pd.DataFrame(scaler.inverse_transform(train_features_with_pred.values))

        # label the columns names as it was before and add true and predicted value as column name
        train_data_with_true_cols = ['y_true_train']
        train_data_with_pred_cols = ['y_pred_train']
        for elem in features_names:
            train_data_with_true_cols.append(elem)
            train_data_with_pred_cols.append(elem)

        # adjust the columns and indices names to be aligned
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

        # select just the output to compute MAPE
        y_true_train = train_data_with_true.iloc[:, 0]
        y_pred_train = train_data_with_pred.iloc[:, 0]

        # print('y_true_train:', y_true_train)
        # print('y_pred_train:', y_pred_train)
        # print('y_true_test:', y_true_test)
        # print('y_pred_test:', y_pred_test)

        err_train = np.mean(np.abs((y_true_train - y_pred_train) / y_true_train)) * 100

        return err_test, err_train, y_true_train, y_pred_train, y_true_test, y_pred_test, y_true_train_cores, y_pred_train_cores, y_true_test_cores, y_pred_test_cores

