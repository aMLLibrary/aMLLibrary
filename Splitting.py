from SequenceDataProcessing import *
import logging


class Task(object):
    def __init__(self):
        self.inputDF = None  # Check with Marco this is a DF, I would create an empty DF
        self.outputDF = None

class DataPrepration(Task):
    """This is the main class defining the pipeline of machine learning task"""

    def __init__(self):
        Task.__init__(self)


class Splitting(DataPrepration):
    """performs splitting of the data based on the input parameters and scaling them """
    def __init__(self):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, inputDF, parameters, run_info):
        """performs scaling and splitting"""

        self.logger.info("Splitting dataset: ")
        self.inputDF = inputDF

        # retrieve the seed and shuffle
        seed = parameters['Splitting']['seed_vector'][len(run_info)-1]
        self.inputDF = shuffle(self.inputDF, random_state=seed)

        # find the samples in training and test set
        train_indices, test_indices = self.getTRTEindices(self.inputDF, parameters)

        # populate the run_info variable
        run_info[-1]['train_indices'] = train_indices
        run_info[-1]['test_indices'] = test_indices

        # split the data
        train_df = inputDF.ix[train_indices]
        test_df = inputDF.ix[test_indices]
        train_labels = train_df.iloc[:, 0]
        train_features = train_df.iloc[:, 1:]

        test_labels = test_df.iloc[:, 0]
        test_features = test_df.iloc[:, 1:]

        # populate the necessary variables in run_info for later use
        run_info[-1]['train_features'] = train_features
        run_info[-1]['train_labels'] = train_labels
        run_info[-1]['test_features'] = test_features
        run_info[-1]['test_labels'] = test_labels

        return train_features, train_labels, test_features, test_labels

    def getTRTEindices(self, df, parameters):
        """find training and test indices in the data based on the datasize and core numbers"""

        # retrieve needed parameters
        image_nums_train_data = parameters['Splitting']['image_nums_train_data']
        image_nums_test_data = parameters['Splitting']['image_nums_test_data']
        core_nums_train_data = parameters['Splitting']['core_nums_train_data']
        core_nums_test_data = parameters['Splitting']['core_nums_test_data']

        # group the samples according to the dataSize column of dataset since this column is one the features that tell
        # with samples are in training set and which are in the test set
        data_size_indices = pd.DataFrame(
            [[k, v.values] for k, v in df.groupby('dataSize').groups.items()], columns=['col', 'indices'])

        # save the samples that can be possible in the training(test) set because their dataSize column value
        # correspond to the predefined value from the input parameters
        data_size_train_indices = \
            data_size_indices.loc[(data_size_indices['col'].isin(image_nums_train_data))]['indices']
        data_size_test_indices = \
            data_size_indices.loc[(data_size_indices['col'].isin(image_nums_test_data))]['indices']

        # gather all the candidate samples from above into a list for training(test) set
        data_size_train_indices = np.concatenate(list(data_size_train_indices), axis=0)
        data_size_test_indices = np.concatenate(list(data_size_test_indices), axis=0)

        # group the samples according to the nContainers column of dataset since this column is one the features that
        # tell with samples are in training set and which are in the test set
        core_num_indices = pd.DataFrame(
            [[k, v.values] for k, v in df.groupby('nContainers').groups.items()],
            columns=['col', 'indices'])

        # For interpolation and extrapolation, put all the cores to the test set.
        print('image_nums_train_data: ', image_nums_train_data)
        print('image_nums_test_data: ', image_nums_test_data)
        if set(image_nums_train_data) != set(image_nums_test_data):
            core_nums_test_data = core_nums_test_data + core_nums_train_data

        # save the samples that can be possible in the training(test) set because their core number column value
        # correspond to the predefined value from the input parameters
        core_num_train_indices = \
            core_num_indices.loc[(core_num_indices['col'].isin(core_nums_train_data))]['indices']
        core_num_test_indices = \
            core_num_indices.loc[(core_num_indices['col'].isin(core_nums_test_data))]['indices']

        # gather all the candidate samples from above into a list for training(test) set
        core_num_train_indices = np.concatenate(list(core_num_train_indices), axis=0)
        core_num_test_indices = np.concatenate(list(core_num_test_indices), axis=0)

        # data_conf["core_nums_train_data"] = core_nums_train_data
        # data_conf["core_nums_test_data"] = core_nums_test_data

        # Take the intersect of indices of datasize and core
        train_indices = np.intersect1d(core_num_train_indices, data_size_train_indices)
        test_indices = np.intersect1d(core_num_test_indices, data_size_test_indices)

        return train_indices, test_indices

