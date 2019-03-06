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


class DataPreprocessing(DataPrepration):
    """performs invesing of needed features and adds them to the dataframe, extends the dataframe to a degree"""
    def __init__(self):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, inputDF, parameters):
        """inversing and extension of features in the dataframe"""
        self.inputDF = inputDF

        to_be_inv_list = parameters['Inverse']['to_be_inv_List']

        target_column = parameters['DataPreparation']['target_column']
        degree = parameters['FS']['degree']

        self.logger.info("Inverting relevant features: ")
        # add the inverted column(s)

        self.outputDF, inversing_cols = self.add_inverse_features(self.inputDF, to_be_inv_list)
        parameters['Inverse']['inversing_cols'] = inversing_cols

        self.logger.info("Extending features: ")
        # add combinatorial terms
        self.outputDF = self.add_all_comb(self.outputDF, inversing_cols, target_column, degree)

        # populate parameter variable with newly computed variables
        parameters['Features'] = {}
        parameters['Features']['Extended_feature_names'] = self.outputDF.columns.values[1:]
        parameters['Features']['Original_feature_names'] = self.inputDF.columns.values[1:]
        return self.outputDF

    def add_inverse_features(self, df, to_be_inv_list):
        """Given a dataframe and the name of columns that should be inversed, add the needed inversed columns and
        returns the resulting df and the indices of two reciprocals separately"""

        # a dictionary of DataFrame used for adding the inverted columns
        df_dict = dict(df)
        for c in to_be_inv_list:
            new_col = 1 / np.array(df[c])
            new_feature_name = 'inverse_' + c
            df_dict[new_feature_name] = new_col

        # convert resulting dictionary to DataFrame
        inv_df = pd.DataFrame.from_dict(df_dict)

        # returns the indices of the columns that should be inverted and their inverted in one tuple
        inverting_cols = []
        for c in to_be_inv_list:
            cidx = inv_df.columns.get_loc(c)
            cinvidx = inv_df.columns.get_loc('inverse_' + c)
            inv_idxs = (cidx, cinvidx)
            inverting_cols.append(inv_idxs)
        return inv_df, inverting_cols

    def add_all_comb(self, inv_df, inversed_cols_tr, output_column_idx, degree):
        """Given a dataframe, returns an extended df containing all combinations of columns except the ones that are
        inversed"""

        # obtain needed parameters for extending DataFrame
        features_names = inv_df.columns.values
        df_dict = dict(inv_df)
        data_matrix = pd.DataFrame.as_matrix(inv_df)
        indices = list(range(data_matrix.shape[1]))

        # compute all possible combinations with replacement
        for j in range(2, degree + 1):
            combs = list(itertools.combinations_with_replacement(indices, j))
            # finds the combinations containing features and inverted of them
            remove_list_idx = []
            for ii in combs:
                for kk in inversed_cols_tr:
                    if len(list(set.intersection(set(ii), set(kk)))) >= 2:
                        remove_list_idx.append(ii)
                if output_column_idx in ii:
                    remove_list_idx.append(ii)
            # removes the combinations containing features and inverted of them
            for r in range(0,len(remove_list_idx)):
                combs.remove(remove_list_idx[r])
            # compute resulting column of the remaining combinations and add to the df
            for cc in combs:
                new_col = self.calculate_new_col(data_matrix, list(cc))
                new_feature_name = ''
                for i in range(len(cc)-1):
                    new_feature_name = new_feature_name+features_names[cc[i]]+'_'
                new_feature_name = new_feature_name+features_names[cc[i+1]]

                # adding combinations each as a column to a dictionary
                df_dict[new_feature_name] = new_col
        # convert the dictionary to a dataframe
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

