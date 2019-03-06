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


class PreliminaryDataProcessing(DataPrepration):
    """Perform preliminary prossing of data"""
    def __init__(self, input_file):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, parameters):
        """Get the csv file, drops the irrelevant columns and change it to data frame as output"""
        # input_path = parameters['DataPreparation']['input_path']
        # self.logger.info("Input reading: " + input_path)
        # self.outputDF = pd.read_csv(input_path)

        self.outputDF = self.read_csv(parameters)

        target_column = parameters['DataPreparation']['target_column']

        if target_column != 0:
            self.outputDF = self.makeTargetColumnTheFirst(self.outputDF, target_column)

        # set the target_column
        parameters['DataPreparation']['target_column'] = 0

        # drop the run column
        dropping_col = parameters['DataPreparation']['irrelevant_column_name']

        # self.outputDF = self.outputDF.drop(dropping_col, axis=1)
        self.outputDF = self.dropIrrelevantColumns(self.outputDF, dropping_col)

        # drop constant columns
        self.outputDF = self.outputDF.loc[:, (self.outputDF != self.outputDF.iloc[0]).any()]

        return self.outputDF

    def read_csv(self, parameters):
        input_path = parameters['DataPreparation']['input_path']
        self.logger.info("Input reading: " + input_path)
        self.outputDF = pd.read_csv(input_path)
        return self.outputDF

    def makeTargetColumnTheFirst(self, df, target_column):
        column_names = list(df)
        target_column_name = column_names[target_column]
        temp_target_column = df[target_column_name]
        df = df.drop(target_column_name, axis=1)
        df.insert(0, target_column_name, temp_target_column)
        return df

    def dropIrrelevantColumns(self, df, dropping_col):
        for col in dropping_col:
            df = df.drop(col, axis=1)
        return df
