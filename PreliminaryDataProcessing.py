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
        input_path = parameters['DataPreparation']['input_path']
        self.logger.info("Input reading: " + input_path)
        self.outputDF = pd.read_csv(input_path)

        # drop the run column
        dropping_col = parameters['DataPreparation']['irrelevant_column_name']
        self.outputDF = self.outputDF.drop(dropping_col, axis=1)

        # manually decrease the number of target column since we dropped the first columns, this maybe changed in
        # different input files
        parameters['DataPreparation']['target_column'] -= 1

        # drop constant columns
        self.outputDF = self.outputDF.loc[:, (self.outputDF != self.outputDF.iloc[0]).any()]
        return self.outputDF

