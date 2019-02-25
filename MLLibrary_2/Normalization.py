from SequenceDataProcessing import *
import logging


class Normalization(DataPrepration):

    def __init__(self):
        DataPrepration.__init__(self)
        self.logger = logging.getLogger(__name__)

    def process(self, inputDF):
        """Normalizes the data using StandardScaler module"""
        self.logger.info("Scaling: ")
        self.inputDF = inputDF
        self.outputDF, scaler = self.scale_data(self.inputDF)
        return self.outputDF, scaler

    def scale_data(self, df):
        """scale the dataframe"""

        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df.values)
        scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)
        return scaled_df, scaler

    def denormalize_data(self, scaled_df, scaler):
        """descale the dataframe"""

        array = scaler.inverse_transform(scaled_df)
        df = pd.DataFrame(array, index=scaled_df.index, columns=scaled_df.columns)
        return df
