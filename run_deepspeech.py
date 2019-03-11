from data_preparation_tf_deepspeech import DataPreparation
import feature_extender
from SequenceDataProcessing import *
from FeatureSelection import FeatureSelection

config_path = "./parameters_tf_deepspeech.ini"
input_path = "./tf_deepspeech_originale_0.csv"
gaps_path = ""

def main():

    sdp = SequenceDataProcessing()
    fs = FeatureSelection.FeatureSelection()

    dp_deepSpeech = DataPreparation()
    dp_deepSpeech.read_inputs(config_path, input_path, gaps_path)
    print(dp_deepSpeech.df.columns.values)
    print(dp_deepSpeech.df.shape)
    dp_deepSpeech.extend_generate(config_path)
    print(dp_deepSpeech.df.shape)

    run_info = {}
    train_features, train_labels, test_features, test_labels, feature_names, scaler, data_conf = dp_deepSpeech.split_data(1234)
    y_pred_train, y_pred_test = fs.process(train_features, train_labels, test_features, test_labels, sdp.parameters, run_info)

if __name__ == '__main__':
    main()



# df = pd.read_csv('tf_deepspeech_originale_0.csv')
#
# column_names = list(df)
#
# rel = []
#
# df = df.loc[:, (df != df.iloc[0]).any()]