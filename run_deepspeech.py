import pandas as pd
from data_preparation_tf_deepspeech import DataPreparation
config_path = "./parameters_tf_deepspeech.ini"
input_path = "./tf_deepspeech_originale_0.csv"
gaps_path = ""

def main():

    dp_deepSpeech = DataPreparation()

    dp_deepSpeech.read_inputs(config_path, input_path, gaps_path)




if __name__ == '__main__':
    main()






#
# df = pd.read_csv('tf_deepspeech_originale_0.csv')
#
# column_names = list(df)
#
# rel = []
#
# df = df.loc[:, (df != df.iloc[0]).any()]
#
#
