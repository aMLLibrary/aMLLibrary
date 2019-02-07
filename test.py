from SequenceDataProcessing import *


def main():
    dp = SequenceDataProcessing()
    dp.process()


if __name__ == '__main__':
    main()

#
# from sklearn.utils import shuffle
# import pandas as pd
# seed_v = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 1023]
# criterion_col_list = ['dataSize','nContainers']
#
#
# def make_splitting_df(df, criterion_col_list):
#     criterion_col_list = ['nContainers', 'dataSize']
#     splitting_df = pd.DataFrame(index=range(df.shape[0]))
#     splitting_df['original_index'] = df.index
#     for col in criterion_col_list:
#         splitting_df[col] = df[col]
#
#     return splitting_df
#
#
# def shuffleSamples(df, splitting_df, seed):
#     df, splitting_df = shuffle(df, splitting_df, random_state=seed)
#     return df, splitting_df
#
#
# df = pd.read_csv('P8_kmeans.csv')
#
# # drop the run column
# df = df.drop(['run'], axis=1)
#
# # drop constant columns
# df = df.loc[:, (df != df.iloc[0]).any()]
#
# s_df = make_splitting_df(df, criterion_col_list)
#
# for iter in range(len(seed_v)):
#     df, s_df = shuffle(df, s_df, random_state=seed_v[iter])
#     print(df.index[0:10])
#     print(s_df.index[0:10])

