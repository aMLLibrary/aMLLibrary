import pandas as pd
import numpy.matlib as matlib
import numpy as np
import scipy.stats as stats


df = pd.read_csv('dumydata2.csv')
df = df.drop(['run'], axis=1)
target_column_idx = 0
Confidence_level = 0.8

# Replace 1 if independent
# Replace 0 if dependent

def Screening(df, target_column_idx, Confidence_level):

    Y = df.iloc[:,target_column_idx]
    Y = pd.DataFrame.as_matrix(Y)

    output_name = df.columns.values[target_column_idx]
    X = df.loc[:,df.columns != output_name]
    X = pd.DataFrame.as_matrix(X)

    independence_list = []
    if X.shape[0] == Y.shape[0]:
        for c in range(len(X.shape[1])):
            independence_list.append(independentTest(X[:, c], Y, Confidence_level))

    irrelevant_col = [col for col, e in enumerate(independence_list) if e == 1]
    return irrelevant_col



def independentTest(x,y, Confidence_level):

    N = x.shape[0]

    xx = np.tile(x, [1,N])
    xxp = xx.transpose()
    diff_x = xx - xxp
    norm_x = np.sqrt(diff_x**2)

    temp1_x = norm_x.mean(0).reshape((1, N))
    temp2_x = np.tile(temp1_x, [N, 1])

    temp3_x = norm_x.mean(1).reshape((N, 1))
    temp4_x = np.tile(temp3_x, [1, N])

    X_MATRIX = norm_x - temp2_x - temp4_x + norm_x.mean()


    yy = np.tile(y, [1,N])
    yyp = yy.transpose()
    diff_y = yy - yyp
    norm_y = np.sqrt(diff_y**2)

    temp1_y = norm_y.mean(0).reshape((1, N))
    temp2_y = np.tile(temp1_y, [N, 1])

    temp3_y = norm_y.mean(1).reshape((N, 1))
    temp4_y = np.tile(temp3_y, [1, N])

    Y_MATRIX = norm_y - temp2_y - temp4_y + norm_y.mean()

    XYPair = np.multiply(X_MATRIX, Y_MATRIX)

    Vxy = XYPair.mean()

    alpha = 1 - Confidence_level
    T = N * Vxy / (norm_x.mean() * norm_y.mean())
    P = T <= (stats.norm.ppf(1 - alpha / 2)**2)

    return 1 if P else 0



indtestdf = pd.DataFrame(data=np.array([[1,7,10], [2,5,20], [3,4,30], [4,60,40], [5,12,50], [6,1,60]]), index=range(0,6), columns=['x1','x2','y'])

p = Screening(indtestdf, 2, 0.8)
x = np.array([[1],[2],[3],[4],[5],[6]])
y = np.array([[7],[5],[4],[60],[12],[1]])

