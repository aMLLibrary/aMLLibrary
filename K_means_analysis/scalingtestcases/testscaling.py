
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_data(df):
    """scale and normalized the data using standard scaler and returns the scaled data, the scaler object, the mean
    value and std of the output column for later use"""

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)

    # obtain the mean and std of the standard scaler using the formula z = (x-u)/s
    # if u is the mean and s is the std, z is the scaled version of the x

    output_scaler_mean = scaler.mean_
    output_scaler_std = (df.values[0][0] - output_scaler_mean[0])/ scaled_array[0][0]

    # return the mean and std for later use
    # return scaled_df, scaler, output_scaler_mean[0], output_scaler_std
    return scaled_df, scaler, df.mean(), df.std()

def denormalizeCol(test_labels, y_mean , y_std ):
    denorm_test_labels = (test_labels*y_std)+y_mean
    return denorm_test_labels

nums = list(range(0,1000))
orgdf = pd.DataFrame(nums)

scaled_df, scaler, df_mean, df_std = scale_data(orgdf)

inversed_scaled = scaler.inverse_transform(scaled_df)

inversed_scaled_using_mean_std = denormalizeCol(scaled_df, df_mean[0], df_std[0])


diff_df_inversed_scaled = np.sum(np.abs(orgdf - inversed_scaled))
diff_df_inversed_scaled_using_mean_std = np.sum(np.abs(orgdf-inversed_scaled_using_mean_std))

ones = list(np.ones(1000))
onesdf = pd.DataFrame(ones)

y_predicted = range(5, 1005)
y_hat = scaled_df + onesdf
inversed_y_hat_using_mean_std = denormalizeCol(y_hat, df_mean[0], df_std[0])
inversed_y_hat_using_scaler = scaler.inverse_transform(y_hat)




xpadded = padding_vector(x, 1, 45)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

x1 = pd.DataFrame(np.random.normal(1, 2, 10))
x2 = pd.DataFrame(np.random.normal(4,6,6))
scaler = StandardScaler()
x1_scaled = scaler.fit_transform(x1)
print('mean = ', scaler.mean_)
print('var = ', scaler.var_)
inversed_scaled_x1 = scaler.inverse_transform(x1_scaled)

partialscaled_x2 = scaler.partial_fit(x2)
print('mean = ', scaler.mean_)
print('var = ', scaler.var_)

inversed_scaled_x1_second = scaler.inverse_transform(x1_scaled)


"""new """

x3 = pd.DataFrame(np.random.normal(4, 6, 1000))

scaler2 = StandardScaler()
scaled_x1 = scaler2.fit_transform(pd.DataFrame(x1))

partialscaled_x2 = scaler.partial_fit(pd.DataFrame(x2))
partialscaled_x2 = scaler.transform(pd.DataFrame(x2))
inv_scaled_x1 = scaler2.inverse_transform(scaled_x1)

diff = sum(abs(scaled_x1 - inv_scaled_x1))


scaled_df = pd.DataFrame(scaled_array, index = df.index, columns = df.columns)