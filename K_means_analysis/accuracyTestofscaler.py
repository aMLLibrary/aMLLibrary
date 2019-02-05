Y_test_real = pd.DataFrame.as_matrix(test_labels)
y_hat_test_real = Y_hat_test_real

y_hat_test_real_one_by_one = np.zeros(10)

for i in range(X_train.shape[0] - 1):
    sample_x = pd.DataFrame.as_matrix(test_features)[i, final_sel_idx]
    scaled_sample_x = (sample_x - train_features_scaler.mean_[final_sel_idx]) / np.sqrt(
        train_features_scaler.var_[final_sel_idx])
    scaled_y_hat = best_trained_model.predict(np.array(scaled_sample_x).reshape(1, 2))
    y_hat_sample = (scaled_y_hat * np.sqrt(train_labels_scaler.var_)) + train_labels_scaler.mean_
    y_hat_test_real_one_by_one[i] = y_hat_sample

Data = np.zeros((int(10), 8))

data_dict = {}
data_dict['Y_TR_real'] = list(Y_training_real[:-1])
data_dict['Y_TR_Scaled'] = Y_train.tolist()[:-1]
data_dict['Y_hat_TR_real'] = list(Y_hat_training_real[:-1])
data_dict['Y_hat_TR_scaled'] = list(Y_hat_training_scaled[:-1])
data_dict['Y_TE_real'] = list(Y_test_real)
data_dict['Y_TE_scaled'] = Y_test.tolist()
data_dict['Y_hat_TE_real'] = list(Y_hat_test_real)
data_dict['Y_hat_TE_scaled'] = list(Y_hat_test_scaled)

df = pd.DataFrame.from_dict(data_dict)
pd.DataFrame.to_csv(df, 'data_pred2.csv', sep='\t')



def padding_vector(x, padding_value, target_sample_num):
    padding_value = 1
    if type(x) == np.ndarray:

        if len(x) < target_sample_num:
            xtemp = list(x)
            while len(xtemp) != target_sample_num:
                print('len(x)', len(x))
                print('exp: ', len(x) != target_sample_num)
                xtemp.append(padding_value)
            xtemp = pd.DataFrame(xtemp)



    return xtemp