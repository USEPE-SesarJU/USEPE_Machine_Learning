from data import data_import
from data import data_export
from ml_ridge import ridge_regressor
from ml_RF import RF_regressor
from ml_SVM import SVM_regressor
from mapping import on_map

# from sklearn.neural_network import MLPRegressor

file = 'TEST_LOGGER_logger_20220124_20-39-04.log'
result_file_name = (file[:-4] + '.csv')
result_file = data_export(result_file_name)

# write original route
d = data_import(file,1)
d_Lat_Lon = d[[' lat', ' lon']]
d_Lat_Lon[' Data Type'] = 'Original Route'
d_Lat_Lon.to_csv(result_file, index=False)

# Select data columns for ML training
d = d.to_numpy()
X = d[:,[0,5,6,7,8,9,10,11,12]]
y = d[:,[3,4]]

ridge_regressor(result_file, X, y)
RF_regressor(result_file, X, y)
SVM_regressor(result_file, X, y)

on_map(result_file_name)


# # Neural Network (Multi-layer Perceptron)
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# regr = MLPRegressor(max_iter=5000, activation = 'logistic', alpha=1e-20).fit(X, y)
# ml_NN = regr.predict(X)
# print(ml_NN)
# ml_NN = pd.DataFrame(ml_NN, columns = [' lat', ' lon'])
# ml_NN['Data Type'] = 'Neural Network'
# ml_NN.to_csv(output, index=False, mode='a', header=False)

