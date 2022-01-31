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
print(d)
d_Lat_Lon = d[[' lat', ' lon']]
d_Lat_Lon[' Data Type'] = 'Original Route'
d_Lat_Lon.to_csv(result_file, index=False)

# Select data columns for ML training
d = d.to_numpy()
X = d[:,[0,5,6,7,8,9,10,11,12]]
y = d[:,[3,4]]

#ridge_regressor(result_file, X, y)    # This is working but not needed for D4.2
RF_regressor(result_file, X, y)
SVM_regressor(result_file, X, y)

on_map(result_file_name)

