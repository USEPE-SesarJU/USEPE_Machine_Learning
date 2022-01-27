from data import *
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


file = 'TEST_LOGGER_logger_20220124_20-39-04.log'

file_name = (file[:-4] + '.csv')
output = data_export(file_name)


# write original route
d = data_import(file,1)
d_Lat_Lon = d[[' lat', ' lon']]
d_Lat_Lon[' Data Type'] = 'Original Route'
d_Lat_Lon.to_csv(output, index=False)

# select data columns for ML training
d = d.to_numpy()
X = d[:,[0,5,6,7,8,9,10,11,12]]
y = d[:,[3,4]]

# Ridge
regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
ml_Ridge = regr.predict(X)
ml_Ridge = pd.DataFrame(ml_Ridge, columns = [' lat', ' lon'])
ml_Ridge['Data Type'] = 'Ridge'
ml_Ridge.to_csv(output, index=False, mode='a', header=False)

# Random Forest
regr = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
ml_Random_Forest = regr.predict(X)
ml_Random_Forest = pd.DataFrame(ml_Random_Forest, columns = [' lat', ' lon'])
ml_Random_Forest['Data Type'] = 'Random Forest'
ml_Random_Forest.to_csv(output, index=False, mode='a', header=False)

# Support Vector Machine
regr = MultiOutputRegressor(SVR(kernel = "rbf", C = 1e3, gamma = 1e-8, epsilon = 0.001)).fit(X, y)
ml_SVM = regr.predict(X)
ml_SVM = pd.DataFrame(ml_SVM, columns = [' lat', ' lon'])
ml_SVM['Data Type'] = 'Support Vector Machine'
ml_SVM.to_csv(output, index=False, mode='a', header=False)
