from data_importer import *
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

file = 'TEST_LOGGER_logger_20220124_20-39-04.log'
d = data(file)
d_Lat_Lon = d[[' lat', ' lon']]
d_Lat_Lon[' Data Type'] = 'Original Data'
d_Lat_Lon.to_csv('TEST_LOGGER_logger_20220124_20-39-04.csv',index=False)


d = d.to_numpy()
X = d[:,[0,5,6,7,8,9,10,11,12]]
y = d[:,[3,4]]

regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)

output = regr.predict(X)
output = pd.DataFrame(output, columns = [' lat', ' lon'])
output['Data Type'] = 'Ridge ML'
output.to_csv('TEST_LOGGER_logger_20220124_20-39-04.csv',index=False, mode='a', header=False)