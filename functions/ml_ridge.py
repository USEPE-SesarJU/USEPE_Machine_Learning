# Ridge Regressor
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

def ridge_regressor(result_file, X, y):
    regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    ml_Ridge = regr.predict(X)
    ml_Ridge = pd.DataFrame(ml_Ridge, columns = [' lat', ' lon'])
    ml_Ridge['Data Type'] = 'Ridge'
    ml_Ridge.to_csv(result_file, index=False, mode='a', header=False)
    return()