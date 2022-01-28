# Random Forest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def RF_regressor(result_file, X, y):
    regr = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    ml_Random_Forest = regr.predict(X)
    ml_Random_Forest = pd.DataFrame(ml_Random_Forest, columns = [' lat', ' lon'])
    ml_Random_Forest['Data Type'] = 'Random Forest'
    ml_Random_Forest.to_csv(result_file, index=False, mode='a', header=False)
