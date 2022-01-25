import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
X, y = load_linnerud(return_X_y=True)
print(X)
print('y')
print(y)
regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
o = regr.predict(X[[0]])

print(o)