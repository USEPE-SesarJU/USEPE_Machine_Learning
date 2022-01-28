# Support Vector Machine
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

def SVM_regressor(result_file, X, y):
    regr = MultiOutputRegressor(SVR(kernel = "rbf", C = 1e3, gamma = 1e-8, epsilon = 0.001)).fit(X, y)
    ml_SVM = regr.predict(X)
    ml_SVM = pd.DataFrame(ml_SVM, columns = [' lat', ' lon'])
    ml_SVM['Data Type'] = 'Support Vector Machine'
    ml_SVM.to_csv(result_file, index=False, mode='a', header=False)