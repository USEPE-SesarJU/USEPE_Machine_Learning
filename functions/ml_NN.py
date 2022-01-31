import pandas as pd
from matplotlib import pyplot
import tensorflow.keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error


def NN_regressor(result_file, X, y):
	regr = Sequential()
	print(regr.summary)
	regr.add(Dense(12, input_dim=9, activation='relu'))
	regr.add(Dense(9, activation='relu'))
	regr.add(Dense(2, activation='sigmoid'))
	# compile the keras regr
	regr.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit the keras regr on the dataset
	X = K.constant(X)
	y = K.constant(y)
	regr.fit(X, y, verbose=0)
	# make class predictions with the regr
	ml_NN = regr.predict(X)
	ml_NN = pd.DataFrame(ml_NN, columns = [' lat', ' lon'])
	ml_NN['Data Type'] = 'Neural Network'
	ml_NN.to_csv(result_file, index=False, mode='a', header=False)
