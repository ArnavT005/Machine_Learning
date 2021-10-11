import pandas as pd
import numpy as np

# parse training data (CSV) and generate training/test data
# parameter file stores the name of the file
def parse_csv(file):
	# read csv file (no header)
	df = pd.read_csv(file, header=None)
	# convert to a numpy array
	data_set = df.to_numpy()
	# return data set
	return data_set

# split the data_set into input features (input) and target value (output)
def split_data(data_set):
	# store the shape of data set
	m, n = data_set.shape
	# first n - 1 columns store the input features
	X = data_set[:, 0:(n - 1)]
	# last column stores the target value
	Y = data_set[:, (n - 1):n]
	# return the split data set
	return X, Y

# convert input data (X) to one-hot encoding
def encode(X):
	# store shape of data
	m, n = X.shape
	# create new input array
	X_ = np.zeros((m, 85))
	for i in range(m):
		for j in [0, 2, 4, 6, 8]:
			X_[i][(j // 2) * 17 + X[i][j] - 1] = 1
			X_[i][(j // 2) * 17 + X[i][j + 1] + 3] = 1
	# return the encoded input
	return X_


file = 'poker-hand-training-true.data'
file_test = 'poker-hand-testing.data'
data_set = parse_csv(file_test)
X, Y_train = split_data(data_set)
X_train = encode(X)
print(X_train.shape, Y_train.shape)