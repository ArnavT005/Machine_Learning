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
def encode_input(X):
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

# convert target labels to one-hot encoding
def encode_output(Y):
	# store shape of data
	m, n = Y.shape
	# create new input array
	Y_ = np.zeros((m, 10))
	for i in range(m):
		Y_[i][Y[i][0]] = 1
	# return the encoded output
	return Y_

# convert one-hot encoding to target labels
def decode_output(Y):
	# store shape of data
	m, n = Y.shape
	# create new input array
	Y_ = np.zeros((m, 1))
	for i in range(m):
		for j in range(10):
			if Y[i][j] == 1:
				Y_[i][0] = j
				break
	# return the encoded output
	return Y_

# function to return sigmoid of a matrix
# Z: matrix
def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

# function to perform forward propagation
# X: training data
# A: list of activated layer-output vectors
# Z: list of layer-output vectors
# theta: layer parameters
def forward_propagation(X, A, Z, theta):
	# store number of layers
	num_layers = len(Z)
	# store shape of input data
	m, n = X.shape
	# initialize A[0]
	A[0] = np.column_stack((np.ones((m, 1)), X))
	# perform forward propagation and store results
	for i in range(1, num_layers):
		# determine layer-output matrix using activated output of previous layer
		Z[i] = A[i - 1] @ theta[i].T
		# determine activated output
		A[i] = np.column_stack((np.ones((m, 1)), sigmoid(Z[i])))
	# forward propagation complete, return
	return

# function to perform backward propagation
# X: training data
# Y: target labels
# A: list of activated layer-output vectors
# Z: list of layer-output vectors
# theta: layer parameters
# grad_t: parameter (theta) gradients
# grad_z: Z gradients
def backward_propagation(X, Y, A, Z, theta, grad_t, grad_z):
	# store number of layers (excluding input)
	num_layers = len(Z) - 1
	# store shape of input data
	m, n = X.shape
	# initialize grad_z[num_layers]
	gZ = sigmoid(Z[num_layers])
	grad_z[num_layers] = (-1 / m) * (Y - gZ) * gZ * (1 - gZ)
	# calculate grad_t[num_layers]
	grad_t[num_layers] = grad_z[num_layers].T @ A[num_layers - 1]
	# perform backward-propagation
	for j in reversed(range(1, num_layers)):
		gZ = sigmoid(Z[j])
		temp = grad_z[j + 1] @ theta[j + 1]
		grad_z[j] = temp[:, 1:] * gZ * (1 - gZ)
		grad_t[j] = grad_z[j].T @ A[j - 1]
	# all gradients calculated, return
	return


# function to train a neural network (fully-connected)
# X: training data
# Y: class labels
# M: mini-batch size
# hidden_layers: list of number of perceptrons in a hidden layer
# r: number of target classes
# eta: learning rate
def neural_network(X, Y, M, hidden_layers, r, eta):
	# store shape of input
	m, n = X.shape
	# total number of layers
	num_layers = len(hidden_layers) + 1
	# all layers
	layers = [n]
	layers.extend(hidden_layers)
	layers.append(r)
	# create num_layers parameters (theta)
	theta = [None]
	for i in range(1, num_layers + 1):
		# random initialization
		param = np.random.normal(size=(layers[i], layers[i - 1] + 1))
		theta.append(param)
	# define A, Z, grad_t and grad_z list
	A = [None for i in range(num_layers + 1)]
	Z = [None for i in range(num_layers + 1)]
	grad_t = [None for i in range(num_layers + 1)]
	grad_z = [None for i in range(num_layers + 1)]
	# perform forward-propagation
	forward_propagation(X, A, Z, theta)
	# determine initial cost
	Y_hat = sigmoid(Z[num_layers])
	J = np.trace((Y - Y_hat) @ (Y - Y_hat).T) / (2 * m)
	eps = 1e-6
	# print(J)
	# perform stochastic gradient descent
	while True:
		# store cost
		temp = 0
		# total of m // M mini-batches
		for b in range(m // M):
			count = 0
			if b == (m // M) - 1:
				X_mini = X[b*M:m, :]
				Y_mini = Y[b*M:m, :]
				count = m - b*M
			else:
				X_mini = X[b*M:(b+1)*M, :]
				Y_mini = Y[b*M:(b+1)*M, :]
				count = M
			# perform forward-propagation
			forward_propagation(X_mini, A, Z, theta)
			# perform back-propagation
			backward_propagation(X_mini, Y_mini, A, Z, theta, grad_t, grad_z)
			# update parameters
			for i in range(1, len(theta)):
				theta[i] -= eta * grad_t[i]
			# compute cost
			forward_propagation(X_mini, A, Z, theta)
			Y_hat_mini = sigmoid(Z[num_layers])
			temp += np.trace((Y_mini - Y_hat_mini) @ (Y_mini - Y_hat_mini).T) / (2 * count)
		# epoch complete, check convergence
		temp /= (m // M)
		# print(temp)
		if abs(J - temp) < eps:
			break
		J = temp
	# parameters learnt
	return theta


file = 'poker-hand-training-true.data'
file_test = 'poker-hand-testing.data'
data_set = parse_csv(file)
X, Y = split_data(data_set)
X_train, Y_train = encode_input(X), encode_output(Y)
eta = 0.1
hidden_layers = [25]
M = 100
r = 10
print("Training Started")
theta = neural_network(X_train, Y_train, M, hidden_layers, r, eta)
print("Network trained")
data_set = parse_csv(file_test)
X, Y = split_data(data_set)
X_test, Y_test = encode_input(X), encode_output(Y)
A = [None for i in range(3)]
Z = [None for i in range(3)]
forward_propagation(X_test, A, Z, theta)
prediction = decode_output(sigmoid(Z[2]))
Y_test = decode_output(Y_test)
accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
print("Accuracy: " + str(accuracy))