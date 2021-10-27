from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import math
import time
np.warnings.filterwarnings('ignore', 'overflow')
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
	# store number of columns in data_set
	n = data_set.shape[1]
	# first n - 1 columns store the input features
	X = data_set[:, 0:(n - 1)]
	# last column stores the target value
	Y = data_set[:, (n - 1):n]
	# return the split data set
	return X, Y

# convert input data (X) to one-hot encoding
def encode_input(X):
	# store number of rows in X
	m = X.shape[0]
	# create new input array (85 features)
	X_ = np.zeros((m, 85))
	for i in range(m):
		for j in [0, 2, 4, 6, 8]:
			X_[i][(j // 2) * 17 + X[i][j] - 1] = 1
			X_[i][(j // 2) * 17 + X[i][j + 1] + 3] = 1
	# return the encoded input
	return X_

# convert target labels to one-hot encoding
def encode_output(Y):
	# store number of rows in Y
	m = Y.shape[0]
	# create new input array (10 classes)
	Y_ = np.zeros((m, 10))
	for i in range(m):
		Y_[i][Y[i][0]] = 1
	# return the encoded output
	return Y_

# convert one-hot encoding to target labels
def decode_output(Y):
	# store number of rows in Y
	m = Y.shape[0]
	# create new input array (single label)
	Y_ = np.zeros((m, 1))
	for i in range(m):
		for j in range(10):
			if Y[i][j] == 1:
				Y_[i][0] = j
				break
	# return the decoded output
	return Y_

# function to return sigmoid of a matrix
# Z: matrix
def sigmoid(Z):
	return 1 / (1 + np.exp(-Z))

# function to return ReLU of a matrix
# Z: matrix
def ReLU(Z):
	return np.maximum(0, Z)

# function to perform forward propagation
# X: training data
# A: list of layer activated outputs (with intercept term)
# Z: list of layer outputs
# theta: layer parameters
# code: code of the activation function used for hidden-layers (sigmoid is used for output layer)
#		0 for sigmoid (default), 1 for ReLU
def forward_propagation(X, A, Z, theta, code=0):
	# store number of layers
	num_layers = len(Z)
	# store number of rows in input data
	m = X.shape[0]
	# initialize A[0], include intercept term
	A[0] = np.column_stack((np.ones((m, 1)), X))
	# perform forward propagation and store results
	for i in range(1, num_layers):
		# determine layer-output matrix using activated output of previous layer
		Z[i] = A[i - 1] @ theta[i].T
		# determine activated output, include intercept term
		if i == num_layers - 1:
			# output layer (use sigmoid)
			A[i] = np.column_stack((np.ones((m, 1)), sigmoid(Z[i])))
		else:
			# hidden layer (use code value)
			if code == 0:
				# use sigmoid
				A[i] = np.column_stack((np.ones((m, 1)), sigmoid(Z[i])))
			else:
				# use ReLU
				A[i] = np.column_stack((np.ones((m, 1)), ReLU(Z[i])))
	# forward propagation complete, return
	return

# function to perform backward propagation
# X: training data
# Y: target labels
# A: list of layer activated outputs (with intercept term)
# Z: list of layer outputs
# theta: layer parameters
# grad_t: theta gradients
# grad_z: Z gradients
# code: code of the activation function used for hidden-layers (sigmoid is used for output layer)
#		0 for sigmoid (default), 1 for ReLU
def backward_propagation(X, Y, A, Z, theta, grad_t, grad_z, code=0):
	# store number of layers
	num_layers = len(Z)
	# store number of rows in input data
	m = X.shape[0]
	# initialize grad_z[num_layers - 1]
	gZ = A[num_layers - 1][:, 1:]
	grad_z[num_layers - 1] = (-1 / m) * (Y - gZ) * gZ * (1 - gZ)
	# calculate grad_t[num_layers - 1] using grad_z[num_layers - 1]
	grad_t[num_layers - 1] = grad_z[num_layers - 1].T @ A[num_layers - 2]
	# perform backward-propagation for determining derivatives for hidden layers
	for j in reversed(range(1, num_layers - 1)):
		if code == 0:
			# use sigmoid derivative
			temp = A[j] * (1 - A[j]) * (grad_z[j + 1] @ theta[j + 1])
		else:
			# use ReLU derivative (0 at 0 assumption)
			temp = (A[j] > 0) * (grad_z[j + 1] @ theta[j + 1])
		# compute layer gradients
		grad_z[j] = temp[:, 1:]
		grad_t[j] = grad_z[j].T @ A[j - 1]
	# all gradients calculated, return
	return


# function to train a neural network (fully-connected)
# X: training data (one-hot encoded)
# Y: class labels (one-hot encoded)
# M: mini-batch size
# hidden_layers: list of number of perceptrons in a hidden layer
# r: number of target classes
# eta: learning rate
# eps: stopping threshold
# code: code of the activation function used for hidden-layers (sigmoid is used for output layer)
#		0 for sigmoid (default), 1 for ReLU 
# adaptive: flag to indicate whether adaptive learning rate needs to be used
def neural_network(X, Y, M, hidden_layers, r, eta, eps, code=0, adaptive=False):
	# store shape of input
	m, n = X.shape
	# total number of layers (include input/output layer)
	num_layers = len(hidden_layers) + 2
	# number of units per layer
	# n features/units in input layer
	layers = [n]
	layers.extend(hidden_layers)
	# r units/classes in output layer
	layers.append(r)
	# create num_layers parameters (theta)
	theta = [None for i in range(num_layers)]
	momentum = [None for i in range(num_layers)]
	# randomly initialize theta[1..num_layers-1] (theta[0] is not used)
	for i in range(1, num_layers):
		# random initialization
		theta[i] = np.random.normal(size=(layers[i], layers[i - 1] + 1))
		theta[i][:, 0:1] = np.zeros((layers[i], 1))
		momentum[i] = np.zeros((layers[i], layers[i - 1] + 1))
	# define A, Z, grad_t and grad_z list
	# A: activated output of a layer
	A = [None for i in range(num_layers)]
	# Z: linear output of a layer
	Z = [None for i in range(num_layers)]
	# grad_t: gradient of mini-batch cost function w.r.t theta
	grad_t = [None for i in range(num_layers)]
	# grad_z: gradient of mini-batch cost function w.r.t Z
	grad_z = [None for i in range(num_layers)]
	# determine initial cost
	# perform forward-propagation to generate output
	forward_propagation(X, A, Z, theta, code)
	# compute activated output of final layer (leave out intercept term)
	Y_hat = A[num_layers - 1][:, 1:]
	# compute element-wise difference
	diff = Y - Y_hat
	# cost is equal to 0.5 * (average of square of element-wise difference)
	J = np.sum(diff ** 2) / (2 * m)
	# randomly shuffle the training data
	# epoch count
	epoch = 0
	# perform stochastic gradient descent
	while True:
		# random_indices = [i for i in range(m)]
		# np.random.shuffle(random_indices)
		# X = X[random_indices, :]
		# Y = Y[random_indices, :]
		# increment epoch count
		epoch += 1
		# store cost
		temp = 0
		# total of ceil(m / M) mini-batches (last batch may be shorter)
		for b in range(math.ceil(m / M)):
			# count number of samples in mini-batch
			sample_count = 0
			# select mini-batch
			if b == math.ceil(m / M) - 1:
				X_mini = X[(b * M):m, :]
				Y_mini = Y[(b * M):m, :]
				sample_count = m - b * M
			else:
				X_mini = X[(b * M):((b + 1) * M), :]
				Y_mini = Y[(b * M):((b + 1) * M), :]
				sample_count = M
			# perform forward-propagation
			forward_propagation(X_mini, A, Z, theta, code)
			# perform backward-propagation
			backward_propagation(X_mini, Y_mini, A, Z, theta, grad_t, grad_z, code)
			# update parameters
			if adaptive:
				# use adaptive learning rate
				for i in range(1, num_layers):
					momentum[i] = 0.9 * momentum[i] + eta * grad_t[i]
					theta[i] -= momentum[i]
					# theta[i] -= ((eta / math.sqrt(epoch)) * grad_t[i])
			else:
				# use constant learning rate
				for i in range(1, num_layers):
					momentum[i] = 0.9 * momentum[i] + eta * grad_t[i]
					theta[i] -= momentum[i]
					# theta[i] -= (eta * grad_t[i])
			# compute updated cost, perform forward propagation
			forward_propagation(X_mini, A, Z, theta, code)
			# compute activated output of final layer (leave out intercept term)
			Y_hat_mini = A[num_layers - 1][:, 1:]
			# compute element-wise difference
			diff_mini = Y_mini - Y_hat_mini
			# cost is equal to 0.5 * (average of square of element-wise difference)
			# add cost to temp
			temp += np.sum(diff_mini ** 2) / (2 * sample_count)
		# epoch complete, determine average cost
		temp /= (m // M)
		# check convergence
		# print(eps)
		if abs(J - temp) < eps:
			print("Hello")
			break
		# update cost
		J = temp
		if epoch % 100 == 0:
			print("Epoch: " + str(epoch) + ", Cost: " + str(J) + ", Units: " + str(hidden_layers[0]))
	# parameters learnt
	return theta


# function to train a neural network using MLPClassifier
# X: training data (one-hot encoded)
# Y: class labels (one-hot encoded)
# M: mini-batch size
# hidden_layers: tuple of number of perceptrons in a hidden layer
# r: number of target classes
def MLPClassifier_neural_network(X, Y, M, hidden_layers):
	# train classifier (using ReLU)
	clf_relu = MLPClassifier(hidden_layer_sizes=hidden_layers, solver='sgd', batch_size=M, learning_rate='adaptive', max_iter=1000).fit(X, Y)
	clf_sigmoid = MLPClassifier(hidden_layer_sizes=hidden_layers, solver='sgd', batch_size=M, learning_rate='adaptive', activation='logistic', max_iter=1000).fit(X, Y)
	# return the trained classifier
	return clf_relu, clf_sigmoid


file_train = 'poker-hand-training-true.data'
file_test = 'poker-hand-testing.data'
data_set = parse_csv(file_train)
X, Y = split_data(data_set)
X_train, Y_train = encode_input(X), encode_output(Y)
data_set = parse_csv(file_test)
X, Y = split_data(data_set)
X_test, Y_test = encode_input(X), Y
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
# eta = 0.1
# M = 100
# r = 10
# eps = 1e-6
# A = [None for i in range(3)]
# Z = [None for i in range(3)]
# file = open("data.txt", "w")
# hidden_layer_sizes = [5, 10, 15, 20, 25]
# for size in hidden_layer_sizes:
# 	file.write("No. of units: " + str(size) + "\n")
# 	start = time.time()
# 	theta = neural_network(X_train, Y_train, M, [size], r, eta, eps)
# 	end = time.time()
# 	file.write("Training time: " + str(end - start) + "\n")
# 	# determine training accuracy
# 	forward_propagation(X_train, A, Z, theta)
# 	prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((X_train.shape[0], 1))
# 	Y = decode_output(Y_train)
# 	accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
# 	file.write("Training accuracy: " + str(100 * accuracy) + "\n")
# 	# determine test accuracy
# 	forward_propagation(X_test, A, Z, theta)
# 	prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((X_test.shape[0], 1))
# 	accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
# 	file.write("Test accuracy: " + str(100 * accuracy) + "\n")
# 	confusion_matrix = np.zeros((10, 10))
# 	for i in range(prediction.shape[0]):
# 		confusion_matrix[Y_test[i][0]][prediction[i][0]] += 1
# 	for i in range(10):
# 		file.write(str(confusion_matrix[i]) + "\n")
# 	file.write("_________________________________________________________________\n")

# eta = 10
# M = 100
# r = 10
# eps = 1e-6
# A = [None for i in range(3)]
# Z = [None for i in range(3)]
# file = open("data.txt", "w")
# hidden_layer_sizes = [5, 10, 15, 20, 25]
# for size in hidden_layer_sizes:
# 	file.write("No. of units: " + str(size) + "\n")
# 	start = time.time()
# 	theta = neural_network(X_train, Y_train, M, [25], r, eta, eps, 0, True)
# 	end = time.time()
# 	file.write("Training time: " + str(end - start) + "\n")
# 	# determine training accuracy
# 	forward_propagation(X_train, A, Z, theta)
# 	prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((X_train.shape[0], 1))
# 	Y = decode_output(Y_train)
# 	accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
# 	file.write("Training accuracy: " + str(100 * accuracy) + "\n")
# 	# determine test accuracy
# 	forward_propagation(X_test, A, Z, theta)
# 	prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((X_test.shape[0], 1))
# 	accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
# 	file.write("Test accuracy: " + str(100 * accuracy) + "\n")
# 	confusion_matrix = np.zeros((10, 10))
# 	for i in range(prediction.shape[0]):
# 		confusion_matrix[Y_test[i][0]][prediction[i][0]] += 1
# 	for i in range(10):
# 		file.write(str(confusion_matrix[i]) + "\n")
# 	file.write("_________________________________________________________________\n")

eta = 0.1
M = 100
r = 10
eps = 1e-6
A = [None for i in range(4)]
Z = [None for i in range(4)]
file = open("data.txt", "w")
for code in range(2):
	file.write("Activation code: " + str(1 - code) + "\n")
	start = time.time()
	theta = neural_network(X_train.copy(), Y_train.copy(), M, [100, 100], r, eta, eps, 0, True)
	end = time.time()
	file.write("Training time: " + str(end - start) + "\n")
	# determine training accuracy
	forward_propagation(X_train, A, Z, theta, 1 - code)
	prediction = np.argmax(sigmoid(Z[3]), axis=1).reshape((X_train.shape[0], 1))
	Y = decode_output(Y_train)
	accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
	print(accuracy)
	file.write("Training accuracy: " + str(100 * accuracy) + "\n")
	# determine test accuracy
	forward_propagation(X_test, A, Z, theta, 1 - code)
	prediction = np.argmax(sigmoid(Z[3]), axis=1).reshape((X_test.shape[0], 1))
	accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
	print(accuracy)
	file.write("Test accuracy: " + str(100 * accuracy) + "\n")
	confusion_matrix = np.zeros((10, 10))
	for i in range(prediction.shape[0]):
		confusion_matrix[Y_test[i][0]][prediction[i][0]] += 1
	for i in range(10):
		file.write(str(confusion_matrix[i]) + "\n")
	file.write("_________________________________________________________________\n")

# # print("Training Started")
# # theta = neural_network(X_train, Y_train, M, hidden_layers, r, eta, eps, 1)
# # print("Network trained")
# # data_set = parse_csv(file_test)
# # X, Y = split_data(data_set)
# X_test, Y_test = encode_input(X), Y


# clf = MLPClassifier_neural_network(X_train, Y_train, M, (100, 100))
# prediction = clf.predict(X_test)
# prediction = prediction.reshape((-1, 1))
# accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
# print("Accuracy: " + str(accuracy))
