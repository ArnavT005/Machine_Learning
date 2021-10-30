from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import time
import sys

# ignore overflow warnings (for sigmoid)
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
def split_data(data_set, output=1):
	# store number of columns in data_set
	n = data_set.shape[1]
	# first n - 1 columns store the input features
	X = data_set[:, 0:(n - output)].copy()
	# last column stores the target value
	Y = data_set[:, (n - output):n].copy()
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
	gZ = A[num_layers - 1][:, 1:].copy()
	grad_z[num_layers - 1] = (-1 / m) * (Y - gZ) * gZ * (1 - gZ)
	# calculate grad_t[num_layers - 1] using grad_z[num_layers - 1]
	grad_t[num_layers - 1] = grad_z[num_layers - 1].T @ A[num_layers - 2]
	# perform backward-propagation for determining derivatives for hidden layers
	for j in reversed(range(1, num_layers - 1)):
		if code == 0:
			# use sigmoid derivative
			temp = (A[j] * (1 - A[j]) * (grad_z[j + 1] @ theta[j + 1]))[:, 1:].copy()
		else:
			# use ReLU derivative (1 at 0 assumption)
			temp = (Z[j] >= 0) * ((grad_z[j + 1] @ theta[j + 1])[:, 1:])
		# compute layer gradients
		grad_z[j] = temp.copy()
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
# max_iter: maximum number of epochs in SGD
# init: define the random initialization of weights in NN
def neural_network(X, Y, M, hidden_layers, r, eta, eps, code=0, adaptive=False, max_iter=6000, init=1):
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
	# randomly initialize theta[1..num_layers-1] (theta[0] is not used)
	for i in range(1, num_layers):
		# random initialization
		if init == 0:
			theta[i] = np.random.normal(size=(layers[i], layers[i - 1] + 1))
		else:
			theta[i] = np.random.normal(size=(layers[i], layers[i - 1] + 1)) * np.sqrt(2 / (layers[i] + layers[i - 1]))
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
	random_indices = [i for i in range(m)]
	np.random.shuffle(random_indices)
	X = X[random_indices, :].copy()
	Y = Y[random_indices, :].copy()
	# epoch count
	epoch = 0
	# consecutive counter, used for checking convergence
	counter = 0
	# perform stochastic gradient descent
	while True:
		# increment epoch count
		epoch += 1
		# break if epoch exceeds max_iter
		if epoch > max_iter:
			break
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
					theta[i] -= ((eta / math.sqrt(epoch)) * grad_t[i])
			else:
				# use constant learning rate
				for i in range(1, num_layers):
					theta[i] -= (eta * grad_t[i])
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
		if abs(J - temp) < eps:
			counter += 1
		else:
			counter = 0
		# break if difference is less than eps, 10 consecutive times
		if counter >= 10:
			break
		# update cost
		J = temp
	# parameters learnt
	return theta

# function to train a neural network using MLPClassifier
# X: training data (one-hot encoded)
# Y: class labels (one-hot encoding)
# M: mini-batch size
# hidden_layers: tuple of number of perceptrons in a hidden layer
# r: number of target classes
def MLPClassifier_neural_network(X, Y, M, hidden_layers):
	# determine training accuracy (using sigmoid)
	clf_sigmoid = MLPClassifier(hidden_layer_sizes=hidden_layers, solver='sgd', batch_size=M, learning_rate='adaptive', activation='logistic', learning_rate_init = 0.1, max_iter=1000, random_state=49).fit(X.copy(), Y.copy())
	# train classifier (using ReLU)
	clf_relu = MLPClassifier(hidden_layer_sizes=hidden_layers, solver='sgd', batch_size=M, learning_rate='adaptive', max_iter=1000, learning_rate_init=0.1).fit(X.copy(), Y.copy())
	# return the trained classifier
	return clf_relu, clf_sigmoid

# driver function
def main():
	# get file name
	file_train = sys.argv[1]
	file_test = sys.argv[2]
	# get part num
	part_num = sys.argv[3]
	if part_num == 'a':
		# load training data from file
		data_set = parse_csv(file_train)
		X, Y = split_data(data_set)
		# encode input data
		X_train, Y_train = encode_input(X), encode_output(Y)
		# load test data from file
		data_set = parse_csv(file_test)
		X, Y_test = split_data(data_set)
		# encode input data
		X_test = encode_input(X)
		# save data in files, by converting to DataFrame
		df_train = pd.DataFrame(np.column_stack((X_train, Y_train)))
		df_train.to_csv("train.csv", header=False, index=False)
		df_test = pd.DataFrame(np.column_stack((X_test, Y_test)))
		df_test.to_csv("test.csv", header=False, index=False)
		print("Encoded data saved in train.csv and test.csv respectively")
	elif part_num in ['b', 'c']:
		# load data from 'train.csv'
		data_set = parse_csv("train.csv")
		X_train, Y_train = split_data(data_set, 10)
		# decode output
		Y = decode_output(Y_train)
		# load data from 'test.csv'
		data_set = parse_csv("test.csv")
		X_test, Y_test = split_data(data_set)
		# hidden layer list
		hidden = [5, 10, 15, 20, 25]
		# learning rate
		eta = 0.1
		# stopping threshold
		eps = 1e-6
		# mini-batch size
		M = 100
		# lists to store data
		train_list = []
		test_list = []
		time_list = []
		# define A and Z list (for prediction calculation)
		Z = [None for _ in range(3)]
		A = [None for _ in range(3)]
		for size in hidden:
			print("Number of hidden layer units: " + str(size))
			start = time.time()
			theta = neural_network(X_train.copy(), Y_train.copy(), M, [size], 10, eta, eps)
			end = time.time()
			# store time taken
			time_list.append(end - start)
			print("Training time (in s): " + str(end - start))
			# determine training accuracy
			forward_propagation(X_train, A, Z, theta)
			prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((-1, 1))
			accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
			# store training accuracy
			train_list.append(100 * accuracy)
			print("Training Accuracy (in %): " + str(100 * accuracy))
			# determine test accuracy
			forward_propagation(X_test, A, Z, theta)
			prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((-1, 1))
			accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
			# store test accuracy
			test_list.append(100 * accuracy)
			print("Test Accuracy (in %): " + str(100 * accuracy))
			# form confusion matrix
			confusion_matrix = np.zeros((10, 10))
			for i in range(prediction.shape[0]):
				confusion_matrix[int(Y_test[i][0])][int(prediction[i][0])] += 1
			# save confusion matrix
			df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
			plt.figure(size)	
			sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 8})
			plt.savefig("part_c_" + str(size) + "units.png")
		# plot training and testing accuracy
		plt.figure(1)
		plt.plot(hidden, train_list, label='Training Accuracy')
		plt.plot(hidden , test_list, label='Test Accuracy')
		plt.xlabel("Number of units in hidden layer")
		plt.ylabel("Prediction Accuracy (in %)")
		plt.legend()
		plt.savefig("part_c_accuracy.png")
		# plot training time
		plt.figure(2)
		plt.plot(hidden, time_list)
		plt.xlabel("Number of units in hidden layer")
		plt.ylabel("Training time (in seconds)")
		plt.savefig("part_c_time.png")
	elif part_num == 'd':
		# load data from 'train.csv'
		data_set = parse_csv("train.csv")
		X_train, Y_train = split_data(data_set, 10)
		# decode output
		Y = decode_output(Y_train)
		# load data from 'test.csv'
		data_set = parse_csv("test.csv")
		X_test, Y_test = split_data(data_set)
		# hidden layer list
		hidden = [5, 10, 15, 20, 25]
		# learning rate
		eta = 3
		# stopping threshold
		eps = 1e-6
		# mini-batch size
		M = 100
		# lists to store data
		train_list = []
		test_list = []
		time_list = []
		# define A and Z list (for prediction calculation)
		Z = [None for _ in range(3)]
		A = [None for _ in range(3)]
		for size in hidden:
			print("Number of hidden layer units: " + str(size))
			start = time.time()
			# use adaptive learning rate
			theta = neural_network(X_train.copy(), Y_train.copy(), M, [size], 10, eta, eps, 0, True)
			end = time.time()
			# store time taken
			time_list.append(end - start)
			print("Training time (in s): " + str(end - start))
			# determine training accuracy
			forward_propagation(X_train, A, Z, theta)
			prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((-1, 1))
			accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
			# store training accuracy
			train_list.append(100 * accuracy)
			print("Training Accuracy (in %): " + str(100 * accuracy))
			# determine test accuracy
			forward_propagation(X_test, A, Z, theta)
			prediction = np.argmax(sigmoid(Z[2]), axis=1).reshape((-1, 1))
			accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
			# store test accuracy
			test_list.append(100 * accuracy)
			print("Test Accuracy (in %): " + str(100 * accuracy))
			# form confusion matrix
			confusion_matrix = np.zeros((10, 10))
			for i in range(prediction.shape[0]):
				confusion_matrix[int(Y_test[i][0])][int(prediction[i][0])] += 1
			# save confusion matrix
			df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
			plt.figure(size)	
			sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 8})
			plt.savefig("part_d_" + str(size) + "units.png")
		# plot training and testing accuracy
		plt.figure(1)
		plt.plot(hidden, train_list, label='Training Accuracy')
		plt.plot(hidden , test_list, label='Test Accuracy')
		plt.xlabel("Number of units in hidden layer")
		plt.ylabel("Prediction Accuracy (in %)")
		plt.legend()
		plt.savefig("part_d_accuracy.png")
		# plot training time
		plt.figure(2)
		plt.plot(hidden, time_list)
		plt.xlabel("Number of units in hidden layer")
		plt.ylabel("Training time (in seconds)")
		plt.savefig("part_d_time.png")
	elif part_num == 'e':
		# load data from 'train.csv'
		data_set = parse_csv("train.csv")
		X_train, Y_train = split_data(data_set, 10)
		# decode output
		Y = decode_output(Y_train)
		# load data from 'test.csv'
		data_set = parse_csv("test.csv")
		X_test, Y_test = split_data(data_set)
		# learning rate
		eta = 3
		# stopping threshold
		eps = 1e-6
		# mini-batch size
		M = 100
		# define A and Z list (for prediction calculation)
		A = [None for i in range(4)]
		Z = [None for i in range(4)]
		# try both sigmoid and relu
		for code in [0, 1]:
			if code == 0:
				print("Using Sigmoid function")
			else:
				print("Using ReLU function")
			start = time.time()
			theta = neural_network(X_train.copy(), Y_train.copy(), M, [100, 100], 10, eta, eps, code, True, 3000, code)
			end = time.time()
			print("Training time (in s): " + str(end - start))
			# determine training accuracy
			forward_propagation(X_train, A, Z, theta, code)
			prediction = np.argmax(sigmoid(Z[3]), axis=1).reshape((-1, 1))
			accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
			print("Training Accuracy (in %): " + str(100 * accuracy))
			# determine test accuracy
			forward_propagation(X_test, A, Z, theta, code)
			prediction = np.argmax(sigmoid(Z[3]), axis=1).reshape((-1, 1))
			accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
			print("Test Accuracy (in %): " + str(100 * accuracy))
			# form confusion matrix
			confusion_matrix = np.zeros((10, 10))
			for i in range(prediction.shape[0]):
				confusion_matrix[int(Y_test[i][0])][int(prediction[i][0])] += 1
			# save confusion matrix
			df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
			plt.figure(code)	
			sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 8})
			plt.savefig("part_e_" + str(code) + ".png")
	elif part_num == 'f':
		# load data from 'train.csv'
		data_set = parse_csv("train.csv")
		X_train, Y_train = split_data(data_set, 10)
		# decode output
		Y = decode_output(Y_train)
		# load data from 'test.csv'
		data_set = parse_csv("test.csv")
		X_test, Y_test = split_data(data_set)
		# mini-batch size
		M = 100
		hidden_layers = (100, 100)
		clf_relu, clf_sigmoid = MLPClassifier_neural_network(X_train, Y_train, M, hidden_layers)
		# determine training accuracy (using relu)
		prediction = decode_output(clf_relu.predict(X_train.copy()))
		accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
		print("Training Accuracy (in %) (ReLU): " + str(100 * accuracy))
		# determine test accuracy (using relu)
		prediction = decode_output(clf_relu.predict(X_test.copy()))
		accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
		print("Test Accuracy (in %) (ReLU): " + str(100 * accuracy))
		# form confusion matrix
		confusion_matrix = np.zeros((10, 10))
		for i in range(prediction.shape[0]):
			confusion_matrix[int(Y_test[i][0])][int(prediction[i][0])] += 1
		# save confusion matrix
		df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
		plt.figure(1)	
		sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 8})
		plt.savefig("part_f_relu.png")
		# determine training accuracy (using sigmoid)
		prediction = decode_output(clf_sigmoid.predict(X_train.copy()))
		accuracy = np.sum((prediction == Y)*1) / X_train.shape[0]
		print("Training Accuracy (in %) (Sigmoid): " + str(100 * accuracy))
		# determine test accuracy (using sigmoid)
		prediction = decode_output(clf_sigmoid.predict(X_test.copy()))
		accuracy = np.sum((prediction == Y_test)*1) / X_test.shape[0]
		print("Test Accuracy (in %) (Sigmoid): " + str(100 * accuracy))
		# form confusion matrix
		confusion_matrix = np.zeros((10, 10))
		for i in range(prediction.shape[0]):
			confusion_matrix[int(Y_test[i][0])][int(prediction[i][0])] += 1
		# save confusion matrix
		df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
		plt.figure(2)	
		sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 8})
		plt.savefig("part_f_sigmoid.png")
		
main()