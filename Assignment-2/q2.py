from libsvm.svmutil import *
from cvxopt import solvers
from cvxopt import matrix
import pandas as pd
import numpy as np


## PRE-PROCESSING FUNCTIONS ##

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
# also scales down the input feature values to range [0, 1]
def split_data(data_set):
	# store the shape of data set
	m, n = data_set.shape
	# first n - 1 columns store the input features
	X = data_set[:, 0:(n - 1)]
	# scale data to range [0, 1]
	X = X / 255.0
	# last column stores the target value
	Y = data_set[:, (n - 1):n]
	# return the split data set
	return X, Y

# filter classes 'class1' and 'class2' from from data_set (X and Y)
def filter_data(X, Y, class1, class2):
	# store shape of data set (X)
	m, n = X.shape
	# store row numbers of required data
	row_indices = []
	# go through entire data set for filtering
	for i in range(m):
		if Y[i][0] == class1 or Y[i][0] == class2:
			# this is desired data
			row_indices.append(i)
	# select and return appropriate subset of data_set (X and Y)
	return X[row_indices, :], Y[row_indices, :]


## MODEL LEARNING FUNCTIONS (CVXOPT) ##

# function to train a (linear kernel) SVM binary classifier (using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
# C: regularization hyperparameter
def svm_train_binary_linear_CVXOPT(X, Y, class_num, C):
	# store shape of training data
	m, n = X.shape
	# update class labels to -1 and 1
	for i in range(m):
		if Y[i][0] == class_num:
			Y[i][0] = -1
		else:
			# other class (change to 1)
			Y[i][0] = 1
	# we can multiply the dual objective by -1 to convert it into minimization problem
	# express dual objective as (1/2) x.T @ P @ x + q.T @ x
	# subject to constraints, Gx <= h and Ax = b
	# determining matrix A = Y.T (transpose)
	A = matrix(Y.T, tc='d')
	# determining matrix b
	b = matrix(np.array([0]), tc='d')
	# determining matrix G
	temp_1 = -np.eye(m)
	temp_2 = np.eye(m)
	# row_stack temp_1 and temp_2
	temp = np.row_stack((temp_1, temp_2))
	G = matrix(temp, tc='d')
	# determining matrix h (C)
	temp_1 = np.zeros((m, 1))
	temp_2 = C * np.ones((m, 1))
	# row_stack temp_1 and temp_2
	temp = np.row_stack((temp_1, temp_2))
	h = matrix(temp, tc='d')
	# determine matrix q (column of ones)
	temp_1 = -np.ones((m, 1))
	q = matrix(temp_1, tc='d')
	# determine matrix P
	# X_y is the matrix formed by multiplying Y.T and X.T element-wise (with broadcasting)
	X_y = X.T * Y.T
	temp = X_y.T @ X_y
	P = matrix(temp, tc='d')
	# solve the dual optimization (minimization) problem
	solution = solvers.qp(P, q, G, h, A, b)
	# extract optimal value of alpha (column vector)
	alpha = np.array(solution['x'])
	# determine 'w' parameter = X.T @ (alpha * Y) (* denotes element-wise multiplication)
	w = X.T @ (alpha * Y)
	# determine support vectors (alpha > 1e-4)
	support_vectors_indices = []
	for i in range(m):
		# non-zero coefficient
		if alpha[i][0] > 1e-4:
			support_vectors_indices.append(i)
	# slice alpha, X and Y to get support vectors
	support_vectors_alpha = alpha[support_vectors_indices, :]
	support_vectors_X = X[support_vectors_indices, :]
	support_vectors_Y = Y[support_vectors_indices, :]
	# determine intercept term, as all support_alpha < 1, therefore, noise terms are all zero
	# hence, b = support_vector_Y - w.T @ support_vector_X (constraint becomes equality)
	# determining all such b's using all such vectors
	b = support_vectors_Y - support_vectors_X @ w
	# return parameters w and b
	return support_vectors_indices, support_vectors_alpha, support_vectors_X, support_vectors_Y, w, b

# function to train a (gaussian kernel) SVM binary classifier (using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
# C: regularization hyperparameter
# gamma: kernel parameter
def svm_train_binary_gaussian_CVXOPT(X, Y, class_num, C, gamma):
	# store shape of training data
	m, n = X.shape
	# update class labels to -1 and 1
	for i in range(m):
		if Y[i][0] == class_num:
			Y[i][0] = -1
		else:
			# other class (change to 1)
			Y[i][0] = 1
	# we can multiply the dual objective by -1 to convert it into minimization problem
	# express dual objective as (1/2) x.T @ P @ x + q.T @ x
	# subject to constraints, Gx <= h and Ax = b
	# all matrices except P are same as that in case of linear kernel
	# determining matrix A = Y.T (transpose)
	A = matrix(Y.T, tc='d')
	# determining matrix b
	b = matrix(np.array([0]), tc='d')
	# determining matrix G
	temp_1 = -np.eye(m)
	temp_2 = np.eye(m)
	# row_stack temp_1 and temp_2
	temp = np.row_stack((temp_1, temp_2))
	G = matrix(temp, tc='d')
	# determining matrix h (C)
	temp_1 = np.zeros((m, 1))
	temp_2 = C * np.ones((m, 1))
	# row_stack temp_1 and temp_2
	temp = np.row_stack((temp_1, temp_2))
	h = matrix(temp, tc='d')
	# determine matrix q (column of ones)
	temp_1 = -np.ones((m, 1))
	q = matrix(temp_1, tc='d')
	# determine matrix P (P_ij = y_i*y_j*exp(-gamma*||x_i-x_j||^2))
	# creating 2D matrix of pairwise distances (using (x1 - x2).T @ (x1 - x2) = x1.T @ x1 + x2.T @ x2 - 2 * x1.T @ x2)
	temp_1 = X @ X.T
	temp_2 = np.diag(np.diag(temp_1))
	temp_3 = np.ones((m, m))
	temp_4 = temp_2 @ temp_3 + temp_3 @ temp_2
	# creating matrix of pairwise target-products (used in kernel matrix)
	temp_5 = Y @ Y.T
	# create P = temp_5 * np.exp(-0.05(temp_4 - 2 * temp_1))
	P = temp_5 * np.exp(-gamma * (temp_4 - 2 * temp_1))
	# convert to matrix
	P = matrix(P, tc='d')
	# solve the dual optimization (minimization) problem
	solution = solvers.qp(P, q, G, h, A, b)
	# extract optimal value of alpha (column vector)
	alpha = np.array(solution['x'])
	# determine support vectors (alpha > 1e-4)
	support_vectors_indices = []
	for i in range(m):
		# non-zero coefficient
		if alpha[i][0] > 1e-4:
			support_vectors_indices.append(i)
	# slice alpha, X and Y to get support vectors
	support_vectors_alpha = alpha[support_vectors_indices, :]
	support_vectors_X = X[support_vectors_indices, :]
	support_vectors_Y = Y[support_vectors_indices, :]
	# return support vectors
	return support_vectors_indices, support_vectors_alpha, support_vectors_X, support_vectors_Y

# function to train a (gaussian kernel) SVM multi-class classifier (using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# k: number of classes
# C: regularization hyperparameter
# gamma: kernel parameter
def svm_train_multi_gaussian_CVXOPT(X, Y, k, C, gamma):
	# create alpha_s, X_s and Y_s to store support vector information
	alpha_s = [[None for j in range(k)] for i in range(k)]
	X_s = [[None for j in range(k)] for i in range(k)]
	Y_s = [[None for j in range(k)] for i in range(k)]
	# go through all possible pairs and train models (k (k - 1) / 2)
	for i in range(k):
		for j in range(i + 1, k):
			# filter training data and retrieve subset (class = i + 1 and j + 1)
			X_subset, Y_subset = filter_data(X, Y, i + 1, j + 1)
			# train model, store in alpha_s, X_s and Y_s (higher class is set to one)
			alpha_s[i][j], X_s[i][j], Y_s[i][j] = svm_train_binary_gaussian_CVXOPT(X_subset, Y_subset, i + 1, C, gamma)
	# all models trained
	# return parameters
	return alpha_s, X_s, Y_s


## MODEL TESTING FUNCTIONS (CVXOPT) ##

# function to test SVM binary classification (linear kernel)
# w and b are provided explicitly
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class1: class number that needs to be treated as -1, other class (class2) will be 1
# w, b are the model parameters
def svm_test_binary_linear_CVXOPT(X, Y, class1, class2, w, b):
	# store the shape of test data
	m, n = X.shape
	# initialise accuracy count to 0
	accuracy_count = 0
	# go through all examples and make predictions
	for i in range(m):
		if X[i, :] @ w + b >= 0:
			# predict class2
			if Y[i] == class2:
				accuracy_count += 1
		else:
			# predict class1
			if Y[i] == class1:
				accuracy_count += 1
	# return prediction accuracy
	return (accuracy_count * 100) / m

# function to test SVM binary classification (gaussian kernel)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class1: class number that needs to be treated as -1, other class (class2) will be 1
# alpha_s, X_s and Y_s are support vectors coefficients, input features and target values respectively
# gamma: kernel parameter
def svm_test_binary_gaussian_CVXOPT(X, Y, class1, class2, alpha_s, X_s, Y_s, gamma):
	# store the shape of test data
	m, n = X.shape
	# store number of support vectors
	num_vectors = alpha_s.shape[0]
	# determine overall matrix (all examples)
	X_total = np.row_stack((X_s, X))
	# determine kernel matrix of total matrix
	temp_1 = X_total @ X_total.T
	temp_2 = np.diag(np.diag(temp_1))
	temp_3 = np.ones((num_vectors + m, num_vectors + m))
	temp_4 = temp_2 @ temp_3 + temp_3 @ temp_2
	kernel = np.exp(-gamma * (temp_4 - 2 * temp_1))
	# determine intercept term, b = y_s - summation(alpha_j * y_j * K(x_s, x_j)) over j
	temp = np.sum(alpha_s * Y_s * kernel[0:num_vectors, 0])
	# for j in range(num_vectors):
	# 	temp += alpha_s[j][0] * Y_s[j][0] * np.exp(-0.05 * ((X_s[0, :] - X_s[j, :]) @ (X_s[0, :] - X_s[j, :]).T))
	b = Y_s[0][0] - temp
	# initialise accuracy count to 0
	accuracy_count = 0
	# go through all examples and make predictions
	for i in range(m):
		# determine w.T @ x = summation(alpha_j * y_j * K(x_j, x)) over j
		temp = np.sum(alpha_s * Y_s * kernel[0:num_vectors, num_vectors + i])
		# for j in range(num_vectors):
		# 	temp += alpha_s[j][0] * Y_s[j][0] * np.exp(-0.05 * ((X[i, :] - X_s[j, :]) @ (X[i, :] - X_s[j, :]).T))
		if temp + b >= 0:
			# predict class2
			if Y[i] == class2:
				accuracy_count += 1
		else:
			# predict class1
			if Y[i] == class1:
				accuracy_count += 1
	# return prediction accuracy
	return (accuracy_count * 100) / m

# function to test a multi-class SVM classifier (trained using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# alpha_s, X_s and Y_s are support vectors data
def svm_test_multi_CVXOPT(X, Y, alpha_s, X_s, Y_s):
	# store number of examples
	m = X.shape[0]
	# store number of classes 
	k = len(alpha_s)
	# initialise accuracy count to 0
	accuracy_count = 0
	# go through all examples and make predictions
	for i in range(m):
		# initialise vote array
		vote = [0 for j in range(k)]
		# go through all models and vote
		for j in range(k):
			for l in range(j + 1, k):
				# determine number of support vectors
				num_vectors = alpha_s[j][l].shape[0]
				# determine intercept term, b = y_s - summation(alpha_j * y_j * K(x_s, x_j)) over j
				temp = 0
				for s in range(num_vectors):
					temp += alpha_s[j][l][s][0] * Y_s[j][l][s][0] * np.exp(-0.05 * ((X_s[j][l][0, :] - X_s[j][l][s, :]) @ (X_s[j][l][0, :] - X_s[j][l][s, :]).T))
				b = Y_s[j][l][0][0] - temp
				# determine w.T @ x = summation(alpha_j * y_j * K(x_j, x)) over j
				temp = 0
				for s in range(num_vectors):
					temp += alpha_s[j][l][s][0] * Y_s[j][l][s][0] * np.exp(-0.05 * ((X[i, :] - X_s[j][l][s, :]) @ (X[i, :] - X_s[j][l][s, :]).T))
				if temp + b >= 0:
					# predict 1
					prediction = 1
				else:
					# predict -1
					prediction = -1
				# increase vote count
				if prediction == 1:
					vote[l] += 1
				else:
					vote[j] += 1
		# determine class with maximum vote
		maxClass = -1
		maxVote = 0
		for j in range(k):
			if vote[j] >= maxVote:
				maxClass = j
				maxVote = vote[j]
		# check if prediction is correct
		if Y[i] == maxClass + 1:
			accuracy_count += 1
	# return accuracy
	return (accuracy_count * 100) / m	


## MODEL LEARNING FUNCTIONS (LIBSVM) ##

# function to train a (linear kernel) SVM binary classifier (using LIBSVM)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
# C: regularization hyperparameter
def svm_train_binary_linear_LIBSVM(X, Y, class_num, C):
	# store shape of training data
	m, n = X.shape
	# update class labels to -1 and 1
	for i in range(m):
		if Y[i][0] == class_num:
			Y[i][0] = -1
		else:
			# other class (change to 1)
			Y[i][0] = 1
	# learn SVM model (linear kernel, C )
	model = svm_train(Y, X, '-t 0 -c ' + str(C))
	# return learnt model
	return model

# function to train a (gaussian kernel) SVM binary classifier (using LIBSVM)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
# C: regularization hyperparameter
# gamma: kernel parameter
def svm_train_binary_gaussian_LIBSVM(X, Y, class_num, C, gamma):
	# store shape of training data
	m, n = X.shape
	# update class labels to -1 and 1
	for i in range(m):
		if Y[i][0] == class_num:
			Y[i][0] = -1
		else:
			# other class (change to 1)
			Y[i][0] = 1
	# learn SVM model (gaussian kernel, C, gamma)
	model = svm_train(Y, X, '-t 2 -c ' + str(C) + ' -g ' + str(gamma))
	# return learnt model
	return model

# function to train a (gaussian kernel) SVM multi-class classifier (using LIBSVM)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# C: regularization hyperparameter
# gamma: kernel parameter
def svm_train_multi_gaussian_LIBSVM(X, Y, C, gamma):
	# store shape of training data
	m, n = X.shape
	# learn SVM model (gaussian kernel, C, gamma)
	model = svm_train(Y, X, '-s 0 -t 2 -c ' + str(C) + ' -g ' + str(gamma))
	# return learnt model
	return model


## MODEL TESTING FUNCTIONS (LIBSVM) ##

# function to test SVM binary classification (LIBSVM)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
# model: it is the learnt SVM model
def svm_test_binary_LIBSVM(X, Y, class_num, model):
	# store the shape of test data
	m, n = X.shape
	# update class labels to -1 and 1
	for i in range(m):
		if Y[i][0] == class_num:
			Y[i][0] = -1
		else:
			# other class (change to 1)
			Y[i][0] = 1
	# return prediction accuracy
	return svm_predict(Y, X, model)

# function to test SVM multi-class classification (LIBSVM)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# model: it is the learnt SVM model
def svm_test_multi_LIBSVM(X, Y, model):
	# store the shape of test data
	m, n = X.shape
	# return prediction accuracy
	return svm_predict(Y, X, model)


## CROSS-VALIDATION FUNCTION ##

# function to do K-fold cross-validation to estimate best value of C
# K: number of folds
# X_train: input data (matrix of transposes) (for training/validation)
# Y_train: target variable (column vector) (for training/validation)
# C: list of hyperparameter C that needs to be tried
# gamma: kernel parameter
# X_test: input data (matrix of transposes) (for testing)
# Y_test: target variable (column vector) (for testing)
def cross_validation(K, X_train, Y_train, C, gamma, X_test, Y_test):
	# store the shape of training data
	m, n = X_train.shape
	# divide the training data into K-folds
	index_list = []
	for i in range(K):
		if i + 1 != K:
			index_list.append(range(i * (m // K), (i + 1) * (m // K)))
		else:
			index_list.append(range(i * (m // K), m))
	# max cross-validation accuracy and best value of C
	max_accuracy = 0
	best_C = -1
	# loop through values of C
	for c in C:
		# train K models
		models = [None for i in range(K)]
		# cross-validation accuracy and best model
		accuracy = 0
		best_model = None
		for i in range(K):
			# i^th partition is the validation set
			X_validation = X_train[index_list[i], :]
			Y_validation = Y_train[index_list[i], :]
			# use remaining as training set
			X = np.delete(X_train, index_list[i], axis=0)
			Y = np.delete(Y_train, index_list[i], axis=0)
			# train multi-class classification model (SVM)
			model[i] = svm_train_multi_gaussian_LIBSVM(X, Y, c, gamma)
			# determine validation accuracy
			p_1, p_acc, p_labs = svm_test_multi_LIBSVM(X_validation, Y_validation, model[i])
			if p_acc > accuracy:
				accuracy = p_acc
				best_model = model[i]
		# determine accuracy on test set
		p_1, p_acc, p_labs = svm_test_multi_LIBSVM(X_test, Y_test, best_model)
		# check for max accuracy
		if accuracy > max_accuracy:
			max_accuracy = accuracy
			best_C = c
	# return best value of C
	return best_C

print("Parsing CSV")
data_set = parse_csv("train.csv")
print("CSV parsed")
X, Y = split_data(data_set)
data_set = filter_data(X, Y, 4, 5)

print("Learning model")
alpha_s, X_s, Y_s = svm_train_binary_gaussian_CVXOPT(X, Y, 4, 1, 0.05)
print("Model learnt")
print("Parsing CSV (test)")
data_set = parse_csv("test.csv")
print("CSV parsed (test)")
data_set = filter_data(data_set, 4, 5)
print("Splitting data (test)")
X, Y = split_data(data_set)
print("Data split (test)")
print("Testing model")
print(svm_test_binary_gaussian_CVXOPT(X, Y, 4, 5, alpha_s, X_s, Y_s, 0.05))
print("Model tested")