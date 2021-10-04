import matplotlib.pyplot as plt
from libsvm.svmutil import *
from cvxopt import solvers
from cvxopt import matrix
from PIL import Image
import pandas as pd
import numpy as np
import time
import sys

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

# filter classes present in class_list from data_set (X and Y)
def filter_data(X, Y, class_list):
	# store shape of data set (X)
	m, n = X.shape
	# store row numbers of required data
	row_indices = []
	# go through entire data set for filtering
	for i in range(m):
		if Y[i][0] in class_list:
			# this is desired data
			row_indices.append(i)
	# select and return appropriate subset of data_set (X and Y)
	return X[row_indices, :], Y[row_indices, :]


## MODEL LEARNING FUNCTIONS (CVXOPT) ##

# function to train a (linear kernel) SVM binary classifier (using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
# C: model hyperparameter
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
	# determine matrix q (column of minus ones)
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
	# determine 'w' parameter = X.T @ (alpha * Y) (* denotes element-wise multiplication)
	w = support_vectors_X.T @ (support_vectors_alpha * support_vectors_Y)
	# determine intercept term, as all support_alpha < 1, therefore, noise terms are all zero (for support vectors)
	# hence, b = support_vector_Y - w.T @ support_vector_X (constraint becomes equality)
	# determining all such b's using all such vectors
	b = support_vectors_Y - support_vectors_X @ w
	# return parameters w and b (all terms in vector will be the same)
	# also return indices of SV and their coefficients
	return support_vectors_indices, support_vectors_alpha, w, b

# function to train a (gaussian kernel) SVM binary classifier (using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
# C: model hyperparameter
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
	# determine matrix q (column of minus ones)
	temp_1 = -np.ones((m, 1))
	q = matrix(temp_1, tc='d')
	# determine matrix P (P_ij = y_i*y_j*exp(-gamma*||x_i-x_j||^2))
	# creating 2D matrix of pairwise distances (using (x1 - x2).T @ (x1 - x2) = x1.T @ x1 + x2.T @ x2 - 2 * x1.T @ x2)
	temp_1 = X @ X.T	
	temp_2 = (X**2).sum(axis=1).reshape((m, 1))
	# creating matrix of pairwise target-products (used in kernel matrix)
	temp_3 = Y @ Y.T
	# create temp_4 = temp_3 * np.exp(-0.05(temp_2 + temp_2.T - 2 * temp_1)) (temp_2 is getting broadcasted)
	temp_4 = temp_3 * np.exp(-gamma * (temp_2 + temp_2.T - 2 * temp_1))
	# convert to matrix
	P = matrix(temp_4, tc='d')
	# solve the dual optimization (minimization) problem
	solution = solvers.qp(P, q, G, h, A, b)
	# extract optimal value of alpha (column vector)
	alpha = np.array(solution['x'])
	# determine support vectors (alpha > 1e-4)
	support_vectors_indices = []
	for i in range(m):
		# non-zero coefficient
		if alpha[i][0] > 1.2e-3:
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
	index_s = [[None for j in range(k)] for i in range(k)]
	alpha_s = [[None for j in range(k)] for i in range(k)]
	X_s = [[None for j in range(k)] for i in range(k)]
	Y_s = [[None for j in range(k)] for i in range(k)]	
	# go through all possible pairs and train models (k (k - 1) / 2)
	for i in range(k):
		for j in range(i + 1, k):
			# filter training data and retrieve subset (class = i and j)
			X_subset, Y_subset = filter_data(X, Y, [i, j])
			# train model, store in alpha_s, X_s and Y_s (higher class is set to one)
			index_s[i][j], alpha_s[i][j], X_s[i][j], Y_s[i][j] = svm_train_binary_gaussian_CVXOPT(X_subset.copy(), Y_subset.copy(), i, C, gamma)
			print("Classifier trained: " + str(i) + " vs " + str(j))	
	# all models trained
	# return parameters
	return index_s, alpha_s, X_s, Y_s


## MODEL TESTING FUNCTIONS (CVXOPT) ##

# function to test SVM binary classification (linear kernel)
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
# alpha_s, X_s and Y_s: support vector coefficients, input features and target values respectively
# gamma: kernel parameter
def svm_test_binary_gaussian_CVXOPT(X, Y, class1, class2, alpha_s, X_s, Y_s, gamma):
	# store the shape of test data
	m, n = X.shape
	# determine square norm of each example (store as row vector)
	X_2 = (X**2).sum(axis=1).reshape((1, m))	
	# store number of support vectors
	num_vectors = alpha_s.shape[0]
	# determine intercept term, b = y_s - summation(alpha_j * y_j * K(x_s, x_j)) over j
	temp = 0	
	for i in range(num_vectors):
		temp += alpha_s[i][0] * Y_s[i][0] * np.exp(-gamma * ((X_s[0, :] - X_s[i, :]).T @ (X_s[0, :] - X_s[i, :])))
	b = Y_s[0][0] - temp
	# determine square-norm matrix of X_s (store as column vector)
	X_s_2 = (X_s**2).sum(axis=1).reshape((num_vectors, 1))
	# determine test_vector-support_vector cross-outer-product matrix	
	cross_outer_product = X_s @ X.T
	# detemine kernel matrix
	K = np.exp(-gamma * (X_s_2 + X_2 - 2 * cross_outer_product))
	# determine alpha-label product matrix (for determining w.T @ x)
	alpha_label_product = alpha_s * Y_s
	# determine signed margin for each example (using kernel)
	margin = K.T @ alpha_label_product + b
	# make prediction
	prediction = (margin >= 0)	
	# initialise accuracy count to 0
	accuracy_count = 0
	# go through all examples and make predictions
	for i in range(m):
		if prediction[i][0]:
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
# gamma: kernel parameter
def svm_test_multi_gaussian_CVXOPT(X, Y, alpha_s, X_s, Y_s, gamma):
	# store number of examples
	m = X.shape[0]
	# store number of classes 
	k = len(alpha_s)
	# initialize confusion matrix
	confusion_matrix = np.zeros((k, k))
	# initialise accuracy count to 0
	accuracy_count = 0
	# initialise vote array
	vote = [[0 for j in range(k)] for i in range(m)]
	# initialise max score array (absolute functional margin)
	max_score = [[0 for j in range(k)] for i in range(m)]
	# determine square-norm matrix of X
	X_2 = (X**2).sum(axis=1).reshape((1, m))
	# go through all models and vote on each example
	for j in range(k):
		for l in range(j + 1, k):
			print("Testing classifier: " + str(j) + " vs " + str(l))	
			# determine number of support vectors
			num_vectors = alpha_s[j][l].shape[0]
			# determine intercept term, b = y_s - summation(alpha_i * y_i * K(x_s, x_i)) over i
			temp = 0
			for i in range(num_vectors):
				temp += alpha_s[j][l][i][0] * Y_s[j][l][i][0] * np.exp(-gamma * ((X_s[j][l][i, :] - X_s[j][l][0, :]).T @ (X_s[j][l][i, :] - X_s[j][l][0, :])))
			b = Y_s[j][l][0][0] - temp
			# determine square-norm matrix of X_s[j][l]
			X_s_2 = (X_s[j][l]**2).sum(axis=1).reshape((num_vectors, 1))
			# determine test_vector-support_vector cross-outer-product matrix	
			cross_outer_product = X_s[j][l] @ X.T
			# detemine kernel matrix
			K = np.exp(-gamma * (X_s_2 + X_2 - 2 * cross_outer_product))
			# determine alpha-label product matrix (for determining w.T @ x)
			alpha_label_product = alpha_s[j][l] * Y_s[j][l]
			# determine signed margin for each example (using kernel)
			margin = K.T @ alpha_label_product + b
			# make prediction
			prediction = (margin >= 0)
			# go through all examples and vote
			for i in range(m):
				if prediction[i][0]:
					vote[i][l] += 1
					max_score[i][l] = max(max_score[i][l], margin[i][0])
				else:
					vote[i][j] += 1
					max_score[i][j] = max(max_score[i][j], -(margin[i][0]))
	# for each class make prediction
	for i in range(m):
		prediction_class = 0
		for j in range(1, k):
			if vote[i][j] > vote[i][prediction_class]:
				prediction_class = j
			elif vote[i][j] == vote[i][prediction_class] and max_score[i][j] >= max_score[i][prediction_class]:
				prediction_class = j
		# increase appropriate entry in confusion matrix
		confusion_matrix[Y[i][0]][prediction_class] += 1
		# increase accuracy count, if correct
		if Y[i] == prediction_class:
			accuracy_count += 1
	# return accuracy and confusion matrix
	return (accuracy_count * 100) / m, confusion_matrix


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
	# learn SVM model (linear kernel, C)
	model = svm_train(Y.reshape(-1), X, '-t 0 -c ' + str(C))
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
	model = svm_train(Y.reshape(-1), X, '-t 2 -c ' + str(C) + ' -g ' + str(gamma))
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
	model = svm_train(Y.reshape(-1), X, '-s 0 -t 2 -c ' + str(C) + ' -g ' + str(gamma))
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
	return svm_predict(Y.reshape(-1), X, model)

# function to test SVM multi-class classification (LIBSVM)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# model: it is the learnt SVM model
def svm_test_multi_LIBSVM(X, Y, model):
	# return prediction accuracy
	return svm_predict(Y.reshape(-1), X, model)


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
	# validation accuracy per C
	validation_accuracy = [0 for i in range(len(C))]
	# test accuracy per C
	test_accuracy = [0 for i in range(len(C))]
	# loop through values of C
	for i in range(len(C)):
		# cross-validation accuracy and best model
		accuracy = 0
		best_model = None
		for j in range(K):
			# j^th partition is the validation set
			X_validation = X_train[index_list[j], :]
			Y_validation = Y_train[index_list[j], :]
			# use remaining as training set
			X = np.delete(X_train, index_list[j], axis=0)
			Y = np.delete(Y_train, index_list[j], axis=0)
			# train multi-class classification model (SVM)
			model = svm_train_multi_gaussian_LIBSVM(X.copy(), Y.copy(), C[i], gamma)
			# determine validation accuracy
			p_labs, p_acc, p_vals = svm_test_multi_LIBSVM(X_validation.copy(), Y_validation.copy(), model)
			if p_acc[0] > accuracy:
				accuracy = p_acc[0]
				best_model = model
		# add accuracy to validation_accuracy
		validation_accuracy[i] = accuracy
		# determine accuracy on test set
		p_labs, p_acc, p_vals = svm_test_multi_LIBSVM(X_test.copy(), Y_test.copy(), best_model)
		# add accuracy to test_accuracy
		test_accuracy[i] = p_acc[0]
	# return cross-validation and test accuracies
	return validation_accuracy, test_accuracy


## EXECUTION FUNCTIONS ##

# driver function (main)
def main():
	# process command line arguments
	if len(sys.argv) < 5:
		# insufficient number of arguments, print error and exit
		print("Error: All arguments not provided.")
		exit()
	if len(sys.argv) > 5:
		# extra arguments provided, print warning
		print("Warning: Extra arguments are provided")
	# five arguments provided
	# assuming that the arguments are correct, collect relevant data 
	train_filename = sys.argv[1]
	test_filename = sys.argv[2]
	is_multi = True if sys.argv[3] == '1' else False
	part_num = sys.argv[4]
	# load training/test data
	X_train, Y_train = split_data(parse_csv(train_filename))
	X_test, Y_test = split_data(parse_csv(test_filename))
	# for binary classification
	if not is_multi:
		# last digit of entry number=4 (2019CS10424)
		# filter training/test data for classes 4 and 5
		X_train, Y_train = filter_data(X_train, Y_train, [4, 5])
		X_test, Y_test = filter_data(X_test, Y_test, [4, 5])
		# switch on parts
		if part_num == 'a':
			# use linear kernel (CVXOPT)
			# get support vectors and parameters w and b
			start_time = time.time()
			SV_indices, SV_coeff, w, b = svm_train_binary_linear_CVXOPT(X_train.copy(), Y_train.copy(), 4, 1)
			end_time = time.time()
			print("Training time (in s): " + str(end_time - start_time))
			print("Number of support vectors, nSV: " + str(len(SV_indices)))
			print("SV indices in X: " + str(SV_indices))
			print("SV coefficients (transpose): " + str(SV_coeff.T))
			print("\'w\' parameter (transpose): " + str(w.T))
			print("\'b\' parameter: " + str(b[0]))
			# test model on training set
			accuracy = svm_test_binary_linear_CVXOPT(X_train.copy(), Y_train.copy(), 4, 5, w, b[0])
			print("Training accuracy (in %): " + str(accuracy))
			# test model on test set
			accuracy = svm_test_binary_linear_CVXOPT(X_test.copy(), Y_test.copy(), 4, 5, w, b[0])
			print("Test accuracy (in %): " + str(accuracy))
		elif part_num == 'b':
			# use gaussian kernel (CVXOPT)
			# get support vectors
			start_time = time.time()
			SV_indices, SV_coeff, SV_x, SV_y = svm_train_binary_gaussian_CVXOPT(X_train.copy(), Y_train.copy(), 4, 1, 0.05)
			end_time = time.time()
			print("Training time (in s): " + str(end_time - start_time))
			print("Number of support vectors, nSV: " + str(len(SV_indices)))
			print("SV indices in X: " + str(SV_indices))
			print("SV coefficients (transpose): " + str(SV_coeff.T))
			# test model on training set
			accuracy = svm_test_binary_gaussian_CVXOPT(X_train.copy(), Y_train.copy(), 4, 5, SV_coeff, SV_x, SV_y, 0.05)
			print("Training accuracy (in %): " + str(accuracy))
			# test model on test set
			accuracy = svm_test_binary_gaussian_CVXOPT(X_test.copy(), Y_test.copy(), 4, 5, SV_coeff, SV_x, SV_y, 0.05)
			print("Test accuracy (in %): " + str(accuracy))
		elif part_num == 'c':
			# use linear kernel (LIBSVM)
			start_time = time.time()
			model_linear = svm_train_binary_linear_LIBSVM(X_train.copy(), Y_train.copy(), 4, 1)
			end_time = time.time()
			print("Training time (in s) (Linear Kernel): " + str(end_time - start_time))
			# get SV indices and coefficients
			SV_indices, SV_coeff = model_linear.get_sv_indices(), model_linear.get_sv_coef()
			print("Number of support vectors, nSV: " + str(len(SV_indices)))
			# make 0-indexed
			for i in range(len(SV_indices)):
				SV_indices[i] -= 1
			print("SV indices in X: " + str(SV_indices))
			print("SV coefficients: " + str(SV_coeff))
			# SV_alpha = np.array(SV_coeff).reshape((len(SV_indices), 1))
			# SV_x = X_train[SV_indices, :]
			# SV_y = Y_train[SV_indices, :]
			# for i in range(len(SV_indices)):
			# 	SV_y[i][0] = -1 if SV_y[i][0] == 4 else 1
			# w = SV_x.T @ (SV_alpha * SV_y)
			# print(SV_alpha * SV_y)
			# b = SV_y - SV_x @ w
			SV_indices.sort()
			print("SV indices in sorted order for comparison: " + str(SV_indices))
			# test model on test set
			p_labs, p_acc, p_vals = svm_test_binary_LIBSVM(X_test.copy(), Y_test.copy(), 4, model_linear)
			print("Test accuracy (in %) (Linear Kernel): " + str(p_acc[0]))
			# use gaussian kernel (LIBSVM)
			start_time = time.time()
			model_gaussian = svm_train_binary_gaussian_LIBSVM(X_train.copy(), Y_train.copy(), 4, 1, 0.05)
			end_time = time.time()
			print("Training time (in s) (Gaussian Kernel): " + str(end_time - start_time))
			# get SV indices and coefficients
			SV_indices, SV_coeff = model_gaussian.get_sv_indices(), model_gaussian.get_sv_coef()
			print("Number of support vectors, nSV: " + str(len(SV_indices)))
			# make 0-indexed
			for i in range(len(SV_indices)):
				SV_indices[i] -= 1
			print("SV indices in X: " + str(SV_indices))
			print("SV coefficients: " + str(SV_coeff))
			SV_indices.sort()
			print("SV indices in sorted order for comparison: " + str(SV_indices))
			# test model on test set
			p_labs, p_acc, p_vals = svm_test_binary_LIBSVM(X_test.copy(), Y_test.copy(), 4, model_gaussian)
			print("Test accuracy (in %) (Gaussian Kernel): " + str(p_acc[0]))
	else:
		# multi-class problem
		# switch on parts
		if part_num == 'a':
			# use gaussian kernel (CVXOPT)
			SV_indices, SV_coeff, SV_x, SV_y = svm_train_multi_gaussian_CVXOPT(X_train.copy(), Y_train.copy(), 10, 1, 0.05)
			# test model on test data
			accuracy, confusion_matrix = svm_test_multi_gaussian_CVXOPT(X_test.copy(), Y_test.copy(), SV_coeff, SV_x, SV_y, 0.05)
			print("Test accuracy (in %): " + str(accuracy))
		elif part_num == 'b':
			# use gaussian kernel (LIBSVM)
			start_time = time.time()
			model_gaussian = svm_train_multi_gaussian_LIBSVM(X_train.copy(), Y_train.copy(), 1, 0.05)
			end_time = time.time()
			print("Training time (in s): " + str(end_time - start_time))
			# test model on test set
			p_labs, p_acc, p_vals = svm_test_multi_LIBSVM(X_test.copy(), Y_test.copy(), model_gaussian)
			print("Test accuracy (in %): " + str(p_acc[0]))
		elif part_num == 'c':
			# determine confusion matrix (multi-class)
			# use gaussian kernel (LIBSVM)
			model_gaussian = svm_train_multi_gaussian_LIBSVM(X_train.copy(), Y_train.copy(), 1, 0.05)
			# test model on test set (p_labs contain labels)
			p_labs, p_acc, p_vals = svm_test_multi_LIBSVM(X_test.copy(), Y_test.copy(), model_gaussian)
			# initialise confusion matrix
			confusion_matrix = np.zeros((10, 10))
			# go through all examples
			for i in range(len(p_labs)):
				# increment appropriate entry
				confusion_matrix[Y_test[i][0]][int(p_labs[i])] += 1
			print("Confusion matrix (multi-class, LIBSVM)")
			print(confusion_matrix)
			# save confusion matrix in a file
			file = open("Q2_multi_confusion.txt", "w")
			file.write('Confusion matrix in case of multi-class classification (LIBSVM)\n')
			file.write('(Rows represent actual digit (0-9) and columns represent predicted digit (0-9))\n')
			file.write(str(confusion_matrix))	
			file.write('\n\n')
			print()
			# determine confusion matrix (multi-class)
			# use gaussian kernel (CVXOPT)
			SV_indices, SV_coeff, SV_x, SV_y = svm_train_multi_gaussian_CVXOPT(X_train.copy(), Y_train.copy(), 10, 1, 0.05)
			# test model on test data
			accuracy, confusion_matrix = svm_test_multi_gaussian_CVXOPT(X_test.copy(), Y_test.copy(), SV_coeff, SV_x, SV_y, 0.05)
			print("Confusion matrix (multi-class, CVXOPT)")
			print(confusion_matrix)
			# save confusion matrix in a file
			file.write('Confusion matrix in case of multi-class classification (CVXOPT)\n')
			file.write('(Rows represent actual digit (0-9) and columns represent predicted digit (0-9))\n')
			file.write(str(confusion_matrix))	
			file.write('\n\n')
			print()
			# close file
			file.close()
		elif part_num == 'd':
			# perform cross-validation
			# list of C values that need to be tried
			C = [1e-5, 1e-3, 1, 5, 10]
			# determine cross-validation and test-accuracies
			validation_accuracy, test_accuracy = cross_validation(5, X_train.copy(), Y_train.copy(), C, 0.05, X_test.copy(), Y_test.copy())
			print("Cross-Validation accuracies (in %): " + str(validation_accuracy))
			print("Test accuracies (in %): " + str(test_accuracy))
			# plot accuracies on graph (log scale on x-axis)
			plt.xscale("log")
			# plot both curves
			plt.plot(C, validation_accuracy, label="Cross-Validation")
			plt.plot(C, test_accuracy, label="Test")
			# show legend
			plt.legend()
			# set axis-label
			plt.xlabel("Value of C hyperparameter")
			plt.ylabel("Prediction Accuracy (in %)")
			plt.savefig("Q2_cv.jpg")



main()