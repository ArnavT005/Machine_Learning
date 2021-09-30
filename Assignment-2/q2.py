from libsvm.svmutil import *
from cvxopt import solvers
from cvxopt import matrix
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

# filter classes 'class_num' and 'class_num + 1' from from data_set
def filter_data(data_set, class_num):
	# store shape of data set
	m, n = data_set.shape
	# store row numbers of required data
	row_indices = []
	# go through entire data set for filtering
	for i in range(m):
		if data_set[i][n - 1] == class_num or data_set[i][n - 1] == class_num + 1:
			# this is desired data
			row_indices.append(i)
	# select subset array from data_set
	final_data_set = data_set[row_indices, :]
	# return the final data set
	return final_data_set

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

# function to train a (linear kernel) SVM binary classifier (using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
def svm_train_binary_linear_CVXOPT(X, Y, class_num):
	# store shape of training data
	m, n = X.shape
	# update class labels to -1 and 1
	for i in range(m):
		if Y[i][0] == class_num:
			Y[i][0] = -1
		else:
			# label is class_num + 1 (change to 1)
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
	# determining matrix h (C = 1)
	temp_1 = np.zeros((m, 1))
	temp_2 = np.ones((m, 1))
	# row_stack temp_1 and temp_2
	temp = np.row_stack((temp_1, temp_2))
	h = matrix(temp, tc='d')
	# determine matrix q (column of ones)
	q = matrix(-temp_2, tc='d')
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
	# determine intercept term, as all alpha < 1, therefore, noise terms are all zero
	# hence, b = support_vector_Y - w.T @ support_vector_X (constraint becomes equality)
	# determining all such b's using all such vectors
	b = support_vectors_Y - support_vectors_X @ w
	# return parameters w and b
	return w, b

# function to train a (gaussian kernel) SVM binary classifier (using CVXOPT)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class will be 1
def svm_train_binary_gaussian_CVXOPT(X, Y, class_num):
	# store shape of training data
	m, n = X.shape
	# update class labels to -1 and 1
	for i in range(m):
		if Y[i][0] == class_num:
			Y[i][0] = -1
		else:
			# label is class_num + 1 (change to 1)
			Y[i][0] = 1
	# we can multiply the dual objective by -1 to convert it into minimization problem
	# express dual objective as (1/2) x.T @ P @ x + q.T @ x
	# subject to constraints, Gx <= h and Ax = b
	# all matrices except P will remain the same even in gaussian kernel
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
	# determining matrix h (C = 1)
	temp_1 = np.zeros((m, 1))
	temp_2 = np.ones((m, 1))
	# row_stack temp_1 and temp_2
	temp = np.row_stack((temp_1, temp_2))
	h = matrix(temp, tc='d')
	# determine matrix q (column of ones)
	q = matrix(-temp_2, tc='d')
	# determine matrix P (P_ij = y_i*y_j*exp(-gamma*||x_i-x_j||^2))
	P = np.zeros((m, m), dtype=np.float64)
	for i in range(m):
		for j in range(m):
			P[i][j] += Y[i][0] * Y[j][0] * np.exp(-0.05 * ((X[i, :] - X[j, :]) @ (X[i, :] - X[j, :]).T))
	# convert to matrix
	P = matrix(P, tc='d')
	# solve the dual optimization (minimization) problem
	solution = solvers.qp(P, q, G, h, A, b)
	# extract optimal value of alpha (column vector)
	alpha = np.array(solution['x'])
	# determine support vectors (alpha > 1e-5)
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
	return support_vectors_alpha, support_vectors_X, support_vectors_Y

# function to test SVM binary classification
# w and b are provided explicitly
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class (class_num + 1) will be 1
# w, b are the model parameters
def svm_test_binary_explicit(X, Y, class_num, w, b):
	# store the shape of test data
	m, n = X.shape
	# initialise accuracy count to 0
	accuracy_count = 0
	# go through all examples and make predictions
	for i in range(m):
		if X[i, :] @ w + b >= 0:
			# predict class_num + 1
			if Y[i] == class_num + 1:
				accuracy_count += 1
		else:
			# predict class_num
			if Y[i] == class_num:
				accuracy_count += 1
	# return prediction accuracy
	return (accuracy_count * 100) / m

# function to test SVM binary classification
# alpha is provided (w and b are implicit)
# X: input data (matrix of transposes)
# Y: target variable (column vector)
# class_num: class number that needs to be treated as -1, other class (class_num + 1) will be 1
# alpha_s, X_s and Y_s are support vectors coefficients, input features and target values respectively
def svm_test_binary_implicit(X, Y, class_num, alpha_s, X_s, Y_s):
	# store the shape of test data
	m, n = X.shape
	# store number of support vectors
	num_vectors = alpha_s.shape[0]
	# determine intercept term, b = y_s - summation(alpha_j * y_j * K(x_s, x_j)) over j
	temp = 0
	for j in range(num_vectors):
		temp += alpha_s[j][0] * Y_s[j][0] * np.exp(-0.05 * ((X_s[0, :] - X_s[j, :]) @ (X_s[0, :] - X_s[j, :]).T))
	b = Y_s[0][0] - temp
	# initialise accuracy count to 0
	accuracy_count = 0
	# go through all examples and make predictions
	for i in range(m):
		# determine w.T @ x = summation(alpha_j * y_j * K(x_j, x))
		temp = 0
		for j in range(num_vectors):
			temp += alpha_s[j][0] * Y_s[j][0] * np.exp(-0.05 * ((X[i, :] - X_s[j, :]) @ (X[i, :] - X_s[j, :]).T))
		if temp + b >= 0:
			# predict class_num + 1
			if Y[i] == class_num + 1:
				accuracy_count += 1
		else:
			# predict class_num
			if Y[i] == class_num:
				accuracy_count += 1
	# return prediction accuracy
	return (accuracy_count * 100) / m

print("Parsing CSV")
data_set = parse_csv("train.csv")
print("CSV parsed")
print("Filtering data")
data_set = filter_data(data_set, 4)
print("Data filtered")
print("Splitting data")
X, Y = split_data(data_set)
print("Data split")
# for i in range(Y.shape[0]):
# 	if Y[i][0] == 4:
# 		Y[i][0] = -1
# 	else:
# 		Y[i][0] = 1
# Y = Y.reshape(-1)
# m = svm_train(Y, X, '-c 1 -t 2 -g 0.05')
# V = m.get_SV()
alpha_s, X_s, Y_s, = svm_train_binary_gaussian_CVXOPT(X, Y, 4)
# # w, b = svm_train_binary_linear_CVXOPT(X, Y, 4)
print("Parsing CSV (test)")
data_set = parse_csv("test.csv")
print("CSV parsed (test)")
print("Filtering data (test)")
data_set = filter_data(data_set, 4)
print("Data filtered (test)")
print("Splitting data (test)")
X, Y = split_data(data_set)
print("Data split (test)")
# for i in range(Y.shape[0]):
# 	if Y[i][0] == 4:
# 		Y[i][0] = -1
# 	else:
# 		Y[i][0] = 1
# Y = Y.reshape(-1)
print("Testing model")
accuracy = svm_test_binary_implicit(X, Y, 4, alpha_s, X_s, Y_s)
print("Model tested")
# p_labs, p_acc, p_vals = svm_predict(Y, X, m)

print("Accuracy: " + str(accuracy))
# print(p_acc)
#print(V)