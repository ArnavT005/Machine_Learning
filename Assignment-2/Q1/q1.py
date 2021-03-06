import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
import random
import string
import json
import nltk
import sys

nltk.download('punkt')
nltk.download('stopwords')

# parse review data and generate training/test set
# file: name of the file
def parse_json(file):
	# read review file line-by-line and store the dataframe
	df = pd.read_json(file, lines=True)
	# store reviewText and overall ratings (and summary)
	# convert to list
	X = df["reviewText"].tolist()
	Y = df["overall"].tolist()
	summary = df["summary"].tolist()
	# return input/output data and summary
	return X, Y, summary

# function to preprocess input data
# X is the list of texts
def preprocess(X):
	# translate table (to remove punctuations)
	table = str.maketrans('', '', string.punctuation)
	# initialise new input list (to be returned)
	# for every example, it stores the list of words
	X_ = [[] for i in range(len(X))]
	# split every input
	for i in range(len(X)):
		# split on spaces/punctuations
		words = nltk.word_tokenize(X[i])
		# convert to lower case and remove punctuations
		# only keep alphabetical words
		X_[i] = [word.lower().translate(table) for word in words if word.lower().translate(table).isalpha()]
	# return preprocessed data
	return X_

# function to preprocess input data
# filter stop words and perform stemming
# X is the list of texts (input data)
# punctuation: flag to indicate that ? and ! need to be kept
def preprocess_advanced(X, punctuation=False):
	# translate table (to remove punctuations)
	table = str.maketrans('', '', string.punctuation)
	# store stop words (english)
	stop_words = Counter(nltk.corpus.stopwords.words('english'))
	# create stemmer object
	stemmer = nltk.stem.porter.PorterStemmer()
	# initialise new input list (to be returned)
	X_ = [[] for i in range(len(X))]
	# split every input
	for i in range(len(X)):
		# split on spaces/punctuations
		words = nltk.word_tokenize(X[i])
		# store punctuations if required
		punc = []
		if punctuation:
			punc = [word for word in words if word == '!' or word == '?']
		words = [word.lower() for word in words]
		words = [word.translate(table) for word in words]
		for word in words:
			# only store alphabetical words (no numerics or alphanumerics)
			if word.isalpha():
				# stem word
				word = stemmer.stem(word)
				# append word into X_ list (only if it is not a stop word)
				if word not in stop_words:
					X_[i].append(word)
		X_[i].extend(punc)
	# return preprocessed data
	return X_

# function to generate bigrams from given tokens
# X: tokenized list of reviews
def bigram_generator(X):
	X_ = []
	# go through all examples and generate bigrams
	for i in range(len(X)):
		X_.append(list(nltk.bigrams(X[i])))
	# return the new list
	return X_

# function to extend training data by including summarized review text
# X: review text data
# summary: summarized review data
def extend_data(X, summary):
	# store number of input examples
	m = len(X)
	# go through all examples, extend data
	for i in range(m):
		# store number of words in X[i] and summary
		num_words_X = len(X[i])
		num_words_S = len(summary[i])
		# extend X[i] only if X[i] is empty or summary is non-empty
		if X[i] == []:
			X[i].extend(summary[i])
			continue
		if num_words_S != 0:
			for j in range(num_words_X // (min(8 * num_words_S, num_words_X))):
				X[i].extend(summary[i])
	# return the extended data
	return X

# create and return vocabulary (V)
# X is the list of texts (input data)
# bigrams: flag that tells if a bigram vocabulary is being created
def create_vocabulary(X, bigrams=False):
	# initialise vocabulary (dictionary)
	V, size = {}, 0
	# split every input
	for i in range(len(X)):
		for word in X[i]:
			# add to dictionary (if not present already)
			if not word in V.keys():
				V[word] = size + 1
				size += 1
	# add unknown token to the dictionary (for unseen data)
	if bigrams:
		V[('UNKNOWN', 'UNKNOWN')] = size + 1
	else:
		V['UNKNOWN'] = size + 1
	size += 1
	# return the vocabulary (and size)
	return V, size

# function to train a Naive-Bayes classifier (multinomial)
# X: input data (list of list of words)
# Y: target variable (review rating)
# V: word vocabulary (size = |V|)
def naive_bayes_train(X, Y, V, size):
	# number of training examples
	m = len(Y)
	# there are five classes in total (5 phi parameters and 5*size theta parameters)
	# store majority class
	majority_class = 1
	# phi[i] = probability that target class is (i + 1) = [count of (i + 1)] / m
	phi = [0 for i in range(5)]
	for j in range(m):
		# count occurence of class Y[j] - 1
		phi[Y[j] - 1] += 1
		# check if it is majority class
		if phi[Y[j] - 1] > phi[majority_class - 1]:
			majority_class = Y[j]
	# divide by total count of examples to get class prior
	for i in range(5):
		phi[i] /= m
	# theta[i][j] = probability of occurence of word (j + 1) given target class (i + 1)
	# theta[i][j] = (count of word (j + 1) in reviews of class (i + 1) + 1) / (total word count in all reviews of class (i + 1) + |V|)
	# 1 and |V| have been added for Laplace Smoothing
	theta = [[1 for j in range(size)] for i in range(5)]
	# store word count for each class
	word_count = [0 for i in range(5)]
	# go through all training data
	for j in range(m):
		# increment word count of class Y[j] - 1
		word_count[Y[j] - 1] += len(X[j])
		# go through all words in an example
		for word in X[j]:
			# determine word index in dictionary
			word_index = V[word]
			# add 1 to theta[Y[j] - 1][word_index - 1]
			theta[Y[j] - 1][word_index - 1] += 1
	# divide each parameter by total word count in review of that class + |V| (size)
	for i in range(5):
		for j in range(size):
			theta[i][j] /= (word_count[i] + size)
	# all parameters computed
	# return these parameters
	return phi, theta, majority_class

# function to test Naive-Bayes classifier (multinomial)
# X: test data (input, list of list of words)
# Y: test data (output, review rating)
# V: word vocabulary
# phi, theta: model parameters
# bigrams: flag that tells if a bigram vocabulary is being used
def naive_bayes_test(X, Y, V, phi, theta, bigrams=False):
	# number of test examples
	m = len(Y)
	# count accuracy = (accuracy_count / m) * 100
	accuracy_count = 0
	# initialise confusion matrix
	confusion_matrix = np.zeros((5, 5))
	# go through every example and make prediction
	for i in range(m):
		# maximise P(x|y=k)P(y=k) over 1 <= k <= 5
		# equivalent to maximising log(P(x|y=k)) + log(P(y=k))
		max_value = 0
		max_class = 0
		for k in range(5):
			# add the prior probability
			temp_value = np.log(phi[k])
			# go through all words in example i
			for word in X[i]:
				if word in V.keys():
					# get word index
					word_index = V[word]
					# add to temp_value
					temp_value += np.log(theta[k][word_index - 1])
				else:
					# word is UNKNOWN, add log(P(UNKNOWN|y=k))
					if bigrams:
						word_index = V[('UNKNOWN', 'UNKNOWN')]
					else:
						word_index = V['UNKNOWN']
					temp_value += np.log(theta[k][word_index - 1])
			# check if it is the maximum value
			if max_class == 0:
				max_value = temp_value
				max_class = k + 1
			elif temp_value > max_value:
				max_value = temp_value
				max_class = k + 1
		# predict class (max_class)
		if max_class == Y[i]:
			# correct prediction, increase accuracy count
			accuracy_count += 1
		# add entry to confusion matrix
		confusion_matrix[Y[i] - 1][max_class - 1] += 1
	# return accuracy = (accuracy_count / m) * 100 and confusion matrix
	return (accuracy_count * 100) / m, confusion_matrix

# function to give random predictions on test data
# Y: test data (output, review rating)
def random_test(Y):
	# number of test examples
	m = len(Y)
	# initialise confusion matrix
	confusion_matrix = np.zeros((5, 5))
	# go through all examples (predict randomly)
	# count accuracy = (correct predictions) / m
	accuracy_count = 0
	for i in range(m):
		prediction = random.choice([1, 2, 3, 4, 5])
		if Y[i] == prediction:
			accuracy_count += 1
		# increment appropriate entry
		confusion_matrix[Y[i] - 1][prediction - 1] += 1
	# return accuracy = (accuracy_count / m) * 100 and confusion matrix
	return (accuracy_count * 100) / m, confusion_matrix

# function to give majority predictions on test data
# Y: test data (output, review rating)
# majority_class: class which occurs most of the times in the training data
def majority_test(Y, majority_class):
	# number of test examples
	m = len(Y)
	# initialise confusion matrix
	confusion_matrix = np.zeros((5, 5))
	# go through all examples (predict majority)
	# count accuracy = (accuracy_count / m) * 100
	accuracy_count = 0
	for i in range(m):
		if Y[i] == majority_class:
			accuracy_count += 1
		# increment appropriate entry
		confusion_matrix[Y[i] - 1][majority_class - 1] += 1
	# return accuracy = (accuracy_count / m) * 100 and confusion matrix
	return (accuracy_count * 100) / m, confusion_matrix

# function to compute F1-score (macro, per-class)
# confusion_matrix: confusion matrix for the data
def f1_score(confusion_matrix):
	# store number of classes
	k = confusion_matrix.shape[0]
	# determine per-class precision, recall and f1-score
	precision = [0 for i in range(k)]
	recall = [0 for i in range(k)]
	f1 = [0 for i in range(k)]
	for i in range(k):
		# true positives
		TP = confusion_matrix[i][i]
		# count total examples which are actually of class (i + 1)
		TP_FN = 0
		# count total examples which are predicted to be of class (i + 1)
		TP_FP = 0
		for j in range(k):
			TP_FN += confusion_matrix[i][j]
			TP_FP += confusion_matrix[j][i]
		# determine precision, recall, f1-score
		precision[i] = 0 if TP == 0 else TP / TP_FP
		recall[i] = 0 if TP == 0 else TP / TP_FN
		if precision[i] == 0 and recall[i] == 0:
			f1[i] = 0
		else:
			f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
	# determine macro-averaged f1-score
	macro_f1 = sum(f1) / k
	# return f1 and macro-f1 score
	return f1, macro_f1


## EXECUTION FUNCTIONS ##

# driver function (main)
def main():
	# process command line arguments
	if len(sys.argv) < 4:
		# insufficient number of arguments, print error and exit
		print("Error: All arguments not provided.")
		exit()
	if len(sys.argv) > 4:
		# extra arguments provided, print warning
		print("Warning: Extra arguments are provided")
	# at least four arguments provided
	# assuming that the arguments are correct, collect relevant data 
	train_filename = sys.argv[1]
	test_filename = sys.argv[2]
	part_num = sys.argv[3]
	# load training/test data
	X_train, Y_train, summary_train = parse_json(train_filename)
	X_test, Y_test, summary_test = parse_json(test_filename)
	# switch on parts
	if part_num == 'a':
		# preprocess (simple) input data, not using summary
		X_train = preprocess(X_train)
		X_test = preprocess(X_test)
		# create vocabulary
		V, size = create_vocabulary(X_train)
		# train model
		phi, theta, majority_class = naive_bayes_train(X_train, Y_train, V, size)
		# test model on training set
		accuracy, confusion_matrix = naive_bayes_test(X_train, Y_train, V, phi, theta)
		print("Training accuracy (in %): " + str(accuracy))
		# test model on test set
		accuracy, confusion_matrix = naive_bayes_test(X_test, Y_test, V, phi, theta)
		print("Test accuracy (in %): " + str(accuracy))
		# determine macro-f1 score on test data
		f1, macro_f1 = f1_score(confusion_matrix)
		print("Macro F1-score: " + str(macro_f1))
		# print f1-score per class
		for i in range(5):
			print("F1-score for class " + str(i + 1) + ": " + str(f1[i]))
	elif part_num == 'b':
		# determine majority class in training data
		majority_class, count = Counter(Y_train).most_common()[0]
		# determine majority accuracy
		accuracy, confusion_matrix = majority_test(Y_test, majority_class)
		print("Majority accuracy (in %): " + str(accuracy))
		# determine macro-f1 score on test data
		f1, macro_f1 = f1_score(confusion_matrix)
		print("Macro F1-score: " + str(macro_f1))
		# print f1-score per class
		for i in range(5):
			print("F1-score for class " + str(i + 1) + ": " + str(f1[i]))
		# determine random accuracy
		accuracy, confusion_matrix = random_test(Y_test)
		print("Random accuracy (in %): " + str(accuracy))
		# determine macro-f1 score on test data
		f1, macro_f1 = f1_score(confusion_matrix)
		print("Macro F1-score: " + str(macro_f1))
		# print f1-score per class
		for i in range(5):
			print("F1-score for class " + str(i + 1) + ": " + str(f1[i]))
	elif part_num == 'c':
		# draw confusion matrix
		# first train and test data
		X_train = preprocess(X_train)
		X_test = preprocess(X_test)
		# create vocabulary
		V, size = create_vocabulary(X_train)
		# train model
		phi, theta, majority_class = naive_bayes_train(X_train, Y_train, V, size)
		# test model on test set, find confusion matrixs
		accuracy, confusion_matrix = naive_bayes_test(X_test, Y_test, V, phi, theta)
		# plot and save confusion matrix
		df_cm = pd.DataFrame(confusion_matrix, range(1, 6), range(1, 6))
		plt.figure(1)	
		sns_plot = sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 8})
		plt.savefig("Q1_cm.png")
		print("Confusion Matrix saved as Q1_cm.png")
	elif part_num == 'd':
		# stopword removal and stemming
		# preprocess input data, not using summary 
		X_train = preprocess_advanced(X_train)
		X_test = preprocess_advanced(X_test)
		# create vocabulary
		V, size = create_vocabulary(X_train)
		# train model
		phi, theta, majority_class = naive_bayes_train(X_train, Y_train, V, size)
		# test model on test set
		accuracy, confusion_matrix = naive_bayes_test(X_test, Y_test, V, phi, theta)
		print("Test accuracy (in %): " + str(accuracy))
		# determine macro-f1 score on test data
		f1, macro_f1 = f1_score(confusion_matrix)
		print("Macro F1-score: " + str(macro_f1))
		# print f1-score per class
		for i in range(5):
			print("F1-score for class " + str(i + 1) + ": " + str(f1[i]))
	elif part_num == 'e':
		# feature engineering: two additions done
		# new features: bigrams
		# feature manipulation: unigrams can also correspond to
		#                       some special punctuations like "!" or "?"
		# process data (enforce manipulation)
		X_train = preprocess_advanced(X_train, True)
		X_test = preprocess_advanced(X_test, True)
		# generate bigrams
		X_train_bigrams = bigram_generator(X_train)
		X_test_bigrams = bigram_generator(X_test)
		# add new features to data
		for i in range(len(X_train)):
			X_train[i].extend(X_train_bigrams[i])
		for i in range(len(X_test)):
			X_test[i].extend(X_test_bigrams[i])
		# create vocabulary
		V, size = create_vocabulary(X_train)
		# train model
		phi, theta, majority_class = naive_bayes_train(X_train, Y_train, V, size)
		# test model on test set
		accuracy, confusion_matrix = naive_bayes_test(X_test, Y_test, V, phi, theta)
		print("Test accuracy (in %): " + str(accuracy))
		# determine macro-f1 score on test data
		f1, macro_f1 = f1_score(confusion_matrix)
		print("Macro F1-score: " + str(macro_f1))
		# print f1-score per class
		for i in range(5):
			print("F1-score for class " + str(i + 1) + ": " + str(f1[i]))
	elif part_num == 'g':
		# incorporate summary into training data
		X_train = preprocess(X_train)
		summary_train = preprocess(summary_train)
		# extend training data by including summarized review
		# summary is given more weight, as it is kind of a broad description of the entire review
		X_train = extend_data(X_train, summary_train)
		# process test data in similar fashion
		X_test = preprocess(X_test)
		summary_test = preprocess(summary_test)
		X_test = extend_data(X_test, summary_test)
		# create vocabulary
		V, size = create_vocabulary(X_train)
		# train model
		phi, theta, majority_class = naive_bayes_train(X_train, Y_train, V, size)
		# test model on test set
		accuracy, confusion_matrix = naive_bayes_test(X_test, Y_test, V, phi, theta)
		print("Test accuracy (in %): " + str(accuracy))
		# determine macro-f1 score on test data
		f1, macro_f1 = f1_score(confusion_matrix)
		print("Macro F1-score: " + str(macro_f1))
		# print f1-score per class
		for i in range(5):
			print("F1-score for class " + str(i + 1) + ": " + str(f1[i]))
	

main()
