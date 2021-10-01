import pandas as pd
import numpy as np
import string
import random
import nltk
import math

# parse review data and generate training/test set
# parameter file stores the name of the file
def parse_json(file):
	# read review file line-by-line and store the dataframe
	df = pd.read_json(file, lines=True)
	# store reviewText and overall ratings
	# convert to list
	X = df["reviewText"].tolist()
	Y = df["overall"].tolist()
	# return input/output data
	return X, Y

# split input data into words (separated by spaces/punctuations)
# also create and return vocabulary (V)
# X is the list of texts (input data)
def split_input(X):
	# initialise vocabulary (dictionary)
	V, size = {}, 0
	# initialise new input list (to be returned)
	# for every example, it stores the list of words
	X_ = [[] for i in range(len(X))]
	# split every input
	for i in range(len(X)):
		# split on spaces/punctuations
		words = nltk.word_tokenize(X[i])
		# filter punctuations
		for word in words:
			# check if word is void of punctuations
			# only store alphabetical words (no numerics or alphanumerics)
			if word.isalpha():
				# convert word to lower case
				word = word.lower()
				# append word into X_ list 
				X_[i].append(word)
				# add to dictionary (if not present already)
				if not word in V.keys():
					V[word] = size + 1
					size += 1
	# add unknown token to the dictionary (for unseen data)
	V['UNKNOWN'] = size + 1
	size += 1
	# return the split data and vocabulary (and size)
	return X_, V, size

# split input data into words (separated by spaces/punctuations)
# filter stop words and perform stemming
# also create and return vocabulary (V)
# X is the list of texts (input data)
def split_input_advanced(X):
	# store stop words (english)
	stop_words = set(nltk.corpus.stopwords.words('english'))
	# create stemmer object
	stemmer = nltk.stem.porter.PorterStemmer()
	# initialise vocabulary (dictionary)
	V, size = {}, 0
	# initialise new input list (to be returned)
	# for every example, it stores the list of words
	X_ = [[] for i in range(len(X))]
	# split every input
	for i in range(len(X)):
		# split on spaces/punctuations
		words = nltk.word_tokenize(X[i])
		# filter punctuations
		for word in words:
			# check if word is void of punctuations
			# only store alphabetical words (no numerics or alphanumerics)
			if word.isalpha():
				# convert word to lower case
				word = word.lower()
				# stem word
				word = stemmer.stem(word)
				# append word into X_ list (only if it is not a stop word)
				if not word in stop_words:
					X_[i].append(word)
					# add to dictionary (if not present already)
					if not word in V.keys():
						V[word] = size + 1
						size += 1
	# add unknown token to the dictionary (for unseen data)
	V['UNKNOWN'] = size + 1
	size += 1
	# return the split data and vocabulary (and size)
	return X_, V, size

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
# V: word vocabulary (size = |V|)
# phi, theta: learnt parameters (model)
def naive_bayes_test(X, Y, V, size, phi, theta):
	# number of test examples
	m = len(Y)
	# go through every example and make prediction
	# count accuracy = (accuracy_count / m) * 100
	accuracy_count = 0
	for i in range(m):
		# maximise P(x|y=k)P(y=k) over 1 <= k <= 5
		# equivalent to maximising log(P(x|y=k)) + log(P(y=k))
		maxValue = 0
		maxClass = 0
		for k in range(5):
			# add the prior probability
			tempValue = math.log(phi[k])
			# go through all words in example i
			for word in X[i]:
				if word in V.keys():
					# get word index
					word_index = V[word]
					# add to tempValue
					tempValue += math.log(theta[k][word_index - 1])
				else:
					# word is UNKNOWN, add log(P(UNKNOWN=1|y=k))
					tempValue += math.log(theta[k][size - 1])
			# check if it is the maximum value
			if maxClass == 0:
				maxValue = tempValue
				maxClass = k + 1
			elif tempValue >= maxValue:
				maxValue = tempValue
				maxClass = k + 1
		# predict class (maxClass)
		if maxClass == Y[i]:
			# correct prediction, increase accuracy count
			accuracy_count += 1
	# return accuracy = (accuracy_count / m) * 100
	return (accuracy_count * 100) / m

# function to give random predictions on test data
# X: test data (input, list of list of words)
# Y: test data (output, review rating)
def random_test(X, Y):
	# number of test examples
	m = len(Y)
	# go through all examples (predict randomly)
	# count accuracy = (correct predictions) / m
	accuracy_count = 0
	for i in range(m):
		prediction = random.choice([1, 2, 3, 4, 5])
		if Y[i] == prediction:
			accuracy_count += 1
	# return accuracy = (accuracy_count / m) * 100
	return (accuracy_count * 100) / m

# function to give majority predictions on test data
# X: test data (input, list of list of words)
# Y: test data (output, review rating)
# majority_class: class which occurs most of the times in the training data
def majority_test(X, Y, majority_class):
	# number of test examples
	m = len(Y)
	# go through all examples (predict majority)
	# count accuracy = (accuracy_count / m) * 100
	accuracy_count = 0
	for i in range(m):
		if Y[i] == majority_class:
			accuracy_count += 1
	# return accuracy = (accuracy_count / m) * 100
	return (accuracy_count * 100) / m


print("Parsing JSON (train)")
X, Y = parse_json("Music_Review_train.json")
print("JSON parsed (train)")
print("Processing text (train)")
X, V, size = split_input_advanced(X)
print("Text processed (train)")
print("Vocab Size: " + str(size))
print("Learning model")
phi, theta, majority_class = naive_bayes_train(X, Y, V, size)
print("Model learnt")
print("Parsing JSON (test)")
X, Y = parse_json("Music_Review_test.json")
print("JSON parsed (test)")
print("Processing text (test)")
X, V_, size_ = split_input_advanced(X)
print("Text processed (test)")
print("Testing Model")
accuracy = naive_bayes_test(X, Y, V, size, phi, theta)
print("Model tested")
print("NB Accuracy: " + str(accuracy))
accuracy = random_test(X, Y)
print("Random Accuracy: " + str(accuracy))
accuracy = majority_test(X, Y, majority_class)
print("Majority Accuracy: " + str(accuracy))
