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
	summary = df["summary"].tolist()
	# return input/output data and summary
	return X, Y, summary

def preprocess_trivial(X):
	table = str.maketrans('', '', string.punctuation)
	X_ = [[] for i in range(len(X))]
	for i in range(len(X)):
		words = [word.translate(table) for word in X[i].split()]
		for word in words:
			if word.isalpha():
				word = word.lower()
				X_[i].append(word)
			else:
				X_[i].append(word)
	return X_

# function to preprocess input data
# X is the list of texts
def preprocess(X):
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
	# return preprocessed data
	return X_

# function to preprocess input data
# filter stop words and perform stemming
# X is the list of texts (input data)
def preprocess_advanced(X):
	# store stop words (english)
	stop_words = set(nltk.corpus.stopwords.words('english'))
	# create stemmer object
	stemmer = nltk.stem.porter.PorterStemmer()
	# initialise new input list (to be returned)
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
	# return preprocessed data
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
		# extend X[i] only if summary is non-empty
		if X[i] == []:
			X[i].extend(summary[i])
			continue
		if num_words_S != 0:
			for j in range(num_words_X // (min(20 * num_words_S, num_words_X))):
				X[i].extend(summary[i])
	# return the extended data
	return X

# create and return vocabulary (V)
# X is the list of texts (input data)
def create_vocabulary(X):
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
# V: word vocabulary (size = |V|)
# phi, theta: learnt parameters (model)
def naive_bayes_test(X, Y, V, size, phi, theta):
	# number of test examples
	m = len(Y)
	# count accuracy = (accuracy_count / m) * 100
	accuracy_count = 0
	# initialise confusion matrix
	confusion_matrix = [[0 for j in range(5)] for i in range(5)]
	# go through every example and make prediction
	for i in range(m):
		# maximise P(x|y=k)P(y=k) over 1 <= k <= 5
		# equivalent to maximising log(P(x|y=k)) + log(P(y=k))
		max_value = 0
		max_class = 0
		for k in range(5):
			# add the prior probability
			temp_value = math.log(phi[k])
			# go through all words in example i
			for word in X[i]:
				if word in V.keys():
					# get word index
					word_index = V[word]
					# add to tempValue
					temp_value += math.log(theta[k][word_index - 1])
				else:
					# word is UNKNOWN, add log(P(UNKNOWN|y=k))
					temp_value += math.log(theta[k][size - 1])
			# check if it is the maximum value
			if max_class == 0:
				max_value = temp_value
				max_class = k + 1
			elif temp_value >= max_value:
				max_value = temp_value
				max_class = k + 1
		# predict class (maxClass)
		if max_class == Y[i]:
			# correct prediction, increase accuracy count
			accuracy_count += 1
		# add entry to confusion matrix
		confusion_matrix[Y[i] - 1][max_class - 1] += 1
	# return accuracy = (accuracy_count / m) * 100 and confusion matrix
	return (accuracy_count * 100) / m, confusion_matrix

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

# function to compute F1-score (macro, weighted, micro)
# confusion_matrix: confusion matrix for the data
def f1_score(confusion_matrix):
	# store number of classes
	k = len(confusion_matrix)
	# determine per-class precision, recall and total count
	precision = [0 for i in range(k)]
	recall = [0 for i in range(k)]
	f1_score = [0 for i in range(k)]
	total_count = [0 for i in range(k)]
	# store true positives (diagonal entries)
	diagonal_count = 0
	for i in range(k):
		# true positives
		TP = confusion_matrix[i][i]
		diagonal_count += TP
		# count total examples which are actually of class (i + 1)
		TP_FN = 0
		# count total examples which are predicted to be of class (i + 1)
		TP_FP = 0
		for j in range(k):
			TP_FN += confusion_matrix[i][j]
			TP_FP += confusion_matrix[j][i]
		# determine precision, recall, f1-score and total count
		precision[i] = TP / TP_FP
		recall[i] = TP / TP_FN
		f1_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
		total_count[i] = TP_FN
	# determine macro-averaged f1-score
	macro_f1 = sum(f1_score) / len(f1_score)
	# determine weighted f1-score
	total_samples = sum(total_count)
	# determine weighted f1-score
	weighted_sum = 0
	for i in range(k):
		weighted_sum += total_count[i] * f1_score[i]
	weighted_f1 = weighted_sum / total_samples
	# determine micro-averaged f1-score
	micro_f1 = diagonal_count / total_samples
	# return all three scores and per-class score
	return f1_score, macro_f1, weighted_f1, micro_f1

print("Parsing JSON (train)")
X, Y, summary = parse_json("Music_Review_train.json")
print("JSON parsed (train)")
print("Processing text (train)")
X = preprocess(X)
summary = preprocess(summary)
print("Text processed (train)")
print("Extending data (train)")
X = extend_data(X, summary)
print("Data extended (train)")
print("Creating vocabulary")
V, size = create_vocabulary(X)
print("Vocabulary created")
print("Vocab Size: " + str(size))
print("Learning model")
phi, theta, majority_class = naive_bayes_train(X, Y, V, size)
print("Model learnt")
print("Parsing JSON (test)")
X, Y, summary = parse_json("Music_Review_test.json")
print("JSON parsed (test)")
print("Processing text (test)")
X = preprocess(X)
summary = preprocess(summary)
print("Text processed (test)")
print("Extending data (test)")
X = extend_data(X, summary)
print("Data extended (test)")
print("Testing Model")
accuracy, confusion_matrix = naive_bayes_test(X, Y, V, size, phi, theta)
print("Model tested")
print("NB Accuracy: " + str(accuracy))
for i in range(5):
	for j in range(5):
		print(confusion_matrix[i][j], end='\t')
	print()
f1_score, macro_f1, weighted_f1, micro_f1 = f1_score(confusion_matrix)
print(f1_score)
print("Macro: " + str(macro_f1))
print("Weighted: " + str(weighted_f1))
print("Micro: " + str(micro_f1))