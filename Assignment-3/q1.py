from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import sys

# create Tree class (for decision tree nodes)
class Tree:
    # constructor
    def __init__(self, attr, val, parent, children):
        self.attr = attr
        self.val = val
        self.parent = parent
        self.children = children
        self.most_common = -1
        self.child_index = -1
    # get methods
    def getValue(self):
        return self.val
    def getAttribute(self):
        return self.attr
    def getParent(self):
        return self.parent
    def getChildren(self):
        return self.children
    def getCommon(self):
        return self.most_common
    def getIndex(self):
        return self.child_index
    # set methods
    def setValue(self, val):
        self.val = val
    def setAttribute(self, attr):
        self.attr = attr
    def setParent(self, parent):
        self.parent = parent
    def addChild(self, child):
        self.children.append(child)
    def setCommon(self, most_common):
        self.most_common = most_common
    def setIndex(self, index):
        self.child_index = index
    # method to check for leaf node
    def isLeaf(self):
        if self.children == []:
            return True
        else:
            return False
    # method to check for root node
    def isRoot(self):
        if self.parent == None:
            return True
        else:
            return False

# function to give coding to categorical data
# X: numpy array
# type: type of data (string)
def code(X, type):
    # store number of rows
    m = X.shape[0]
    # create new array
    X_ = np.zeros((m, 1))
    if type == "job":
        for i in range(m):
            if X[i][0] == 'admin.':
                X_[i][0] = 1
            elif X[i][0] == 'blue-collar':
                X_[i][0] = 2
            elif X[i][0] == 'entrepreneur':
                X_[i][0] = 3
            elif X[i][0] == 'housemaid':
                X_[i][0] = 4
            elif X[i][0] == 'management':
                X_[i][0] = 5
            elif X[i][0] == 'retired':
                X_[i][0] = 6
            elif X[i][0] == 'self-employed':
                X_[i][0] = 7
            elif X[i][0] == 'services':
                X_[i][0] = 8
            elif X[i][0] == 'student':
                X_[i][0] = 9
            elif X[i][0] == 'technician':
                X_[i][0] = 10
            elif X[i][0] == 'unemployed':
                X_[i][0] = 11
            else:
                # unknown
                X_[i][0] = 0
    elif type == "mar":
        for i in range(m):
            if X[i][0] == 'divorced':
                X_[i][0] = 1
            elif X[i][0] == 'married':
                X_[i][0] = 2
            elif X[i][0] == 'single':
                X_[i][0] = 3
            else:
                # unknown
                X_[i][0] = 0
    elif type == "edu":
        for i in range(m):
            if X[i][0] == 'primary':
                X_[i][0] = 1
            elif X[i][0] == 'secondary':
                X_[i][0] = 2
            elif X[i][0] == 'tertiary':
                X_[i][0] = 3
            else:
                # unknown
                X_[i][0] = 0
    elif type in ["def", "hou", "loa"]:
        for i in range(m):
            if X[i][0] == 'no':
                X_[i][0] = 1
            elif X[i][0] == 'yes':
                X_[i][0] = 2
            else:
                # unknown
                X_[i][0] = 0
    elif type == "con":
        for i in range(m):
            if X[i][0] == 'cellular':
                X_[i][0] = 1
            elif X[i][0] == 'telephone':
                X_[i][0] = 2
            else:
                # unknown
                X_[i][0] = 0
    elif type == "mon":
        for i in range(m):
            if X[i][0] == 'feb':
                X_[i][0] = 1
            elif X[i][0] == 'mar':
                X_[i][0] = 2
            elif X[i][0] == 'apr':
                X_[i][0] = 3
            elif X[i][0] == 'may':
                X_[i][0] = 4
            elif X[i][0] == 'jun':
                X_[i][0] = 5
            elif X[i][0] == 'jul':
                X_[i][0] = 6
            elif X[i][0] == 'aug':
                X_[i][0] = 7
            elif X[i][0] == 'sep':
                X_[i][0] = 8
            elif X[i][0] == 'oct':
                X_[i][0] = 9
            elif X[i][0] == 'nov':
                X_[i][0] = 10
            elif X[i][0] == 'dec':
                X_[i][0] = 11
            else:
                # january
                X_[i][0] = 0
    elif type == "pou":
        for i in range(m):
            if X[i][0] == 'success':
                X_[i][0] = 1
            elif X[i][0] == 'failure':
                X_[i][0] = 2
            elif X[i][0] == 'other':
                X_[i][0] = 3
            else:
                # unknown
                X_[i][0] = 0
    elif type == "y":
        for i in range(m):
            if X[i][0] == 'yes':
                X_[i][0] = 1
            else:
                # no
                X_[i][0] = 0
    # return encoded data
    return X_

# function to give one-hot encoding for a feature vector (column)
# X: column vector (numpy array) (integer coded)
# k: range of values (0 - (k - 1))
def one_hot_encoding(X, k):
    # store number of rows in X
    m = X.shape[0]
    # create new feature matrix
    X_ = np.zeros((m, k))
    # go through all examples
    for i in range(m):
        # activate appropriate entry in X_
        X_[i][int(X[i][0])] = 1
    # return encoded matrix
    return X_

# parse data (CSV) and generate data set (input and output)
# parameter file stores the name of the file
def parse_csv(file):
    # read csv file (header is present)
    df = pd.read_csv(file, sep=';')
    # get different features (convert to numpy arrays)
    # code categorical data into integers (0, 1, 2, ..)
    # also generate one-hot encoded data (for categorical features)
    X_age = df['age'].to_numpy().reshape((-1, 1))
    X_job = code(df['job'].to_numpy().reshape((-1, 1)), "job")
    X_job_hot = one_hot_encoding(X_job, 12) 
    X_mar = code(df['marital'].to_numpy().reshape((-1, 1)), "mar")
    X_mar_hot = one_hot_encoding(X_mar, 4)
    X_edu = code(df['education'].to_numpy().reshape((-1, 1)), "edu")
    X_edu_hot = one_hot_encoding(X_edu, 4)
    X_def = code(df['default'].to_numpy().reshape((-1, 1)), "def")
    X_def_hot = one_hot_encoding(X_def, 3)
    X_bal = df['balance'].to_numpy().reshape((-1, 1))
    X_hou = code(df['housing'].to_numpy().reshape((-1, 1)), "hou")
    X_hou_hot = one_hot_encoding(X_hou, 3)
    X_loa = code(df['loan'].to_numpy().reshape((-1, 1)), "loa")
    X_loa_hot = one_hot_encoding(X_loa, 3)
    X_con = code(df['contact'].to_numpy().reshape((-1, 1)), "con")
    X_con_hot = one_hot_encoding(X_con, 3)
    X_day = df['day'].to_numpy().reshape((-1, 1))
    X_mon = code(df['month'].to_numpy().reshape((-1, 1)), "mon")
    X_mon_hot = one_hot_encoding(X_mon, 12)
    X_dur = df['duration'].to_numpy().reshape((-1, 1))
    X_cam = df['campaign'].to_numpy().reshape((-1, 1))
    X_pda = df['pdays'].to_numpy().reshape((-1, 1))
    X_pre = df['previous'].to_numpy().reshape((-1, 1))
    X_pou = code(df['poutcome'].to_numpy().reshape((-1, 1)), "pou")
    X_pou_hot = one_hot_encoding(X_pou, 4)
    # column stack all features to get the input matrix, X (not one-hot encoded)
    X = np.column_stack((X_age, X_job, X_mar, X_edu, X_def, X_bal, X_hou, X_loa, X_con, X_day, X_mon, X_dur, X_cam, X_pda, X_pre, X_pou))
    # column stack all one-hot encoded features to get input matrix, X_hot
    X_hot = np.column_stack((X_age, X_job_hot, X_mar_hot, X_edu_hot, X_def_hot, X_bal, X_hou_hot, X_loa_hot, X_con_hot, X_day, X_mon_hot, X_dur, X_cam, X_pda, X_pre, X_pou_hot))
    # code output ('yes', 'no') to (1, 0)
    Y = code(df['y'].to_numpy().reshape((-1, 1)), "y")
    # return the data (input/output)
    return X, X_hot, Y

# function to split data on a given attribute
# X: input data
# Y: output data
# attr_index: attribute index to split on
# num_split: whether the attribute to split on is numeric
# cat_range: range in case of categorical data
def split_data(X, Y, attr_index, num_split, cat_range=0):
    # create lists
    X_list = []
    Y_list = []
    m = X.shape[0]
    sum = 0
    if num_split:
        # numeric attribute, binary split
        # find median value of 'attr_index' attribute
        median = np.median(X[:, attr_index].reshape(-1))
        # find examples strictly greater than the median
        filter = (X[:, attr_index] > median)
        # store data with attr_value <= median
        X_list.append(X[filter==False])
        Y_list.append(Y[filter==False])
        # store data with attr_value > median
        X_list.append(X[filter])
        Y_list.append(Y[filter])
        sum += len(X_list[0]) + len(X_list[1])
    else:
        # cat_range-way split
        for i in range(cat_range):
            # filter examples with attr_value i
            filter = (X[:, attr_index] == i)
            # store data with attr_value = i    
            X_list.append(X[filter])
            Y_list.append(Y[filter])
            sum += len(X_list[i])
    assert m == sum
    # return data lists
    return X_list, Y_list

# function to choose the best attribute to split the data on
# X: input data
# Y: output data
# num_attr: set of attribute indices with numeric values
# cat_attr: dictionary containing range of categorical attributes
def best_attribute_split(X, Y, num_attr, cat_attr):
    # store number of examples and features
    m, n = X.shape
    # go through all features and choose the one that maximizes information gain
    min_entropy = -1
    min_index = -1
    for attr_index in range(n):
        if attr_index in num_attr:
            # numeric attribute
            X_list, Y_list = split_data(X, Y, attr_index, True)
        else:
            # categorical attribute
            X_list, Y_list = split_data(X, Y, attr_index, False, cat_attr[attr_index])
        # determine split entropy
        entropy = 0
        for i in range(len(X_list)):
            # store number of examples
            temp = X_list[i].shape[0]
            # compute prob of example having this attr value
            prob = (temp / m)
            # check for zero probability
            if prob == 0:
                continue
            # compute H(y|X=x)
            one_count = np.sum(Y_list[i], axis=0)[0]
            # check for zero entropy
            if one_count == 0 or one_count == temp:
                continue
            h = (one_count / temp) * np.log(temp / one_count) + ((temp - one_count) / temp) * np.log(temp / (temp - one_count))
            # add to entropy (after prior multiplication)
            entropy += (h * prob)
        # check for minimum entropy
        if min_index == -1:
            min_index = attr_index
            min_entropy = entropy
        elif entropy < min_entropy:
            min_index = attr_index
            min_entropy = entropy
    # return attribute that minimizes entropy (maximizes information gain)
    return min_index

# recursive function used to create the decision tree (binary classification)
# X: incoming input (training data)
# Y: incoming output (training data)
# num_attr: set of attribute indices with numeric values
# cat_attr: dictionary containing range of categorical attributes
# curr_depth: depth of data in decison tree
# max_depth: maximum depth of any leaf node, -1 indicates infinity
def decision_tree(X, Y, num_attr, cat_attr, curr_depth, max_depth=-1):
    # store number of examples
    m = X.shape[0]
    # return None, if data set is empty
    if m == 0:
        return None
    # data not empty, determine number of examples of class 1
    checksum = np.sum(Y, axis=0)[0]
    # check if data is homogeneous/pure
    if checksum == 0:
        # all examples have zero output
        # return a leaf node (val=0)
        node = Tree(-1, 0, None, [])
        node.setCommon(0)
        return node
    elif checksum == m:
        # all examples have 'one' output
        # return a leaf node (val=1)
        node = Tree(-1, 1, None, [])
        node.setCommon(1)
        return node
    if curr_depth == max_depth:
        # max_depth reached, return leaf node with majority value
        if checksum > m - checksum:
            # 1 is the majority class
            node = Tree(-1, 1, None, [])
            node.setCommon(1)
            return node
        else:
            # 0 is the majority class
            node = Tree(-1, 0, None, [])
            node.setCommon(0)
            return node
    # else, the data is not homogeneous and not empty, and max_depth is not reached
    # create internal node
    node = Tree(-1, -1, None, [])
    # set most-common class (used later in post-pruning)
    if checksum > m - checksum:
        node.setCommon(1)
    else:
        node.setCommon(0)
    # choose an attribute to split on
    attr_index = best_attribute_split(X, Y, num_attr, cat_attr)
    # set node attribute
    node.setAttribute(attr_index)
    if attr_index in num_attr:
        # attribute is numeric, perform 2-way split on median
        X_list, Y_list = split_data(X, Y, attr_index, True)
        # find median value, and set it as node value
        median = np.median(X[:, attr_index].reshape(-1))
        node.setValue(median)
        # create sub-trees
        for i in range(len(X_list)):
            if X_list[i].shape[0] == 0:
                # no data for this split, create majority leaf
                child = Tree(-1, node.getCommon(), None, [])
                child.setCommon(node.getCommon())
                child.setParent(node)
                child.setIndex(i)
                node.addChild(child)
            else:
                # build children recursively
                child = decision_tree(X_list[i], Y_list[i], num_attr, cat_attr, curr_depth + 1, max_depth)
                child.setParent(node)
                child.setIndex(i)
                node.addChild(child)
    else:
        # attribute is categorical, perform split on every feature value
        X_list, Y_list = split_data(X, Y, attr_index, False, cat_attr[attr_index])
        # create sub-trees
        for i in range(len(X_list)):
            if X_list[i].shape[0] == 0:
                # no data for this split, create majority leaf
                child = Tree(-1, node.getCommon(), None, [])
                child.setCommon(node.getCommon())
                child.setParent(node)
                child.setIndex(i)
                node.addChild(child)
            else:
                # build children recursively
                child = decision_tree(X_list[i], Y_list[i], num_attr, cat_attr, curr_depth + 1, max_depth)
                child.setParent(node)
                child.setIndex(i)
                node.addChild(child)
    # return the node created
    return node    

# function to traverse decision tree for a given example
# tree: decision tree
# X: example
# num_attr: set of attribute indices with numeric values
# node: node to be pruned
def traverse_tree(tree, X, num_attr, node=None):
    # check if it is a leaf node
    if tree.isLeaf():
        return tree.getValue()
    # check if it is the node to be pruned
    if tree == node:
        # return most common output
        return tree.getCommon()
    # else, it is an internal node
    # get attribute index and child nodes
    attr_index = tree.getAttribute()
    children = tree.getChildren()
    if attr_index in num_attr:
        # numeric attribute, split on value (median)
        if X[attr_index] > tree.getValue():
            # traverse right sub-tree
            return traverse_tree(children[1], X, num_attr, node)
        else:
            # traverse left sub-tree
            return traverse_tree(children[0], X, num_attr, node)
    else:
        # categorical attribute
        # traverse sub-tree corresponding to X[attr_index]
        return traverse_tree(children[int(X[attr_index])], X, num_attr, node) 

# function to test decision tree model
# tree: decision tree
# X: test data (input)
# Y: test data (output)
# num_attr: set of attribute indices with numeric values
# node: node to be pruned
def test_tree(tree, X, Y, num_attr, node=None):
    # store number of examples
    m = X.shape[0]
    # initialize accuracy to zero
    accuracy = 0
    # go through every example
    for i in range(m):
        # make prediction by traversing tree
        prediction = traverse_tree(tree, X[i, :], num_attr, node)
        if prediction == Y[i][0]:
            # increment accuracy
            accuracy += 1
    # return test accuracy
    return (accuracy * 100) / m

# function to perform level-order traversal of a decision tree
# tree: decision tree
def level_order_traversal(tree):
    # initialise node-list
    node_list = []
    # initialise leaf count
    leaf_count = 0
    # use lists for traversal
    curr_level = [tree]
    next_level = []
    while curr_level != []:    
        while curr_level != []:
            node = curr_level.pop()
            # check for leaf node
            if node.isLeaf():
                leaf_count += 1
                continue
            # else, store it in list
            node_list.append(node)
            # append children to next_level
            children = node.getChildren()
            for child in children:
                next_level.append(child)
        # make next level the current level
        curr_level = next_level
        next_level = []
    # return node list and leaf count
    return node_list, leaf_count

# function to post-prune a decision tree to avoid overfitting (reduced error pruning)
# tree: decision tree
# X_train: train data (input)
# Y_train: train data (output)
# X_val: validation data (input)
# Y_val: validation data (output)
# X_test: test data (input)
# Y_test: test data (output)
# num_attr: set of attribute indices with numeric values
# node_list: list of tree nodes in bottom-up manner
# num_nodes_internal: list of number of internal nodes
# num_nodes_total: list of total number of nodes
# train_accuracy: list of training accuracy values
# val_accuracy: list of validation accuracy values
# test_accuracy: list of test accuracy values
def post_prune_tree(tree, X_train, Y_train, X_val, Y_val, X_test, Y_test, num_attr, node_list, num_nodes_internal, num_nodes_total, train_accuracy, val_accuracy, test_accuracy):
    # iterate and prune, if improvement
    for node in node_list:
        pre_prune_accuracy = test_tree(tree, X_val, Y_val, num_attr)
        post_prune_accuracy = test_tree(tree, X_val, Y_val, num_attr, node)
        # prune node if post accuracy is at least as much as pre accuracy
        if post_prune_accuracy >= pre_prune_accuracy:
            # prune node
            new_node = Tree(-1, -1, None, [])
            new_node.setValue(node.getCommon())
            new_node.setCommon(node.getCommon())
            if node.isRoot():
                # prune root
                tree = new_node
            else:
                # get parent and index
                node_parent = node.getParent()
                index = node.getIndex()
                # set parent and index
                new_node.setParent(node_parent)
                new_node.setIndex(index)
                # set child
                node_parent.children[index] = new_node
                # remove old node from tree
                node.setParent(None)
            nodes, leaf_count = level_order_traversal(tree)
            num_nodes_internal.append(len(nodes))
            num_nodes_total.append(len(nodes) + leaf_count)
            train_accuracy.append(test_tree(tree, X_train, Y_train, num_attr))
            val_accuracy.append(post_prune_accuracy)
            test_accuracy.append(test_tree(tree, X_test, Y_test, num_attr))
    # tree is suitably pruned
    return tree

# scoring function used to find out-of-bag accuracy (for parameter tuning)
# X: input data (one-hot encoded)
# Y: output data (one-hot encoded)
def oob_scorer(estimator, X, Y):
    estimator.fit(X, Y)
    return estimator.oob_score_

# function to train a random forest using sklearn module and find best hyperparameter using GridSearch
# X_train: training data (input)
# Y_train: training data (output)
# all data is one-hot encoded
# param_grid: parameter grid over which the search is to be conducted
def random_forest_tuning(X_train, Y_train, param_grid):
    # define estimator object (multi-processing)
    forest = RandomForestClassifier(criterion='entropy', oob_score=True, n_jobs=-1)
    # create grid search object (5-fold cross validation)
    clf = GridSearchCV(forest, param_grid, scoring=oob_scorer, n_jobs=-1, verbose=4)
    # fit estimator using all available options and find the best fit (parameters)
    clf.fit(X_train, Y_train.reshape(-1))
    # return the best parameters and the estimator learnt using these params
    return clf.best_params_, clf.best_estimator_

# driver function
def main():
    # get file names
    file_train = sys.argv[1]
    file_test = sys.argv[2]
    file_val = sys.argv[3]
    # load data (both normal and one-hot encoded)
    X_train, X_train_hot, Y_train = parse_csv(file_train)
    X_val, X_val_hot, Y_val = parse_csv(file_val)
    X_test, X_test_hot, Y_test = parse_csv(file_test)
    # get part number
    part_num = sys.argv[4]
    # define tree related constants (like which attributes are numerical, range of categorical data etc.)
    # for normal encoding
    num_attr = set([0, 5, 9, 11, 12, 13, 14])
    cat_attr = {1: 12, 2: 4, 3: 4, 4: 3, 6: 3, 7: 3, 8: 3, 10: 12, 15: 4}
    # for one-hot encoding
    num_attr_hot = set([0, 24, 34, 47, 48, 49, 50])
    cat_attr_hot = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 
                    19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 35: 2, 36: 2, 37: 2, 
                    38: 2, 39: 2, 40: 2, 41: 2, 42: 2, 43: 2, 44: 2, 45: 2, 46: 2, 51: 2, 52: 2, 53: 2, 54: 2}
    # define parameter grid, for part 'c' and 'd'
    param_grid = {
        'n_estimators': [50, 150, 250, 350, 450],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_samples_split': [2, 4, 6, 8, 10]
    }
    if part_num == 'a':
        # construct a decision tree using both normal and one-hot encodings
        # calculate train/val/test accuracy as a normal-tree is grown
        depth = [i for i in range(20)]
        num_nodes_total = []
        num_nodes_internal = []
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        for d in depth:
            tree = decision_tree(X_train, Y_train, num_attr, cat_attr, 0, d)
            node_list, leaf_count = level_order_traversal(tree)
            num_nodes_internal.append(len(node_list))
            num_nodes_total.append(len(node_list) + leaf_count)
            train_accuracy.append(test_tree(tree, X_train, Y_train, num_attr))
            val_accuracy.append(test_tree(tree, X_val, Y_val, num_attr))
            test_accuracy.append(test_tree(tree, X_test, Y_test, num_attr))
        # print final accuracies
        print("Without using one-hot encoding:")
        print("Training Accuracy (in %): " + str(train_accuracy[-1]))
        print("Validation Accuracy (in %): " + str(val_accuracy[-1]))
        print("Test Accuracy (in %): " + str(test_accuracy[-1]))
        # plot graphs
        plt.figure(1)
        plt.plot(num_nodes_internal, train_accuracy, label='Training accuracy')
        plt.plot(num_nodes_internal, val_accuracy, label='Validation accuracy')
        plt.plot(num_nodes_internal, test_accuracy, label='Test accuracy')
        plt.xlabel("Number of decision (internal) nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("accuracy_internal.png")
        plt.figure(2)
        plt.plot(num_nodes_total, train_accuracy, label='Training accuracy')
        plt.plot(num_nodes_total, val_accuracy, label='Validation accuracy')
        plt.plot(num_nodes_total, test_accuracy, label='Test accuracy')
        plt.xlabel("Total number of nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("accuracy_total.png")
        # calculate train/val/test accuracy as a one-hot-encoded-tree is grown
        depth = [i for i in range(36)]
        num_nodes_internal_hot = []
        num_nodes_total_hot = []
        train_accuracy_hot, val_accuracy_hot, test_accuracy_hot = [], [], []
        for d in depth:
            tree_hot = decision_tree(X_train_hot, Y_train, num_attr_hot, cat_attr_hot, 0, d)
            node_list, leaf_count = level_order_traversal(tree_hot)
            num_nodes_internal_hot.append(len(node_list))
            num_nodes_total_hot.append(len(node_list) + leaf_count)
            train_accuracy_hot.append(test_tree(tree_hot, X_train_hot, Y_train, num_attr_hot))
            val_accuracy_hot.append(test_tree(tree_hot, X_val_hot, Y_val, num_attr_hot))
            test_accuracy_hot.append(test_tree(tree_hot, X_test_hot, Y_test, num_attr_hot))
        # print final accuracies
        print("Using one-hot encoding:")
        print("Training Accuracy (in %): " + str(train_accuracy_hot[-1]))
        print("Validation Accuracy (in %): " + str(val_accuracy_hot[-1]))
        print("Test Accuracy (in %): " + str(test_accuracy_hot[-1]))
        # plot graphs
        plt.figure(3)
        plt.plot(num_nodes_internal_hot, train_accuracy_hot, label='Training accuracy')
        plt.plot(num_nodes_internal_hot, val_accuracy_hot, label='Validation accuracy')
        plt.plot(num_nodes_internal_hot, test_accuracy_hot, label='Test accuracy')
        plt.xlabel("Number of decision (internal) nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("accuracy_internal_hot.png")
        plt.figure(4)
        plt.plot(num_nodes_total_hot, train_accuracy_hot, label='Training accuracy')
        plt.plot(num_nodes_total_hot, val_accuracy_hot, label='Validation accuracy')
        plt.plot(num_nodes_total_hot, test_accuracy_hot, label='Test accuracy')
        plt.xlabel("Total number of nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("accuracy_total_hot.png")
    elif part_num == 'b':
        # first grow trees using both normal and one-hot encoding
        # without using one-hot encoding
        tree = decision_tree(X_train, Y_train, num_attr, cat_attr, 0)
        node_list, leaf_count = level_order_traversal(tree)
        node_list = list(reversed(node_list))
        num_nodes_internal = [len(node_list)]
        num_nodes_total = [len(node_list) + leaf_count]
        train_accuracy = [test_tree(tree, X_train, Y_train, num_attr)]
        val_accuracy = [test_tree(tree, X_val, Y_val, num_attr)]
        test_accuracy = [test_tree(tree, X_test, Y_test, num_attr)]
        # prune tree
        tree = post_prune_tree(tree, X_train, Y_train, X_val, Y_val, X_test, Y_test, num_attr, node_list, num_nodes_internal, num_nodes_total, train_accuracy, val_accuracy, test_accuracy)
        # print final accuracies
        print("Without using one-hot encoding:")
        print("Training Accuracy (in %): " + str(train_accuracy[-1]))
        print("Validation Accuracy (in %): " + str(val_accuracy[-1]))
        print("Test Accuracy (in %): " + str(test_accuracy[-1]))
        # plot graphs
        plt.figure(1)
        plt.plot(num_nodes_internal, train_accuracy, label='Training accuracy')
        plt.plot(num_nodes_internal, val_accuracy, label='Validation accuracy')
        plt.plot(num_nodes_internal, test_accuracy, label='Test accuracy')
        plt.xlabel("Number of decision (internal) nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("prune_accuracy_internal.png")
        plt.figure(2)
        plt.plot(num_nodes_total, train_accuracy, label='Training accuracy')
        plt.plot(num_nodes_total, val_accuracy, label='Validation accuracy')
        plt.plot(num_nodes_total, test_accuracy, label='Test accuracy')
        plt.xlabel("Total number of nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("prune_accuracy_total.png")
        # using one-hot encoding
        tree_hot = decision_tree(X_train_hot, Y_train, num_attr_hot, cat_attr_hot, 0)
        node_list, leaf_count = level_order_traversal(tree_hot)
        node_list = list(reversed(node_list))
        num_nodes_internal_hot = [len(node_list)]
        num_nodes_total_hot = [len(node_list) + leaf_count]
        train_accuracy_hot = [test_tree(tree_hot, X_train_hot, Y_train, num_attr_hot)]
        val_accuracy_hot = [test_tree(tree_hot, X_val_hot, Y_val, num_attr_hot)]
        test_accuracy_hot = [test_tree(tree_hot, X_test_hot, Y_test, num_attr_hot)]
        # prune tree
        tree_hot = post_prune_tree(tree_hot, X_train_hot, Y_train, X_val_hot, Y_val, X_test_hot, Y_test, num_attr_hot, node_list, num_nodes_internal_hot, num_nodes_total_hot, train_accuracy_hot, val_accuracy_hot, test_accuracy_hot)
        # print final accuracies
        print("Using one-hot encoding:")
        print("Training Accuracy (in %): " + str(train_accuracy_hot[-1]))
        print("Validation Accuracy (in %): " + str(val_accuracy_hot[-1]))
        print("Test Accuracy (in %): " + str(test_accuracy_hot[-1]))
        # plot graphs
        plt.figure(3)
        plt.plot(num_nodes_internal_hot, train_accuracy_hot, label='Training accuracy')
        plt.plot(num_nodes_internal_hot, val_accuracy_hot, label='Validation accuracy')
        plt.plot(num_nodes_internal_hot, test_accuracy_hot, label='Test accuracy')
        plt.xlabel("Number of decision (internal) nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("prune_accuracy_internal_hot.png")
        plt.figure(4)
        plt.plot(num_nodes_total_hot, train_accuracy_hot, label='Training accuracy')
        plt.plot(num_nodes_total_hot, val_accuracy_hot, label='Validation accuracy')
        plt.plot(num_nodes_total_hot, test_accuracy_hot, label='Test accuracy')
        plt.xlabel("Total number of nodes")
        plt.ylabel("Prediction accuracy (in %)")
        plt.legend()
        plt.savefig("prune_accuracy_total_hot.png")
    elif part_num == 'c':
        best_params, best_estimator = random_forest_tuning(X_train_hot, Y_train, param_grid)
        # write best parameters to a json file, so that these can be used for parameter sensitivity analysis
        with open("params.json", 'w') as file:
            json.dump(best_params, file)
            file.write('\n')
        file.close()
        # print best parameters found
        print("Best n_estimators: " + str(best_params['n_estimators']))
        print("Best max_features: " + str(best_params['max_features']))
        print("Best min_samples_split: " + str(best_params['min_samples_split']))
        # report out-of-bag accuracy
        print("Out-of-Bag accuracy: (in %)" + str(best_estimator.oob_score_ * 100))
        # do prediction on training set
        Y_predict = best_estimator.predict(X_train_hot).reshape((-1, 1))
        accuracy = np.sum((Y_predict == Y_train) * 1) / X_train_hot.shape[0]
        print("Training Accuracy (in %): " + str(accuracy * 100))
        Y_predict = best_estimator.predict(X_val_hot).reshape((-1, 1))
        accuracy = np.sum((Y_predict == Y_val) * 1) / X_val_hot.shape[0]
        print("Validation Accuracy (in %): " + str(accuracy * 100))
        Y_predict = best_estimator.predict(X_test_hot).reshape((-1, 1))
        accuracy = np.sum((Y_predict == Y_test) * 1) / X_test_hot.shape[0]
        print("Test Accuracy (in %): " + str(accuracy * 100))
    elif part_num == 'd':
        # load best parameters
        with open("params.json", 'r') as file:
            best_params = json.load(file) 
        best_n_estimators = best_params['n_estimators']
        best_max_features = best_params['max_features']
        best_min_samples_split = best_params['min_samples_split']
        # parameter sensitivity analysis
        # changing n_estimators
        val_list, test_list = [], []
        oob_list = []
        for param in param_grid['n_estimators']:
            # train forest
            forest = RandomForestClassifier(n_estimators=param, criterion='entropy', min_samples_split=best_min_samples_split, max_features=best_max_features, oob_score=True, n_jobs=-1)
            forest.fit(X_train_hot, Y_train.reshape(-1))
            # determine validation accuracy
            Y_predict = forest.predict(X_val_hot).reshape((-1, 1))
            accuracy = np.sum((Y_predict == Y_val) * 1) / X_val_hot.shape[0]
            val_list.append(accuracy * 100)
            # determine test accuracy
            Y_predict = forest.predict(X_test_hot).reshape((-1, 1))
            accuracy = np.sum((Y_predict == Y_test) * 1) / X_test_hot.shape[0]
            test_list.append(accuracy * 100)
        # plot graph
        plt.figure(1)
        plt.plot(param_grid['n_estimators'], val_list, label='Validation Accuracy')
        plt.plot(param_grid['n_estimators'], test_list, label='Test Accuracy')
        plt.xlabel("Value (n_estimators)")
        plt.ylabel("Prediction Accuracy (in %)")
        plt.legend()
        plt.savefig("n_estimators_sensitivity.png")
        # changing max_features
        val_list, test_list = [], []
        for param in param_grid['max_features']:
            # train forest
            forest = RandomForestClassifier(n_estimators=best_n_estimators, criterion='entropy', min_samples_split=best_min_samples_split, max_features=param, oob_score=True, n_jobs=-1)
            forest.fit(X_train_hot, Y_train.reshape(-1))
            # determine validation accuracy
            Y_predict = forest.predict(X_val_hot).reshape((-1, 1))
            accuracy = np.sum((Y_predict == Y_val) * 1) / X_val_hot.shape[0]
            val_list.append(accuracy * 100)
            # determine test accuracy
            Y_predict = forest.predict(X_test_hot).reshape((-1, 1))
            accuracy = np.sum((Y_predict == Y_test) * 1) / X_test_hot.shape[0]
            test_list.append(accuracy * 100)
        # plot graph
        plt.figure(2)
        plt.plot(param_grid['max_features'], val_list, label='Validation Accuracy')
        plt.plot(param_grid['max_features'], test_list, label='Test Accuracy')
        plt.xlabel("Value (max_features)")
        plt.ylabel("Prediction Accuracy (in %)")
        plt.legend()
        plt.savefig("max_features_sensitivity.png")
        # changing min_samples_split
        val_list, test_list = [], []
        for param in param_grid['min_samples_split']:
            # train forest
            forest = RandomForestClassifier(n_estimators=best_n_estimators, criterion='entropy', min_samples_split=param, max_features=best_max_features, oob_score=True, n_jobs=-1)
            forest.fit(X_train_hot, Y_train.reshape(-1))
            # determine validation accuracy
            Y_predict = forest.predict(X_val_hot).reshape((-1, 1))
            accuracy = np.sum((Y_predict == Y_val) * 1) / X_val_hot.shape[0]
            val_list.append(accuracy * 100)
            # determine test accuracy
            Y_predict = forest.predict(X_test_hot).reshape((-1, 1))
            accuracy = np.sum((Y_predict == Y_test) * 1) / X_test_hot.shape[0]
            test_list.append(accuracy * 100)
        plt.figure(3)
        plt.plot(param_grid['min_samples_split'], val_list, label='Validation Accuracy')
        plt.plot(param_grid['min_samples_split'], test_list, label='Test Accuracy')
        plt.xlabel("Value (min_samples_split)")
        plt.ylabel("Prediction Accuracy (in %)")
        plt.legend()
        plt.savefig("min_samples_split_sensitivity.png")

main()         