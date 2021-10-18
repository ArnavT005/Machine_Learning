import pandas as pd
import numpy as np


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
# k: range of values (0-k)
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
    else:
        # cat_range-way split
        for i in range(cat_range):
            # filter examples with attr_value i
            filter = (X[:, attr_index] == i)
            # store data with attr_value = i    
            X_list.append(X[filter])
            Y_list.append(Y[filter])
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
            h = (one_count / temp) * np.log(temp / one_count) + (1 - one_count / temp) * np.log(temp / (temp - one_count))
            # add to entropy
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
def decision_tree(X, Y, num_attr, cat_attr):
    # store number of examples
    m = X.shape[0]
    # determine number of examples of class 1
    checksum = np.sum(Y, axis=0)[0]
    # return leaf-node with majority class, if data set is empty
    if m == 0:
        if checksum > m - checksum:
            # majority class is '1'
            node = Tree(-1, 1, None, [])
            node.setCommon(1)
            return node
        else:
            # majority class is '0'
            node = Tree(-1, 0, None, [])
            node.setCommon(0)
            return node
    # check if data is homogeneous
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
    # else, the data is not homogeneous and not empty
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
        # find median value, and set it as node value
        median = np.median(X[:, attr_index].reshape(-1))
        node.setValue(median)
        # attribute is numeric, perform 2-way split on median
        X_list, Y_list = split_data(X, Y, attr_index, True)
        # create sub-trees
        for i in range(len(X_list)):
            child = decision_tree(X_list[i], Y_list[i], num_attr, cat_attr)
            child.setParent(node)
            child.setIndex(i)
            node.addChild(child)
    else:
        # attribute is categorical, perform split on every feature value
        X_list, Y_list = split_data(X, Y, attr_index, False, cat_attr[attr_index])
        # create sub-trees
        for i in range(len(X_list)):
            child = decision_tree(X_list[i], Y_list[i], num_attr, cat_attr)
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
    # use lists for traversal
    curr_level = [tree]
    next_level = []
    while curr_level != []:    
        while curr_level != []:
            node = curr_level.pop()
            # check for leaf node
            if node.isLeaf():
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
    # return node list
    return node_list

# function to post-prune a decision tree to avoid overfitting (reduced error pruning)
# tree: decision tree
# X: validation data (input)
# Y: validation data (output)
# num_attr: set of attribute indices with numeric values
def post_prune_tree(tree, X, Y, num_attr):
    # create dummy node, used as leaf node during pruning
    leaf = Tree(-1, -1, None, [])
    iter = 0
    # perform pruning till harmful
    # while True:
    iter += 1
    # compute pre-prune validation accuracy
    # pre_prune_accuracy = test_tree(tree, X, Y, num_attr)
    # get node list
    node_list = list(reversed(level_order_traversal(tree)))
    # iterate and prune, if improvement
    for node in node_list:
        pre_prune_accuracy = test_tree(tree, X, Y, num_attr)
        accuracy = test_tree(tree, X, Y, num_attr, node)
        # update max accuracy and node
        if accuracy >= pre_prune_accuracy:
            # prune node
            new_node = Tree(-1, -1, None, [])
            new_node.setValue(node.getCommon())
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
                # remove parent
                node.setParent(None)
        # # choose the node whose pruning leads to maximum increase
        # max_node = None
        # max_accuracy = 0
        # for node in node_list:
        #     # determine post-prune accuracy
        #     accuracy = test_tree(tree, X, Y, num_attr, node)
        #     # update max accuracy and node
        #     if accuracy >= pre_prune_accuracy and accuracy > max_accuracy:
        #         print(accuracy)
        #         max_accuracy = accuracy
        #         max_node = node
        # print("Hello " + str(iter))
        # if max_node == None:
        #     # no helpful pruning possible
        #     break
        # # prune max node
        # new_node = Tree(-1, -1, None, [])
        # new_node.setValue(max_node.getCommon())
        # if max_node.isRoot():
        #     # prune root
        #     tree = new_node
        # else:
        #     # get parent and index
        #     node_parent = max_node.getParent()
        #     index = max_node.getIndex()
        #     # set parent and index
        #     new_node.setParent(node_parent)
        #     new_node.setIndex(index)
        #     # set child
        #     node_parent.children[index] = new_node
        #     # remove parent
        #     max_node.setParent(None)
    # tree is suitably pruned
    return tree

X, X_hot, Y = parse_csv("bank_train.csv")
num_attr = set([0, 5, 9, 11, 12, 13, 14])
cat_attr = {1: 12, 2: 4, 3: 4, 4: 3, 6: 3, 7: 3, 8: 3, 10: 12, 15: 4}
tree = decision_tree(X, Y, num_attr, cat_attr)

X_test, X_test_hot, Y_test = parse_csv("bank_val.csv")
tree = post_prune_tree(tree, X_test, Y_test, num_attr)
X_test, X_test_hot, Y_test = parse_csv("bank_test.csv")
accuracy = test_tree(tree, X_test, Y_test, num_attr)
print(accuracy)


