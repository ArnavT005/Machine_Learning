import pandas as pd
import numpy as np


# function to give coding to categorical data
# X: numpy array
# type: type of data (string)
def code(X, type):
    # store number of rows
    m = X.shape[0]
    # create new array
    X_ = np.zeros((m, 1), dtype=np.int8)
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
        X_[i][X[i][0]] = 1
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


X, X_hot, Y = parse_csv("bank_train.csv")
print(X_hot.shape)