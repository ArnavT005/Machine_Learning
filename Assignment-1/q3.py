import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


# function to return element-wise sigmoid matrix of "matrix"
def sigmoid(matrix):
    # /, + and np.exp do element-wise division, addition and exponentiation respectively
    return (1 / (1 + np.exp(-1 * matrix)))


# function to train a logistic regression model
# input features and target values to be read from "logisticX" and "logisticY" file respectively 
def logistic_regression(logisticX, logisticY):

    # read "linearX" (CSV) file
    df = pd.read_csv(logisticX, header=None)
    # convert into numpy array
    X_train = df.to_numpy()
    
    # store number of training examples (m) and number of input features per example (n)
    m, n = X_train.shape
    
    # determine mean and standard deviation feature-wise (column-wise, axis=0)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)

    # normalize data with mean 0 and variance 1
    # row dimension of X_mean and X_std is broadcasted to match the dimension of X_train
    # / and - carry out element-wise division and subtraction respectively
    X_norm = (X_train - X_mean) / X_std

    # prepend x0 feature to every input feature in the training set to get X matrix (class notation)
    # np.ones creates an array containing only 1s of given size. np.column_stack stacks up columns
    X = np.column_stack((np.ones((m, 1)), X_norm))

    # read "linearY" (CSV) file
    df = pd.read_csv(logisticY, header=None)
    # convert into numpy array Y (class notation)
    Y = df.to_numpy()

    # error handling, dimension check
    # check whether the number of inputs is equal to the number of outputs or not
    if Y.shape[0] != m:
        print("Error: Inconsistent dimensions. Number of inputs is not equal to number of outputs.")
        return None
    else:
        # check whether target value is real or not
        if Y.shape[1] != 1:
            print("Error: Target value is not a real number.")
            return None
    
    # NEWTON'S METHOD STARTS

    # initialize weight parameter (theta) with 0 vector (dimension=(n + 1))
    theta = np.zeros((n + 1, 1))
    
    # set convergence threshold
    epsilon = 1e-4

    # learn parameters (theta) using Newton's method (maximise log likelihood)
    # initial hypothesis function: sigmoid function; @ performs matrix multiplication
    h = sigmoid(X @ theta)
    # since argmax LL = argmin NLL, I will find parameters that minimise NLL (negative log likelihood)
    # initial negative log likelihood function: NLL =  -(Y' @ log(h(x)) + (1 - Y)' @ log(1 - h(X))), where ' denotes transpose of matrix
    # np.log returns element-wise log matrix of input matrix; .T performs transpose
    NLL = -(Y.T @ np.log(h) + (1 - Y).T @ np.log(1 - h))

    # Newton's method loop (repeat until convergence)
    while True:
        # store pre-update NLL for checking convergence post-update
        temp = NLL
        
        # compute gradient of NLL using expression: grad = X' @ (sigmoid(X @ theta) - Y), ' denotes transpose
        grad = X.T @ (h - Y)

        # h_temp is a column vector that stores the element-wise product h(x_i) * (1 - h(x_i)) 
        # * does element-wise multiplication
        h_temp = h * (1 - h)
        # D_h is the diagonal matrix (m by m) where entry in ith row is h(x_i) * (1 - h(x_i))
        # create diagonal matrix out of h_temp
        D_h = np.diagflat(h_temp)
        # compute hessian matrix of NLL using expression: H = X' @ D_h @ X, where ' denotes transpose
        H = X.T @ D_h @ X

        # update theta using Newton's method
        theta = theta - np.linalg.inv(H) @ grad

        # update sigmoid matrix and negative log likelihood
        h = sigmoid(X @ theta)
        NLL = -(Y.T @ np.log(h) + (1 - Y).T @ np.log(1 - h))

        # check for convergence
        if abs(NLL - temp) < epsilon:
            break

    # convergence achieved, parameter is optimized

    # NEWTON'S METHOD ENDS
    
    # return learnt parameter (optimises LL) and other parameters used in plotting
    return theta, X, X_mean, X_std, Y, m

# driver function, parses command line arguments, invokes other methods and plots graph
def main():
    # parse command line arguments
    if len(sys.argv) < 3:
        print("Error: Insufficient number of arguments are provided. Program terminating!")
        return
    if len(sys.argv) > 3:
        print("Warning: Extra command line arguments are provided. Three arguments are expected!")
    
    # store file names containing input and output values respectively
    linearX = sys.argv[1]
    linearY = sys.argv[2]

    # train logisitic regression model
    theta, X, X_mean, X_std, Y, m = logistic_regression(linearX, linearY)
    # check for errors
    if theta.size == None:
        return

    # print theta
    print("Learnt parameter: " + str(theta))

    # filter normalized data for plotting (according to their target values)
    # output_0 store those inputs which correspond to target value 0 and output_1 stores those inputs which correspond to target value 1
    output_0_x1 = []
    output_0_x2 = []
    output_1_x1 = []
    output_1_x2 = []
    for i in range(0, m):
        if Y[i] == 1:
            output_1_x1.append(X[i, 1])
            output_1_x2.append(X[i, 2])
        else:
            output_0_x1.append(X[i, 1])
            output_0_x2.append(X[i, 2])

    # Figure 1: plot training data (normalized) with label, 0 (triangle) and 1 (square)
    plt.figure(1)
    plt.title("Logistic Regression Model")
    # plot points that have target value 0
    plt.scatter(output_0_x1, output_0_x2, marker="^", color="red", label="Target Value = 0")
    # plot points that have target value 1
    plt.scatter(output_1_x1, output_1_x2, marker="s", color="green", label="Target Value = 1")

    # get limits of x-axis (x1), and sample m input points (x1_sample) 
    x1_min, x1_max = plt.xlim()
    x1_sample = np.linspace(x1_min, x1_max, num=m, endpoint=True)

    # plot decision boundary, theta' * x = 0 and provide legend
    if theta[2] == 0:
        if theta[1] == 0:
            # both coefficients are zero
            print("Warning: Hypothesis cannot be plotted. Feature coefficients (parameters) are both zero.")
        else:
            # theta[2] = 0, hence, hypothesis is a vertical line (vline)
            hypothesis = plt.axvline(x=(-theta[0] / theta[1]), label="Decision boundary, $\Theta^T$$x = 0$")
    else:
        # theta[2] is not equal to zero, hence, x2 = (- theta[0] - theta[1] * x1) / theta[2]
        # plot hypothesis using sampled points
        hypothesis, = plt.plot(x1_sample, (-theta[0] -theta[1] * x1_sample) / theta[2], label="Decision boundary, $\Theta^T$$x = 0$")

    # label axes
    plt.xlabel("$x_1$ ($\mu$=" + str(X_mean[0]) + " and $\sigma$=" + str(round(X_std[0], 6)) + ")")
    plt.ylabel("$x_2$ ($\mu$=" + str(X_mean[1]) + " and $\sigma$=" + str(round(X_std[1], 6)) + ")")
    # display legend
    plt.legend()
    # save figure
    plt.savefig("q3plot.jpg")


# run main
main()