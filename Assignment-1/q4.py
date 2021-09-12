import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys

# function that maps the output label (Canada/Alaska) to binary label (0/1)
def label_map(s):
    # 0 for Canada and 1 for Alaska
    if s == "Canada":
        return 0
    else:
        return 1

# function to train a model (binary classifier) using gaussian discriminant analysis
# 'x' conditioned on 'y' is assumed to follow normal distribution
# input features and target values to be read from "gdaX" and "gdaY" file respectively (space separated files)
# both linear and quadratic boundaries will be learnt in this function  
def gaussian_discriminant_analysis(gdaX, gdaY):

    # read "gdaX" (space-separated file)
    df = pd.read_csv(gdaX, sep='  ', header=None, engine='python')
    # convert into numpy array
    X_train = df.to_numpy()
    
    # store number of training examples (m) and number of input features per example (n)
    m, n = X_train.shape

    # determine mean and standard deviation feature-wise (column, axis=0)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)

    # normalize data with mean 0 and variance 1
    # row dimension of X_mean and X_std is broadcasted to match the dimension of X_train
    # / and - carry out element-wise division and subtraction respectively
    X = (X_train - X_mean) / X_std

    # read "gdaY" (space-separated file)
    df = pd.read_csv(gdaY, sep=' ', header=None)
    # convert into numpy array containing original labels (Alaska/Canada)
    Y_label = df.to_numpy()
    
    # error handling, dimension check
    # check whether the number of inputs is equal to the number of outputs or not
    if Y_label.shape[0] != m:
        print("Error: Inconsistent dimensions. Number of inputs is not equal to number of outputs.")
        return None
    else:
        # check whether target value is real or not
        if Y_label.shape[1] != 1:
            print("Error: Target value is not a real number.")
            return None

    # map the output values to 0 (Canada) and 1 (Alaska), and store in a new array Y (class notation)
    # vectorize label_map function, binary_label is the vectorized function
    binary_label = np.vectorize(label_map)
    # apply binary on Y_label to get Y
    Y = binary_label(Y_label)

    # store number of 0s and 1s in the output label
    num_1 = Y.sum(axis=0)
    num_0 = m - num_1

    if num_1 == 0 or num_0 == 0:
        # data only belongs to one class, classification not possible
        print("Error: All data belongs to a single class. Classification not possible.")
        return None
    
    # determine phi parameter = num_1 / m
    phi = num_1 / m

    # determine mean vectors, mu_0 and mu_1, using formulae derived in class
    # mu_1 = X' @ Y / num_1 and mu_2 = X' @ (1 - Y) / num_0, ' denotes transpose
    mu_1 = (X.T @ Y) / num_1
    mu_0 = (X.T @ (1 - Y)) / num_0

    # determine mean difference matrices (used for determining covariance matrix below)
    # mu_1 difference matrix = X' - mu_1 (broadcasted)
    mu_1_diff = X.T - mu_1
    # multiplying by indicator function for 1 (element wise multiplication with broadcasted Y')
    mu_1_diff = mu_1_diff * Y.T 
    # mu_0 difference matrix = X' - mu_0 (broadcasted)
    mu_0_diff = X.T - mu_0
    # multiplying by indicator function for 0 (element wise multiplication with (1 - Y)')
    mu_0_diff = mu_0_diff * (1 - Y).T

    # determine covariance matrix, when both classes have the same covariance matrix
    # covariance matrix, sigma = (mu_1_diff @ mu_1_diff' + mu_0_diff @ mu_0_diff') / m, ' denotes transpose
    sigma = (mu_1_diff @ mu_1_diff.T + mu_0_diff @ mu_0_diff.T) / m
    
    # determine covariance matrix, when the classes have differenct covariance matrices
    # covariance matrix, sigma_1 = (mu_1_diff @ mu_1_diff') / num_1
    sigma_1 = (mu_1_diff @ mu_1_diff.T) / num_1
    # covariance matrix, sigma_0 = (mu_0_diff @ mu_0_diff') / num_0
    sigma_0 = (mu_0_diff @ mu_0_diff.T) / num_0

    # all parameters determined, return paramaters plus data that will be used in plotting
    return phi, mu_1, mu_0, sigma_1, sigma_0, sigma, X, X_mean, X_std, m, Y 

# driver function, parses command line arguments, invokes other methods and plots graph
def main():
    # parse command line arguments
    if len(sys.argv) < 3:
        print("Error: Insufficient number of arguments are provided. Program terminating!")
        return
    if len(sys.argv) > 3:
        print("Warning: Extra command line arguments are provided. Three arguments are expected!")
    
    # store file names containing input and output values respectively
    gdaX = sys.argv[1]
    gdaY = sys.argv[2]

    # train GDA model
    try:
        phi, mu_1, mu_0, sigma_1, sigma_0, sigma, X, X_mean, X_std, m, Y  = gaussian_discriminant_analysis(gdaX, gdaY)
    except:
        return

    # print paramters
    print("phi: " + str(phi))
    print("mu_1: " + str(mu_1))
    print("mu_0: " + str(mu_0))
    print("sigma_1: " + str(sigma_1))
    print("sigma_0: " + str(sigma_0))
    print("sigma: " + str(sigma))

    # filter normalized data for plotting (according to their target values)
    # output_0 store those inputs which correspond to target value 0 and output_1 stores those inputs which correspond to target value 1
    output_0_x1 = []
    output_0_x2 = []
    output_1_x1 = []
    output_1_x2 = []
    for i in range(0, m):
        if Y[i] == 1:
            output_1_x1.append(X[i, 0])
            output_1_x2.append(X[i, 1])
        else:
            output_0_x1.append(X[i, 0])
            output_0_x2.append(X[i, 1])

    # Figure 1: plot training data (normalized) with label, linear and quadratic (hyperbolic) boundaries
    fig = plt.figure(1)

    # plot points that have target value 0 (Canada, triangle)
    plt.scatter(output_0_x1, output_0_x2, marker="^", color="red", label="Canada ($P(y = 1|x:\Theta) \leq 0.5$)")
    # plot points that have target value 1 (Alaska, square)
    plt.scatter(output_1_x1, output_1_x2, marker="s", color="green", label="Alaska ($P(y = 1|x:\Theta) > 0.5$)")

    # determine limits of x1 and x2 and sample m points in the given range
    x1_min, x1_max = plt.xlim()
    x1_sample = np.linspace(x1_min, x1_max, num=m, endpoint=True)
    x2_min, x2_max = plt.ylim()
    x2_sample = np.linspace(x2_min, x2_max, num=m, endpoint=True)

    # plot the linear decision boundary, a_*x1 + b_*x2 + C_ = 0 

    # determine inverse of covariance-matrix
    sigma_inv = np.linalg.inv(sigma)
    # determine coefficient of x = (mu_1 - mu_0)' @ sigma_inv, ' denotes transpose
    coefficient = (mu_1 - mu_0).T @ sigma_inv
    # determine "a" and "b"
    a_ = coefficient[0, 0]
    b_ = coefficient[0, 1]
    # determine constant term, C = ((mu_0' @ sigma_inv @ mu_0) - (mu_1' @ sigma_inv @ mu_1)) / 2 + log(phi/(1 - phi))
    C_ = (mu_0.T @ sigma_inv @ mu_0 - mu_1.T @ sigma_inv @ mu_1) / 2 + math.log(phi / (1 - phi))

    # plot decision boundary, a*x1 + b*x2 + C = 0
    if b_ == 0:
        if a_ == 0:
            print("Warning: Hypothesis cannot be plotted. Feature coefficients (parameters) are both zero.")
        else:
            hypothesis = plt.axvline(x=(-C_ / a_), label="Linear boundary, $P(y = 1|x:\Theta) = 0.5$")
    else:
        hypothesis = plt.plot(x1_sample, (-C_[0, 0] - a_ * x1_sample) / b_, label="Linear boundary, $P(y = 1|x:\Theta) = 0.5$")

    # plot the quadratic decision boundary, a*x1^2 + b*x1*x2 + c*x2^2 + d*x1 + e*x2 = C 

    # determine inverse of covariance-matrices
    sigma_1_inv = np.linalg.inv(sigma_1)
    sigma_0_inv = np.linalg.inv(sigma_0)

    # determine quadratic coefficient matrix
    quad_coeff = sigma_1_inv - sigma_0_inv
    # determine individual coefficients, a (x1^2), b (x1*x2), c (x2^2)
    a = quad_coeff[0, 0]
    b = quad_coeff[0, 1] + quad_coeff[1, 0]
    c = quad_coeff[1, 1]

    # determine linear coefficient matrix
    lin_coeff = -2 * (mu_1.T @ sigma_1_inv - mu_0.T @ sigma_0_inv)
    # determine individual coefficients, d (x1), e (x2)
    d = lin_coeff[0, 0]
    e = lin_coeff[0, 1]

    # determine constant term
    # determine determinant of inverse matrices
    det_sigma_1_inv = np.linalg.det(sigma_1_inv)
    det_sigma_0_inv = np.linalg.det(sigma_0_inv)
    # determine C
    C = 2 * math.log(phi / (1 - phi)) + math.log(det_sigma_0_inv / det_sigma_1_inv) + (mu_0.T @ sigma_0_inv @ mu_0) - (mu_1.T @ sigma_1_inv @ mu_1)

    # plot decision boundary, a*x1^2 + b*x1*x2 + c*x2^2 + d*x1 + e*x2 = C 
    if c != 0:
        # use quadratic formula for plotting, c*x2^2 + b*x1*x2 + e*x2 + a*x1^2 + d*x1 = C
        # choose those input samples only for which the discriminant is positive using predicate
        predicate = []
        for x1 in x1_sample:
            if (b * x1 + e) ** 2 >= 4 * c * (a * x1 * x1 + d * x1 - C[0, 0]):
                predicate.append(True)
            else:
                predicate.append(False)
        # filter x1_sample
        x1_sample = x1_sample[predicate]
        # get output, -ve boundary
        x2 = (- (b * x1_sample + e) - np.sqrt((b * x1_sample + e) ** 2 - 4 * c * (a * (x1_sample ** 2) + d * x1_sample - C[0, 0]))) / (2 * c)
        hypothesis = plt.plot(x1_sample, x2, "-y", label="Quadratic boundary, $P(y = 1|x:\Theta) = 0.5$")
        # get output, +ve boundary
        x2 = (- (b * x1_sample + e) + np.sqrt((b * x1_sample + e) ** 2 - 4 * c * (a * (x1_sample ** 2) + d * x1_sample - C[0, 0]))) / (2 * c)
        plt.plot(x1_sample, x2, "-y")

    elif a != 0:
        # use quadratic formula for plotting, a*x1^2 + b*x1*x2 + d*x1 + e*x2 = C
        # choose those input samples only for which the discriminant is positive using predicate
        predicate = []
        for x2 in x2_sample:
            if (b * x2 + d) ** 2 >= 4 * a * (e * x2 - C[0, 0]):
                predicate.append(True)
            else:
                predicate.append(False)
        # filter x1_sample
        x2_sample = x2_sample[predicate]
        # get output, -ve boundary
        x1 = (- (b * x2_sample + d) - np.sqrt((d + b * x2_sample) ** 2 - 4 * a * (e * x2_sample - C[0, 0]))) / (2 * a)
        hypothesis = plt.plot(x1, x2_sample, "-y", label="Quadratic boundary, $P(y = 1|x:\Theta) = 0.5$")
        # get output, +ve boundary
        x1 = (- (b * x2_sample + d) + np.sqrt((d + b * x2_sample) ** 2 - 4 * a * (e * x2_sample - C[0, 0]))) / (2 * a)
        plt.plot(x1, x2_sample, "-y")

    else:
        # both a and c are zero, use relation b*x1*x2 + d*x1 + e*x2 = C for plotting
        # filter out that value of x1 that leads to discontinuity using predicate
        predicate = []
        for x1 in x1_sample:
            if b * x1 + e < 0.0000000001:
                predicate.append(False)
            else:
                predicate.append(True)
        # filter x1_sample
        x1_sample = x1_sample[predicate]
        x2 = (C[0, 0] - d * x1_sample) / (e + b * x1_sample)
        hypothesis = plt.plot(x1_sample, x2, label="Quadratic boundary, $P(y = 1|x:\Theta) = 0.5$")

    # label axes
    plt.xlabel("Ring diameter (Fresh water, $\mu$=" + str(X_mean[0]) + ", $\sigma$=" + str(round(X_std[0], 5)) + ")")
    plt.ylabel("Ring diameter (Marine water, $\mu$=" + str(X_mean[1]) + ", $\sigma$=" + str(round(X_std[1], 5)) + ")")

    # display legend
    plt.legend()
    # save plot
    plt.savefig("q4plot1.jpg")

    # Figure 2: plot training data (normalized) with label, linear and quadratic (single) boundaries (for clarity)
    fig = plt.figure(2)

    # plot points that have target value 0 (Canada, triangle)
    plt.scatter(output_0_x1, output_0_x2, marker="^", color="red", label="Canada ($P(y = 1|x:\Theta) \leq 0.5$)")
    # plot points that have target value 1 (Alaska, square)
    plt.scatter(output_1_x1, output_1_x2, marker="s", color="green", label="Alaska ($P(y = 1|x:\Theta) > 0.5$)")

    # plot the linear decision boundary, a*x1 + b*x2 + C = 0
    # plot decision boundary, a*x1 + b*x2 + C = 0
    if b_ == 0:
        if a_ == 0:
            print("Warning: Hypothesis cannot be plotted. Feature coefficients (parameters) are both zero.")
        else:
            hypothesis = plt.axvline(x=(-C_ / a_), label="Linear boundary, $P(y = 1|x:\Theta) = 0.5$")
    else:
        hypothesis = plt.plot(x1_sample, (-C_[0, 0] - a_ * x1_sample) / b_, label="Linear boundary, $P(y = 1|x:\Theta) = 0.5$")

    # plot the quadratic decision boundary, a*x1^2 + b*x1*x2 + c*x2^2 + d*x1 + e*x2 = C 
    # plot decision boundary, a*x1^2 + b*x1*x2 + c*x2^2 + d*x1 + e*x2 = C 
    if c != 0:
        # use quadratic formula for plotting, c*x2^2 + b*x1*x2 + e*x2 + a*x1^2 + d*x1 = C
        # get output, -ve boundary
        x2 = (- (b * x1_sample + e) - np.sqrt((b * x1_sample + e) ** 2 - 4 * c * (a * (x1_sample ** 2) + d * x1_sample - C[0, 0]))) / (2 * c)
        hypothesis = plt.plot(x1_sample, x2, "-y", label="Quadratic boundary, $P(y = 1|x:\Theta) = 0.5$")
    elif a != 0:
        # use quadratic formula for plotting, a*x1^2 + b*x1*x2 + d*x1 + e*x2 = C
        # get output, +ve boundary
        x1 = (- (b * x2_sample + d) + np.sqrt((d + b * x2_sample) ** 2 - 4 * a * (e * x2_sample - C[0, 0]))) / (2 * a)
        plt.plot(x1, x2_sample, "-y")
    else:
        # both a and c are zero, use relation b*x1*x2 + d*x1 + e*x2 = C for plotting
        x2 = (C[0, 0] - d * x1_sample) / (e + b * x1_sample)
        hypothesis = plt.plot(x1_sample, x2, label="Quadratic boundary, $P(y = 1|x:\Theta) = 0.5$")

    # label axes
    plt.xlabel("Ring diameter (Fresh water, $\mu$=" + str(X_mean[0]) + ", $\sigma$=" + str(round(X_std[0], 5)) + ")")
    plt.ylabel("Ring diameter (Marine water, $\mu$=" + str(X_mean[1]) + ", $\sigma$=" + str(round(X_std[1], 5)) + ")")

    # display legend
    plt.legend()
    # save plot
    plt.savefig("q4plot2.jpg")


# run main
main()