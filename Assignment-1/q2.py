from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# animation function
def animate(num, line, theta_0, theta_1, theta_2):
    # increase line's data set
    line.set_data(theta_0[0, :num], theta_1[0, :num])
    line.set_3d_properties(theta_2[0, :num])
    # return line
    return line

# function used to sample N number of points using theta parameter and noise variance=2
# x1 and x2 are fixed to follow N(3, 4) and N(-1, 4) distributions respectively
def sampling(theta, N):

    # x1_sample and x2_sample store the input features for N sampled examples
    # loc is the mean and scale is the standard deviation
    x1_sample = np.random.normal(loc=3, scale=2, size=(N, 1))
    x2_sample = np.random.normal(loc=-1, scale=2, size=(N, 1))
    
    # stack both features together, column-wise (to generate input values of training data)
    # np.column_stack stacks up columns
    X_train = np.column_stack((x1_sample, x2_sample))

    # sample N noise values, so that the target value can be determined accordingly
    # mean = loc = 0 and std = scale = sqrt(2)
    epsilon = np.random.normal(loc=0, scale=math.sqrt(2), size=(N, 1))

    # prepend x0 feature to every input feature in the training set to get X matrix (class notation)
    X = np.column_stack((np.ones((N, 1)), X_train))

    # get corresponding output (target) values, using the relation: Y = X @ theta + epsilon
    # @ denotes matrix multiplication
    Y = X @ theta + epsilon

    # data sampled, return sampled data; Training Set = (X, Y)
    return (X, Y)


# function to train a linear regression model using Stochastic Gradient Descent (Mini-Batch) for optimisation
# sampled input-output data is provided to this function as Numpy arrays X and Y respectively
# Batch size to be used for mini-batch gradient descent is provided as batch_size parameter
def linear_regression_sgd(X, Y, batch_size):

    # store number of training examples (m) and number of input features per example + 1 (n_ = n + 1, as one additional feature is appended)
    m, n_ = X.shape
    
    # augment output vector with the input matrix so that random shuffle can be done consistently
    Training_Set = np.column_stack((X, Y))

    # STOCHASTIC GRADIENT DESCENT STARTS

    # initialize weight parameter (theta) with 0 vector (dimension=(n + 1 = n_))
    theta = np.zeros((n_, 1))
    
    # set learning rate (0.001) and convergence threshold and flag
    eta, epsilon = 0.001, 0.0001
    converged = False

    # initialise iteration number and number of mini-batches per epoch 
    # it is assumed that batch_size divides the number of training examples
    iteration_number = 0
    num_batch = (m // batch_size)

    # average cost over past 1000-2000 iterations (J_old), and average cost over past 0-1000 iterations (J_new)
    # set J_old to negative infinity for base case
    J_old = float('-inf')
    J_new = 0

    # collect theta_0, theta_1 and theta_2 parameters for plotting
    theta_0 = [0]
    theta_1 = [0]
    theta_2 = [0]

    # SGD loop (repeat until convergence)
    # every iteration of while loop is going to correspond to a single epoch
    # declare convergence when average cost over 1000 iterations will differ by less than epsilon
    # randomly shuffle the training set
    # np.random.shuffle(Training_Set)
    while True:
        # randomly shuffle the training set
        np.random.shuffle(Training_Set)
        
        # do mini-batch  gradient descent
        for i in range(0, num_batch):
            # increment iteration number
            iteration_number = iteration_number + 1

            # select batch_size training examples indexed from i * batch_size to (i + 1) * batch_size - 1
            # slice the training set to get input and output respectively
            X_input = Training_Set[i*batch_size:(i+1)*batch_size, 0:n_]
            Y_output = Training_Set[i*batch_size:(i+1)*batch_size, n_:]

            # compute gradient, grad = (1 / batch_size) * X_input' @ (X_input @ theta - Y_output) 
            grad = (1 / batch_size) * X_input.T @ (X_input @ theta - Y_output)
            
            # update theta, eta is the learning rate
            theta = theta - eta * grad

            # append new values into lists
            theta_0.append(theta[0, 0])
            theta_1.append(theta[1, 0])
            theta_2.append(theta[2, 0])

            # compute batch cost, J = (1/(2*batch_size) (X_input @ theta - Y_output)' @ (X_input @ theta - Y_output)
            # ' denotes transpose, add this cost to J_new
            J_new = J_new + (1 / (2 * batch_size)) * (X_input @ theta - Y_output).T @ (X_input @ theta - Y_output)

            # check convergence every 1000 iterations
            if iteration_number % 1000 == 0:
                # determine average cost over last 1000 iterations
                J_new = J_new / 1000
                # check convergence
                if abs(J_new - J_old) < epsilon:
                    converged = True
                    break
                # assign J_new to J_old, reset J_new
                J_old = J_new
                J_new = 0

        # check if the average cost function has converged
        if converged:
            break
    # convergence achieved, parameter is optimized

    # STOCHASTIC GRADIENT DESCENT ENDS

    # return optimized parameter and theta_0, theta_1, theta_2 parameters (for plotting)
    return theta, theta_0, theta_1, theta_2


# function used to test learnt parameters (param) on unseen data (present in filename "file")
# theta is the original parameter from which the data was sampled, will be used for error comparison
def test_param(file, param, theta):
    
    # read "file" (CSV) file
    df = pd.read_csv(file)
    # convert data frame into numpy array
    Training_Set = df.to_numpy()

    # extract input features and target values from training data matrix
    X_train = Training_Set[:, 0:2]
    Y = Training_Set[:, 2:]
    
    # store number of training examples (m) and number of input features per example (n)
    m, n = X_train.shape

    # prepend x0 feature to every input feature in the training set to get X matrix (class notation)
    # np.ones creates an array containing only 1s of given size. np.column_stack stacks up columns
    X = np.column_stack((np.ones((m, 1)), X_train))
   
    # compute test error on learnt parameters (param), using J_test = (1 / (2*m)) (X @ param - Y)' @ (X @ param - Y), ' denotes transpose
    J_test_sgd = (1 / (2 * m)) * (X @ param - Y).T @ (X @ param - Y)

    # compute test error on original parameters (theta), using J_test = (1 / (2*m)) (X @ theta - Y)' @ (X @ theta - Y), ' denotes transpose
    J_test_og = (1 / (2 * m)) * (X @ theta - Y).T @ (X @ theta - Y)

    # report error
    print("Test Error for learnt parameters: " + str(J_test_sgd[0, 0]) + " units.")
    print("Test Error for original parameters: " + str(J_test_og[0, 0]) + " units.")
    
    # find and report difference
    if J_test_og >= J_test_sgd:
        print("Learnt Parameters fit the test data better than Original Parameters.")
        print("J_test_og - J_test_sgd = " + str(J_test_og[0, 0] - J_test_sgd[0, 0]))
    else:
        print("Original Parameters fit the test data better than Learnt Parameters.")
        print("J_test_sgd - J_test_og = " + str(J_test_sgd[0, 0] - J_test_og[0, 0]))

linearX = "linearX.csv"
linearY = "linearY.csv"
theta = np.array([[3], [1], [2]])
N = 1000000
batch_size = 1
(X, Y) = sampling(theta, N)
param, theta_0, theta_1, theta_2 = linear_regression_sgd(X, Y, batch_size)
print(param)
test_param("q2test.csv", param, theta)

# Figure 1: Movement of theta parameter in space (theta_0(x), theta_1(y), theta_2(z))
fig = plt.figure(1)
ax = fig.add_subplot(projection="3d")
# set title
ax.set_title("Movement of $\Theta$ with each update")

# number of points to plot
num_points = len(theta_0)

# convert theta_0, theta_1 and theta_2 into numpy arrays
theta_0 = np.array(theta_0).reshape((1, num_points))
theta_1 = np.array(theta_1).reshape((1, num_points))
theta_2 = np.array(theta_2).reshape((1, num_points))

# plot line
line, = ax.plot(theta_0[0, :], theta_1[0, :], theta_2[0, :], "-r", label="$\Theta = (\Theta_0, \Theta_1, \Theta_2)$")

# mark start and end points
ax.scatter(theta_0[0, 0], theta_1[0, 0], theta_2[0, 0], marker="*", color="blue", label="START of SGD", s=50)
ax.scatter(theta_0[0, num_points - 1], theta_1[0, num_points - 1], theta_2[0, num_points - 1], marker="*", color="green", label="END of SGD", s=50)

# do animation
#animation = animation.FuncAnimation(fig, animate, frames=num_points, fargs=(line, theta_0, theta_1, theta_2), interval=1, blit=False)

# set axes label
ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$\Theta_2$")

# show legend
plt.legend()

# show plot
plt.show()