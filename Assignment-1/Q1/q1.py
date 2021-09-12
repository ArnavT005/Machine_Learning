from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# animation function, used to animate gradient descent trajectory in mesh plot (3D)
# num is the frame number, theta_0, theta_1 and J_val are the data sets containing gradient descent data
# line is the line3D object that will be displayed in the plot (3D)
def animate3D(num, theta_0, theta_1, J_val, line):  
    # set x and y line data
    line.set_data(theta_0[0, :num], theta_1[0, :num])    
    # set z line data
    line.set_3d_properties(J_val[0, :num])   
    # return line and point object
    return line

# animation function, used to animate gradient descent trajectory in contour plot (2D)
# num is the frame number, theta_0 and theta_1 contain gradient descent parametric data
# line is the line2D object that will be displayed in the plot (2D)
def animate2D(num, theta_0, theta_1, line):
    # set x and y data
    line.set_data(theta_0[0, :num], theta_1[0, :num])
    # return line object
    return line

# used to determine cost-value for each point present in mesh grid, used to plot cost as a function of parameters (3D)
# theta_x and theta_y represent the grid, X and Y are the training examples (m examples)
def cost_function(theta_x, theta_y, X, Y, m):
    # store dimensions of the grid
    n_x = theta_x.shape[1]
    n_y = theta_y.shape[0]
    # initialise cost array
    cost_z = np.zeros((n_y, n_x))

    # For all points in grid, compute cost using, J = (1 / (2*m)) * (X @ theta - Y)' @ (X @ theta - Y)
    for i in range(0, n_x):
        for j in range(0, n_y):
            # stack parameters together (theta) for vectorized implementation
            theta = np.row_stack((theta_x[j, i], theta_y[j, i]))
            cost_z[j, i] = (1 / (2 * m)) * (X @ theta - Y).T @ (X @ theta - Y)
    
    # return cost array (over a grid)
    return cost_z

# function to train a linear regression model
# input features and target values to be read from "linearX" and "linearY" file respectively 
# eta is the learning rate, epsilon is the convergence threshold
def linear_regression(linearX, linearY, eta, epsilon):

    # read "linearX" (CSV) file
    df = pd.read_csv(linearX, header=None)
    # convert into numpy array
    X_train = df.to_numpy()
    
    # store number of training examples (m) and number of input features per example (n)
    m, n = X_train.shape
    
    # determine mean and standard deviation (std) feature-wise (column, axis=0)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    
    # normalize data with mean 0 and variance 1
    # row dimension of X_mean and X_std is broadcasted to match the dimension of X_train
    # / and - carry out element-wise division and subtraction respectively
    X_norm = (X_train - X_mean) / X_std

    # prepend x0 feature to every input feature in the training set to get X matrix (class notation)
    # np.ones creates an array containing only 1s of given shape. np.column_stack stacks up columns
    X = np.column_stack((np.ones((m, 1)), X_norm))

    # read "linearY" (CSV) file
    df = pd.read_csv(linearY, header=None)
    # convert into numpy array Y (class notation)
    Y = df.to_numpy()
    
    # error handling, dimension check
    # check whether the number of inputs is equal to the number of outputs
    if Y.shape[0] != m:
        print("Error: Inconsistent dimensions. Number of inputs is not equal to number of outputs.")
        return None
    else:
        # check whether target value is real (single column) or not
        if Y.shape[1] != 1:
            print("Error: Target value is not a real number.")
            return None
    
    # GRADIENT DESCENT STARTS
    
    # initialize weight parameter (theta) with 0 vector (dimension=(n + 1))
    # np.zeros creates an array containing only 0s of given shape
    theta = np.zeros((n + 1, 1))

    # determine initial cost (or MSE)
    # J = (1 / (2*m)) * (X @ theta - Y)' @ (X @ theta - Y), ' denotes transpose
    # .T is used to transpose numpy array and @ is used for matrix multiplication
    J = (1 / (2 * m)) * (X @ theta - Y).T @ (X @ theta - Y)

    # create lists to store parameters (theta) and cost (J); used later for mesh/contour plotting
    theta_0 = [theta[0, 0]]
    theta_1 = [theta[1, 0]]
    J_val = [J[0, 0]]

    # gradient descent loop (repeat until convergence)
    while True:
        # store pre-update cost for checking convergence post-update
        temp = J[0, 0]
        
        # compute batch gradient using expression derived in class: grad = (1 / m) * X' @ (X @ theta - Y), ' denotes transpose
        grad = (1 / m) * X.T @ (X @ theta - Y)

        # update theta, eta is the learning rate
        theta = theta - eta * grad

        # update cost (using the same expression as before)
        J = (1 / (2 * m)) * (X @ theta - Y).T @ (X @ theta - Y)

        # store the parameters and cost, for plotting purposes
        theta_0.append(theta[0, 0])
        theta_1.append(theta[1, 0])
        J_val.append(J[0, 0])
    
        # check for convergence
        if abs(J[0, 0] - temp) < epsilon:
            break
    # convergence achieved, parameter is optimized

    # GRADIENT DESCENT ENDS

    # return model parameters and other parameters used in plotting
    return theta, X, X_mean, X_std, Y, m, theta_0, theta_1, J_val, J

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

    # set learning rate (eta) and convergence threshold (epsilon)
    eta, epsilon = 0.3, 1e-18

    # train linear regression model
    try:
        theta, X, X_mean, X_std, Y, m, theta_0, theta_1, J_val, J = linear_regression(linearX, linearY, eta, epsilon)
    except:
        return

    # store number of Loss-Theta pairs available
    num_points = len(theta_0) 
    print("Number of iterations: " + str(num_points - 1))
    print("Learnt Parameter: " + str(theta))

    # convert iteration data into numpy arrays (row vectors)
    theta_0 = np.array(theta_0).reshape((1, num_points))
    theta_1 = np.array(theta_1).reshape((1, num_points))
    J_val = np.array(J_val).reshape((1, num_points))

    # Figure 1: Plot dataset (normalized) and hypothesis (colored red), Y_hat = h(X) = X @ theta
    plt.figure(1)
    plt.title("Linear Regression Model")
    # scatter the training examples (normalized)
    plt.scatter(X[:, 1], Y, label="Training Examples")

    # get limits of x-axis, and sample m input points (X_sample) 
    x_min, x_max = plt.xlim()
    X_sample = np.linspace(x_min, x_max, num=m, endpoint=True)
    # plot hypothesis (theta[0] + theta[1] * X_sample) for this sample of points 
    plt.plot(X_sample, theta[0, 0] + theta[1, 0] * X_sample, "r", label="Model Prediction ($h_\Theta(x)$)")

    # assign labels to axes
    plt.xlabel("Acidity of wine (normalized pH, mean=" + str(X_mean) + " and std=" + str(X_std) + ")")
    plt.ylabel("Density of wine (relative density)")

    # provide legend for reference and show graph
    plt.legend(bbox_to_anchor=(0,1), loc="upper left")
    plt.savefig("q1plot1.jpg")

    # create video writer to save animations
    ffmpeg_writer = animation.FFMpegFileWriter(fps=5)

    # Figure 2: Draw a 3D-mesh showing loss function J as a function of theta and animate trajectory of gradient descent
    fig = plt.figure(2, figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    # set title
    ax.set_title('Loss Function (MSE) vs $\Theta$ ($\eta = 0.3$)')
    # rotate plot
    ax.view_init(45, -75)

    # set parameter (theta) range for x-y axes (theta_0 on x, theta_1 on y)
    # theta is the learnt parameter
    theta_x_min, theta_x_max = theta[0, 0] - 1.1, theta[0, 0] + 0.5
    theta_y_min, theta_y_max = theta[1, 0] - 0.5, theta[1, 0] + 0.5

    # create parameter grid for 3D plot, use step=0.1 (between sampled points)
    theta_x, theta_y = np.meshgrid(np.arange(theta_x_min, theta_x_max + 0.1, 0.1), np.arange(theta_y_min, theta_y_max + 0.1, 0.1))

    # calculate cost values (z-axis) for each value in theta_x and theta_y (present on grid)
    cost_z = cost_function(theta_x, theta_y, X, Y, m)
    
    # mark start and end points
    ax.scatter(theta_0[0, 0], theta_1[0, 0], J_val[0, 0], marker="*", color="blue", label="START of GD", s=150)
    ax.scatter(theta_0[0, num_points - 1], theta_1[0, num_points - 1], J_val[0, num_points - 1], marker="*", color="green", label="END of GD", s=150)

    # plot loss surface
    ax.plot_surface(theta_x, theta_y, cost_z, rstride=1, cstride=1, alpha=0.5, cmap="jet")

    # plot line (trajectory of gradient descent)
    line, = ax.plot(theta_0[0, :], theta_1[0, :], J_val[0, :], "--r", label="Gradient Descent Trajectory", marker='o')
    line.set_markersize(5)

    # set axes labels
    ax.set_xlabel("$\Theta_0$")
    ax.set_ylabel("$\Theta_1$")
    ax.set_zlabel("Loss Value (MSE)")

    # set z-axis limit
    ax.set_zlim(0, 1)

    # show legend
    ax.legend()
    # save figure
    plt.savefig("q1mesh3D.jpg")

    # do animation, put delay of 0.2 seconds
    anim = animation.FuncAnimation(fig, animate3D, frames=num_points, fargs=(theta_0, theta_1, J_val, line), interval=200, blit=False) 
    anim.save("q1mesh3D.gif", writer=ffmpeg_writer)


    # Figure 3: Draw a 2D-plot showing contours of loss function J as a function of theta and animate trajectory of gradient descent
    fig = plt.figure(3, figsize=(10, 10))
    ax = fig.add_subplot()
    # set title
    ax.set_title('Loss Function (MSE) Contours (in $\Theta$ plane) ($\eta = 0.3$)')
    
    # plot loss contour
    ax.contour(theta_x, theta_y, cost_z, 50, cmap = 'jet')

    # mark start and end points
    ax.scatter(theta_0[0, 0], theta_1[0, 0], marker="*", color="blue", label="START of GD", s=150)
    ax.scatter(theta_0[0, num_points - 1], theta_1[0, num_points - 1], marker="*", color="green", label="END of GD", s=150)

    # plot line (trajectory of gradient descent)
    line, = ax.plot(theta_0[0, :], theta_1[0, :], "--r", label = "Gradient Descent Trajectory", marker="o")
    line.set_markersize(5)

    # set axes labels
    ax.set_xlabel("$\Theta_0$")
    ax.set_ylabel("$\Theta_1$")

    # show legend
    ax.legend()
    # save figure
    plt.savefig("q1contour2D.jpg")

    # do animation, put delay of 0.2 seconds
    anim = animation.FuncAnimation(fig, animate2D, frames=num_points, fargs=(theta_0, theta_1, line), interval=200, blit=False) 
    anim.save("q1contour2D.gif", writer=ffmpeg_writer)


    # make contour plots for learning rate = 0.1, 0.025 and 0.001
    epsilon = 1e-20
    eta = [0.1, 0.025, 0.001]
    for i in range(0, 3):
        # train linear regression model using learning rate eta[i]
        try:
            theta, X, X_mean, X_std, Y, m, theta_0, theta_1, J_val, J = linear_regression(linearX, linearY, eta[i], epsilon)
        except:
            continue

        # store number of Loss-Theta pairs available
        num_points = len(theta_0) 
        print("Number of iterations: " + str(num_points - 1))
        print("Learnt Parameter: " + str(theta))

        # convert iteration data into numpy arrays (row vectors)
        theta_0 = np.array(theta_0).reshape((1, num_points))
        theta_1 = np.array(theta_1).reshape((1, num_points))
        J_val = np.array(J_val).reshape((1, num_points))

        # Figure: Draw a 2D-plot showing contours of loss function J as a function of theta and animate trajectory of gradient descent
        fig = plt.figure(i + 4, figsize=(10, 10))
        ax = fig.add_subplot()
        # set title
        ax.set_title('Loss Function (MSE) Contours (in $\Theta$ plane, $\eta =$' + ' ' + str(eta[i]) + ')')
        
        # plot loss contour
        ax.contour(theta_x, theta_y, cost_z, 100, cmap = 'jet')

        # mark start and end points
        ax.scatter(theta_0[0, 0], theta_1[0, 0], marker="*", color="blue", label="START of GD", s=80)
        ax.scatter(theta_0[0, num_points - 1], theta_1[0, num_points - 1], marker="*", color="green", label="END of GD", s=80)

        # plot line (trajectory of gradient descent)
        line, = ax.plot(theta_0[0, :], theta_1[0, :], "--r", label = "Gradient Descent Trajectory", marker="o")
        line.set_markersize(5)

        # set axes labels
        ax.set_xlabel("$\Theta_0$")
        ax.set_ylabel("$\Theta_1$")

        # show legend
        ax.legend()
        # save figure
        plt.savefig("q1contour2D" + str(i + 1) + ".jpg")

        # do animation, put delay of 0.2 seconds
        anim = animation.FuncAnimation(fig, animate2D, frames=num_points, fargs=(theta_0, theta_1, line), interval=200, blit=False) 
        # show animation
        plt.show()


#run main
main()