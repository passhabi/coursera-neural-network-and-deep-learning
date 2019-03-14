import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture in train_set:
index = 150
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' picture.")
plt.show()

# Exercise: Find the values for:
#
# - m_train (number of training examples)
# - m_test (number of test examples)
# - num_px (= height = width of a training image)

# START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
# END CODE HERE ###

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# Exercise: Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened
#   into single vectors of shape (num_px $*$ num_px $*$ 3, 1).
train_set_x_flatten = train_set_x_orig.reshape([m_train, -1]).T
test_set_x_flatten = test_set_x_orig.reshape([m_test, -1]).T

print("for train set: \n x: {} \n y: {}".format(str(train_set_x_flatten.shape), str(train_set_y.shape)))
print("for test set: \n x: {} \n y: {}".format(str(test_set_x_flatten.shape), str(test_set_y.shape)))

# Standardize the dataset:
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# 4.1 - Helper functions

# Exercise: Using your code from "Python Basics", implement sigmoid(). As you've seen in the figure above, you need to
#   compute $sigmoid( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$ to make predictions. Use np.exp().

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    # START CODE HERE ### (≈ 1 line of code)
    y_hat = 1 / (1 + np.exp(-z))
    # END CODE HERE ###

    return y_hat


# to check:
print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))


# 4.2 - Initializing parameters

# Exercise: Implement parameter initialization in the cell below. You have to initialize w as a vector of zeros.
#   If you don't know what numpy function to use, look up np.zeros() in the Numpy library's documentation.

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    # START CODE HERE ### (≈ 1 line of code)
    w = np.zeros([dim, 1])
    b = 0
    # END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


#  to check:
dim = 2
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))


# 4.3 - Forward and Backward propagation
# Exercise: Implement a function propagate() that computes the cost function and its gradient.

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]  # notice we reshaped(transposed) X to (nx, m)

    # FORWARD PROPAGATION (FROM X TO COST)
    # START CODE HERE ### (≈ 2 lines of code)
    z = np.dot(X.T, w) + b
    # compute activation
    a = sigmoid(z)
    a = np.reshape(a, [1, -1])
    # compute cost
    cost = -1 / m * np.sum(Y * np.log(a) + ((1 - Y) * np.log(1 - a)))
    # END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    # START CODE HERE ### (≈ 2 lines of code)
    dz = a - Y
    dw = 1 / m * np.dot(X, dz.T)
    db = 1 / m * np.sum(dz)
    # END CODE HERE ###

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# to check:
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))


# d) Optimization

# Exercise: Write down the optimization function. The goal is to learn w and b by minimizing the cost function j.
#   For a parameter w the update rule is w = w - c dw, where c is learning rate.
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        # START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        # END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        # START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# to check:
params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))


# Exercise: The previous function will output the learned w and b. We are able to use w and b to predict
#   the labels for a dataset X. Implement the predict() function. There is two steps to computing predictions:

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    # START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(X.T, w) + b)
    # END CODE HERE ###
    print(range(A.shape[1]))
    for i in range(A.shape[0]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        # START CODE HERE ### (≈ 4 lines of code)
        if A[i, 0] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
        # END CODE HERE ###
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


print("predictions = " + str(predict(w, b, X)))
print("X is " + str(X.shape))


# 5 - Merge all functions into a model

# Exercise: Implement the model function. Use the following notation:
#
# - Y_prediction for your predictions on the test set
# - Y_prediction_train for your predictions on the train set
# - w, costs, grads for the outputs of optimize()

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    m = X_train.shape[0]
    w, b = initialize_with_zeros(m)
    print(w.shape)
    # Gradient descent (≈ 1 line of code)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = params["w"]
    b = params["b"]
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    # END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# to check:
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Let's also plot the cost function and the gradients.

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

