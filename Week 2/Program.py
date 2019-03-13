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


# 3 - General Architecture of the learning algorithm
