import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the training and test data
mnist_train = pd.read_csv("Data/mnist/train.csv")
mnist_test = pd.read_csv("Data/mnist/test.csv")

# Visualize the image represented by the first rows of the train data and the test data
train_data_digit1 = np.asarray(mnist_train.iloc[0, 1:]).reshape(28, 28)
test_data_digit1 = np.asarray(mnist_test.iloc[0, :]).reshape(28, 28)

# Create subplots to display the images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(train_data_digit1, cmap=plt.cm.gray_r)
plt.title("First digit in train data")
plt.axis('off')  # Hide axes

plt.subplot(1, 2, 2)
plt.imshow(test_data_digit1, cmap=plt.cm.gray_r)
plt.title("First digit in test data")
plt.axis('off')  # Hide axes

# Show the plots
plt.show()
