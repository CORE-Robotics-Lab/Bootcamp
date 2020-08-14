# CORE Robotics Lab Bootcamp Summer 2020 Week 1

# Authors: Letian Chen, Matthew Gombolay
# Date: 06/12/2020
# Work by CORE Robotics Lab at Georgia Institute of Technology

# Copyright Letian Chen and Matthew Gombolay, 2020.

# Please execute this file in terminal instead of inside PyCharm as it generates updating figures when updating_figure==True

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
np.random.seed(1)        # Set random seed for repeatability

n = 100                  # Number of examples
d = 10                   # Number of features
alpha = 0.1              # Learning rate for all gradient descent methods
I = 10000                # Number of gradient descent iterations, you may wish to make it smaller when updating_figure==True
updating_figure = True   # Whether to generate figures as training goes (useful for debug). Please set to False when generating final results with I=10000, or you will wait forever

# Generate training data
# NOTE: this script is translated from Matlab
# for X, the more conventional representation in Python is [n, d], so are some other variables
# so please be aware of the dimensional ordering when doing calculation!
X = np.random.random([d, n])                      # Inputs
B = np.random.random([d, 1])                      # Ground-truth Model Params
noise = 0.1 * np.random.random([n, 1])            # Noise
Y = np.transpose(np.transpose(B).dot(X)) + noise  # Labels

# Generate parameters of the model's we will learn. Each model is a neural
# network with a single hidden layer with linear activations and no bias
# terms (i.e., linear regression).
theta_o = np.random.random([d, 1])

# Gradient Descent
theta_GD = theta_o.copy()                        # Initialize params
L_GD = np.zeros([I])                             # Store the loss before each iteration

# Stochastic Gradient Descent
theta_SGD = theta_GD.copy()                      # Initialize params
L_SGD = np.zeros([I])                            # Store the loss before each iteration
k_vec = np.random.randint(0, n, [1, I])          # Pre-compute the indices of the random examples we will draw for during training

# Mini-batch Hyperparameter
theta_MbGD = theta_o.copy()                      # Initialize params
n_Mb = 10                                        # Size of the mini-batch
L_MbGD = np.zeros([I])                           # Store the loss before each iteration
k_vec_MbGD = np.random.randint(0, n, [n_Mb, I])  # Pre-compute the indices of the random examples we will draw for during training

fig, axes = plt.subplots(1, 2)


def plot_progress(iter, save=False):
    # Plot the loss after each iteration vs. the iteration number.
    axes[0].clear()
    axes[0].semilogx(range(1, iter + 1), L_GD[:iter], '.-k', label="GD")
    axes[0].semilogx(range(1, iter + 1), L_SGD[:iter], '.-b', label="SGD")
    axes[0].semilogx(range(1, iter + 1), L_MbGD[:iter], '.-g', label="MbGD")
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='upper right')

    # Plot the loss after each iteration vs. the number of samples used
    # thus far to train the model.
    axes[1].clear()
    axes[1].semilogx(np.arange(1, iter + 1) * n, L_GD[:iter], '.-k', label="GD")
    axes[1].semilogx(np.arange(1, iter + 1), L_SGD[:iter], '.-b', label="SGD")
    axes[1].semilogx(np.arange(1, iter + 1) * n_Mb, L_MbGD[:iter], '.-g', label="MbGD")
    axes[1].set_xlabel('Data Samples Used')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    if save:
        fig.savefig('week1.png')


for i in range(I):
    # Gradient Descent
    Y_hat_GD = np.transpose(np.transpose(theta_GD).dot(X))                       # Predict the labels
    L_GD[i] = 0.5 * np.mean((Y - Y_hat_GD) ** 2)                                 # Compute the loss at this iteration
    g = - X.dot(Y - Y_hat_GD) / n                                                # Compute the gradient (Remember: normalize by the number of samples in the full batch!)
    theta_GD = theta_GD - alpha * g                                              # Update the parameters

    # Stochastic gradient descent
    Y_hat_SGD = np.transpose(np.transpose(theta_SGD).dot(X))                     # Predict the labels for the entire training data set (not just the samples chosen)
    L_SGD[i] = 0.5 * np.mean((Y - Y_hat_SGD) ** 2)                               # Compute the loss at this iteration for the entire training data set (not just the samples chosen)
    X_temp = X[:, k_vec[0, i]][:, np.newaxis]
    g = - X_temp.dot(Y[k_vec[0, i]] - Y_hat_SGD[k_vec[0, i]])                    # Compute the gradient (Remember: there is no need to normalize, as the sample size is just 1!)
    theta_SGD = theta_SGD - alpha * g[:, np.newaxis]                             # Update the parameters

    # Mini-batch gradient descent
    Y_hat_MbGD = np.transpose(np.transpose(theta_MbGD).dot(X))                   # Predict the labels for the entire training data set (not just the samples chosen)
    L_MbGD[i] = 0.5 * np.mean((Y - Y_hat_MbGD) ** 2)                             # Compute the loss at this iteration for the entire training data set (not just the samples chosen)
    X_temp = X[:, k_vec_MbGD[:, i]]                                              # Get the input features just for this minibatch
    Y_hat_MbGD_temp = np.transpose(np.transpose(theta_MbGD).dot(X_temp))         # Predict the labels just for the mini-batch
    g = - X_temp.dot(Y[k_vec_MbGD[:, i]] - Y_hat_MbGD[k_vec_MbGD[:, i]]) / n_Mb  # Compute the gradient for the mini-batch (Remember: normalize by the number of samples in the mini-batch!)
    theta_MbGD = theta_MbGD - alpha * g                                          # Update the parameters

    if updating_figure:
        plot_progress(i)

plot_progress(I, save=True)
