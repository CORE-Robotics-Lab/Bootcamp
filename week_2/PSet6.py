'''
Authors: 	Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu
			Rohan Paleja <rpaleja3@gatech.edu>
			Letian Chen <letian.chen@gatech.edu>
            Manisha Natarajan <mnatarajan30@gatech.edu>
			
Date: 		24 JUN 2020
'''

import numpy as np
from createNeuralNet import createNeuralNet
from forwardPass import forwardPass
from backprop import backprop
import matplotlib.pyplot as plt

np.random.seed(1)                                                # Set random seed for repeatability
save = True

# Hyperparameters
alpha = 0.0001                                                    # Learning rate for all gradient descent methods
K = 1                                                            # Number of samples in our minibatch (K = 1 means SGD)
I = 10000                                                        # Number of gradient descent iterations

# Do not change
n = 100                                                          # Number of examples
d = 10                                                           # Number of features
numOutputDims = 1                                                # Dimensionality of the outputs space (DO NOT CHANGE)
L = []                                                           # Store the loss before each iteration

# Generate training data
X = np.random.random((d, n)) - 0.5                               # Inputs
numNodesPerLayer = [5, 5, 5, 5, 5]                               # Vector of the number of hidden nodes per layer, indexed from input -> output
nn_target = createNeuralNet(numNodesPerLayer, d, numOutputDims)  # Create a neural network to generate labels for our training data
Y = forwardPass(nn_target, X)                                    # Perform forward pass on whole data set
noise = np.random.rand(1, n)                               # Noise
Y = Y + noise                                                    # Labels

# Neural Network Setup
numNodesPerLayer = [10, 10, 10, 10, 10]                          # Vector of the number of hidden nodes per layer, indexed from input -> output
numLayers = len(numNodesPerLayer)                                # Number of hidden layers
nn = createNeuralNet(numNodesPerLayer, d, numOutputDims)         # Initialize our learner

for i in range(I):

    # stocahstic gradient descent

    y_hat = forwardPass(nn, X)                                   # Perform forward pass on whole data set

    L.append(0.5 * np.mean((Y - y_hat) ** 2))                    			 # Compute the MSE loss at this iteration for the entire training data set (not just the samples chosen)

    randInd = np.random.randint(0, high=n, size=K)               # Randomly sample a minibatch of size K
    Y_temp = Y[:, randInd]                                       # Get the label
    X_temp = X[:, randInd]                                       # Get the input features

    Y_hat_temp = forwardPass(nn, X_temp)                         # Get predicted label

    g = backprop(nn, X_temp, Y_temp)                             # Compute the gradient for the neural network

    for j in range(len(nn)):
        nn[j][0] = nn[j][0] - alpha * g[j][0]
        nn[j][1] = nn[j][1] - alpha * g[j][1]                      # Update the neural network biases

            
			
    # Plot the loss after every 5 iteration vs. the iteration number.
    print('Iteration: ', i)
    if i % 1000 == 999:
        fig, axes = plt.subplots(1, 1)
        axes.semilogx(range(1, i + 1), L[:i], '.-g')
        axes.set_xlabel('Iterations')
        axes.set_ylabel('Loss')
        axes.set_yscale('log')
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.show()
if save:
    fig.savefig('PSet6_plot.png')
