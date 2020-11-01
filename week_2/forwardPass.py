import numpy as np


def forwardPass(nn, X, return_output=False):
    """
    Authors:    Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu>
                Rohan Paleja <rpaleja3@gatech.edu>
                Letian Chen <letian.chen@gatech.edu>
                Manisha Natarajan <mnatarajan30@gatech.edu>

    Date:       24 JUN 2020

    This function takes as input a neural netowrk, nn, and inputs, X, and
    performs a forward pass on the neural network. The code assumes that
    layers, {1,2,...,l-1}, have ReLU activations and the final layer has a
    linear activation function.

    Inputs:

    nn          -   The weights and biases of the neural network. nn[i][0]
                    corresponds to the weights for the ith layer and
                    nn[i][1] corresponds to the biases for the ith layer
    X           -   This matrix is n x d matrix and contains the input
                    features for n examples, each with d features.

    Outputs:

    Y           -   This term is an n x 1 vector of predicted labels for
                    the n examples.
    """

    numLayers = len(nn)                                            # get the number of layers of our neural network

    outputs = [None] * numLayers                                   # Intialize the cell to store the outputs of the hidden layers
    inputs = []                                                    # Store z values from each layer. To be used in backprop
    for i in range(numLayers):
        # Compute the input to layer i
        if i == 0:
            # z[0] = dot(W[0].T, X) + b[0]   (0-> first layer)
            inputs.append(np.matmul(nn[i][0], X) + nn[i][1])
        else:
            # Subsequent z computations use output from previous layer
            # z[i] = dot(W[i].T, output[i - 1]) + b[i]
            inputs.append(np.matmul(nn[i][0], outputs[i-1]) + nn[i][1])

        if i < numLayers-1:                                         # range function is exclusive so we do numLayers - 1
            # If layer i is not the output layer, then apply the ReLU activation function for the nodes at this layer.
            outputs[i] = inputs[i].copy()
            outputs[i][outputs[i] < 0] = 0
        else:
            # If layer i is the output layer, then apply a linear activation (i.e., no activation)
            outputs[i] = inputs[i]
			
    Y_hat = outputs[-1]
    if return_output:
        return Y_hat, outputs, inputs,
    else:
        return Y_hat


