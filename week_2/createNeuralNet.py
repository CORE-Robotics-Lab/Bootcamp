import numpy as np


def createNeuralNet(numNodesPerLayer, numInputDims, numOutputDims):
    '''
    Authors:    Matthew Gombolay <Matthew.Gombolay@cc.gatech.edu
                Rohan Paleja <rpaleja3@gatech.edu>
                Letian Chen <letian.chen@gatech.edu>

    Date:       24 JUN 2020

    This function takes as input the number of nodes per hidden layer as well
    as the size of the input and outputs of the neural network and returns a
    randomly initialized neural network. Weights for the network are
    generated randomly using the method of He et al. ICCV'15.

    Inputs:

    numNodesPerLayer    -   This vector contains natural numbers for the
                            quantity of nodes contained within each hidden
                            layer.
    numInputDims        -   This number represents the cardinality of the
                            input vector for the neural network
    numOutputDims       -   This number represents the cardinality of the
                            output vector of the neural network

    Outputs:

    nn                  -   nn, the neural network created.

    '''
    nn = []
    numLayers = len(numNodesPerLayer)
    for i in range(numLayers + 1):
        if i == 0:
            # Use numInputDims for the input size
            nn.append([np.random.random((numNodesPerLayer[i], numInputDims)) * np.sqrt(2.0 / numInputDims), 0.01*np.ones((numNodesPerLayer[i], 1))])
        elif i == numLayers:
            # Use numOutputDims for the output size
            nn.append([np.random.random((numOutputDims, numNodesPerLayer[i-1])) * np.sqrt(2.0 / numNodesPerLayer[i-1]), 0.01*np.ones((numOutputDims, 1))])
        else:
            nn.append([np.random.random((numNodesPerLayer[i], numNodesPerLayer[i-1])) * np.sqrt(2.0 / numNodesPerLayer[i-1]), 0.01*np.ones((numNodesPerLayer[i], 1))])
    return nn



