import numpy as np
from numpy import random as ra
import copy


# create a list of dictionairies, 
# each representing a layer with input and output dim aswell as activation function

nnArchitecture = [
    {"inDim" : 2, "outDim": 4, "activation": "reLU"},
    {"inDim" : 4, "outDim": 6, "activation": "reLU"},
    {"inDim" : 6, "outDim": 6, "activation": "reLU"},
    {"inDim" : 6, "outDim": 4, "activation": "reLU"},
    {"inDim" : 4, "outDim": 1, "activation": "sigmoid"}
]

# create function to create a random shuffling of weights and biases

def valueShuffler(nnArchitecture_, seed = 0):
    np.random.seed(seed)
    numLayer = len(nnArchitecture_)
    shuffleValues = {}

    for idx, layer in enumerate(nnArchitecture_):
        layerIdx = idx + 1
        inDim   = layer["inDim"]
        outDim   = layer["outDim"]
                                             #create weight matrix with col = neuron, row connection
        shuffleValues["w" + str(layerIdx)] = ra.randn(outDim, inDim) * 0.1
        shuffleValues["b" + str(layerIdx)] = ra.randn(outDim, 1) * 0.1
    
    return shuffleValues 

# Define activation functions

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derSigmoid(dA, z):
    sigRes = sigmoid(z)
    return dA * (sigRes * (1 - sigRes))

def reLU(z): #for each element, if > 0 then stay if < then =0
    return np.maximum(0, z)

def derReLU(dA, z): #for back-prop, a and z same dim, 
    dz = np.array(dA, copy=True)
    # set relu of each element in dA by comparing with z
    dz[z <= 0] = 0
    return dz

## Create forward propagation

# for 1 layer
def singleLayerForwardProp(aPrev, wCurr, bCurr, activation = "reLU"):
            #NOT MATMUL OR @
    zCurr = np.dot(wCurr, aPrev) + bCurr

    if activation == "reLU":
        return reLU(zCurr), zCurr
    elif activation == "sigmoid":
        return sigmoid(zCurr), zCurr
    else:
        raise Exception("wrong activation function")

def fullLayerForwardProp(input, paramValues, nnArchitecture_):
    history = {}
    #input neurons
    aCurr = copy(input)

    for idx, layer in enumerate(nnArchitecture_):
        layerIdx = idx + 1
        #next layer neurons
        aPrev = aCurr

        wCurr              = paramValues["w" + str(layerIdx)]
        bCurr              = paramValues["b" + str(layerIdx)]
        aCurr, zCurr       = singleLayerForwardProp(aPrev, wCurr, bCurr, layer["activation"])

        history["a" + str(idx)] = aPrev
        history["z" + str(layerIdx)] = zCurr 
    #return the last neuron layer (output) 
    return aCurr, history





