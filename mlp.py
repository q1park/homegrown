import numpy as np
import pandas as pd

## Identity function
def identity(X):
    return X

## Activation functions
def sigmoid(X):
    return 1 / (1 + np.exp(-np.array(X)))

def relu(X):
    return np.piecewise((X),[X<0.0, X>=0.0], [lambda x: 0.001, lambda x: x])

def leakyrelu(X):
    return np.piecewise((X),[X<0.0, X>=0.0], [lambda x: x*1e-5, lambda x: x])

def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

fdict={'identity':identity,
       'relu':relu,
       'leakyrelu':leakyrelu,
       'sigmoid':sigmoid,
       'softmax':softmax}

## Activation Derivatives
def dsigmoid(X):
    return sigmoid(X) * (1 - sigmoid(X))

def drelu(X):
    return np.piecewise((X),[X<0.0, X>=0.0], [lambda x: 0.0, lambda x: 1.0])

def dleakyrelu(X):
    return np.piecewise((X),[X<0.0, X>=0.0], [lambda x: 1e-5, lambda x: 1.0])

def dsoftmax(X):
    dsoft = np.empty((len(X), len(X)) )
    
    for i in range(len(X)):
        for j in range(len(X)):
            if i==j:
                dsoft[i][j] = softmax(X)[i]*(1 - softmax(X)[j])
            else:
                dsoft[i][j] = -softmax(X)[i]*softmax(X)[j]
    return dsoft

dfdict={'identity':identity,
        'relu':drelu,
        'leakyrelu':dleakyrelu,
        'sigmoid':dsigmoid,
        'softmax':dsoftmax}

## Quantum Relative Entropy
def qrelative_entropy(y, output):
    assert y.shape == output.shape
    eps = 1e-16
    y = np.piecewise(y,[y<eps, y>=eps], [lambda x: eps, lambda x: x])
    
    if output is None:
        output = np.random.uniform(eps, 1-eps, size = y.shape)
    else:
        output = np.piecewise(output,[output<eps, output>=eps], [lambda x: eps, lambda x: x])
    
    S = (1/len(y))*np.sum([y[i]*np.log(y[i]/output[i]) for i in range(len(y))])
    dS = -(1/len(y))*np.array([y[i]/output[i] for i in range(len(y))]).reshape(len(y), 1)
    ddS = (1/len(y))*np.array([y[i]/output[i]**2 for i in range(len(y))]).reshape(len(y), 1)
    
    return S, dS, ddS
     
## Linear Perception Layer
class linear:
    def __init__(self, indim, outdim, activation = 'identity'):
        self.weight = np.random.normal(scale=1 / outdim**.5, size=(outdim, indim))
        self.bias = np.random.normal(scale=1 / outdim**.5, size=(outdim, 1))
        self.func = fdict[activation]
        self.dfunc = dfdict[activation]
        
        self.instate = None
        self.outstate = None
        self.sgrad = None
        self.wgrad = np.zeros(self.weight.shape)
        self.bgrad = np.zeros(self.bias.shape)
        self.npasses = 0
    
    def __call__(self, instate):
        self.instate = instate.reshape(instate.shape[0], 1)
        self.outstate = np.dot(self.weight, self.instate) + self.bias
        
        self.sgrad = self.dfunc(self.outstate)
        self.npasses += 1
        return self.func(self.outstate)

    def reset(self):
        self.wgrad = np.zeros(self.weight.shape)
        self.bgrad = np.zeros(self.bias.shape)
        self.npasses = 0
        
## Multi-layered Perception Class
class network:
    def __init__(self, inputdim):
        self.inputdim = inputdim
        self.runningloss = 0.0
        self.nlayers = 0
        
        self.layers = {}
        
        self.runningloss = 0.0
        self.nbatch = 0
        self.batchlosses = list()
        
    def addlinear(self, layerdim, activation):
        self.nlayers += 1
        layer = [linear(self.inputdim, layerdim, activation = activation)]
        layer.append(layer[0].func)
        layer.append(tuple(reversed(layer[0].weight.shape) ) )
        
        self.layers[self.nlayers] = layer
        self.inputdim = layerdim
        
    def forward(self, iX, iY):
        iY = iY.reshape(len(iY), 1)
        state = iX
        for x in self.layers:
            state = self.layers[x][0](state)
            
        loss, dloss, _ = qrelative_entropy(iY, state)
        
        dchain = np.sum(dloss*self.layers[self.nlayers][0].sgrad, axis = 0)
        dchain = dchain.reshape(len(dchain), 1)
        dweight = np.outer(dchain, self.layers[self.nlayers][0].instate )
        dbias = dchain
        self.layers[self.nlayers][0].wgrad+=dweight
        self.layers[self.nlayers][0].bgrad+=dbias
            
        for i in reversed(range(1, self.nlayers) ):
            dchain = np.sum(dchain*self.layers[i + 1][0].weight, axis = 0)
            dchain = dchain.reshape(len(dchain), 1)
            dchain = dchain*self.layers[i][0].sgrad

            dweight = np.outer(dchain, self.layers[i][0].instate)
            dbias = dchain

            self.layers[i][0].wgrad+=dweight
            self.layers[i][0].bgrad+=dbias
            
        self.batchlosses.append(loss)
        
    def backprop(self, learnrate):
        self.runningloss += np.sum(self.batchlosses)/len(self.batchlosses)
        self.nbatch += 1
        self.batchlosses.clear()
        for x in self.layers:
            nbatch = self.layers[x][0].npasses
            self.layers[x][0].weight -= (learnrate/nbatch)*self.layers[x][0].wgrad
            self.layers[x][0].bias -= (learnrate/nbatch)*self.layers[x][0].bgrad
            self.layers[x][0].reset()
            
    def getloss(self):
        if len(self.batchlosses) > 0:
            self.runningloss += np.sum(self.batchlosses)/len(self.batchlosses)
            self.nbatch += 1
            self.batchlosses.clear()
            for x in self.layers:
                self.layers[x][0].reset()
        avgloss = self.runningloss/self.nbatch
        self.runningloss = 0.0
        self.nbatch = 0
        return avgloss
        