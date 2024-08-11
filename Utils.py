import pandas as pd
import numpy as np
import json as js
import math as mt
from matplotlib import pyplot as plt

class layers:

    def init(self):
        for i in range(self.layerSize):
            self.b.append(round(np.random.uniform(-0.5,0.5),2))
        self.b = np.array(self.b)

    def __init__(self, layerSize):
        self.layerSize = layerSize
        self.Z = []
        self.b = []
        self.A = []
        self.W = []
        self.dZ = []
        self.dW = []
        self.db = []
        self.init()

    def weights(self,prevLayerSize):
        # j = self.layerSize
        # k = prevLayerSize
        for j in range(self.layerSize):
            arr = []
            for k in range(prevLayerSize):
                arr.append(np.random.uniform(-0.5,0.5))
            self.W.append(arr)
        self.W = np.array(self.W)

def ReLu(Z):
    return np.maximum(0,Z)

def sumEXP(row):
    value = 0.0
    for element in row:
        value += mt.exp(element)
    return value

def softMax(Z):
    tempZ = np.array(Z)
    tempZ = tempZ.T
    res = []
    for row in tempZ:
        sum = sumEXP(row)
        arr = []
        for element in row:
            arr.append(mt.exp(element) / sum)
        res.append(arr)
    return np.array(res)

def matrixFit(b,m):
    tempB = []
    for i in range(m):
        tempB.append(b)
    return np.array(tempB)

def ReLu_derivative(Z):
    return Z > 0

def one_hot(Y):
    res = []
    for digit in Y:
        arr = []
        for i in range(10):
            if i != digit:
                arr.append(0)
            else:
                arr.append(1)
        res.append(arr)
    return np.array(res).T

def forwardPropogation(l1,l2,l3,l4,m):
    l2.Z = np.dot(l2.W, l1.A) + matrixFit(l2.b, m).T
    l2.A = ReLu(l2.Z)
    l3.Z = np.dot(l3.W, l2.A) + matrixFit(l3.b, m).T
    l3.A = ReLu(l3.Z)
    l4.Z = np.dot(l4.W, l3.A) + matrixFit(l4.b, m).T
    l4.A = softMax(l4.Z).T

def BackwardPropogation(l1,l2,l3,l4,Y,m):
    one_hot_Y = one_hot(Y)

    l4.dZ = l4.A - one_hot_Y
    l4.dW = np.dot(l4.dZ,l3.A.T) / m
    l4.db = np.sum(l4.dZ,axis=1) / m

    l3.dZ = np.dot(l4.W.T, l4.dZ) * ReLu_derivative(l3.Z)
    l3.dW = np.dot(l3.dZ, l2.A.T) / m
    l3.db = np.sum(l3.dZ,axis=1) / m

    l2.dZ = np.dot(l3.W.T, l3.dZ) * ReLu_derivative(l2.Z)
    l2.dW = np.dot(l2.dZ, X.T) / m
    l2.db = np.sum(l2.dZ,axis=1) / m

def gradientDescent(l1,l2,l3,l4,L):
    l2.W -= L*l2.dW
    l2.b -= L*l2.db

    l3.W -= L*l3.dW
    l3.b -= L*l3.db

    l4.W -= L*l4.dW
    l4.b -= L*l4.db

def get_predictions(A):
    return np.argmax(A,0)

def get_accuracy(predictions, Y):
    print(predictions,Y)
    return np.sum(predictions == Y) / Y.size