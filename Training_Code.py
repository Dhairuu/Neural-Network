import numpy as np
import math as mt
import pandas as pd
import json as js
import array
data = pd.read_csv('train.csv')
data = np.array(data)
data_train = data.T
m,n = data.shape
X = data_train[1:n]
Y = data_train[0]
# print(n)
X = X /100
# print(Y.shape)


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

def ReLu(Z):  # Works only for 2D Arrays!!!
    # tempZ = []
    # for row in Z:
    #     arr = []
    #     for element in row:
    #         if element < 0:
    #             arr.append(0)
    #         else:
    #             arr.append(element)
    #     tempZ.append(arr)
    # return np.array(tempZ)
    return np.maximum(0,Z)

def sumEXP(row):
    value = 0.0
    for element in row:
        value += mt.exp(element)
    return value

def softMax(Z):
    # exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    # return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    # Subtract the maximum value for numerical stability
    # shift_Z = Z - np.max(Z, axis=1, keepdims=True)

    # # Compute exponentials
    # exp_shift_Z = np.exp(shift_Z)

    # # Normalize
    # return exp_shift_Z / np.sum(exp_shift_Z, axis=1, keepdims=True)

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
    # res = []
    # for row in Z:
    #     arr = []
    #     for element in row:
    #         if element < 0:
    #             arr.append(0)
    #         else:
    #             arr.append(1)
    #     res.append(arr)
    # return np.array(res)
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
    # print(l4.db.shape)
    # print(l4.W.T.shape, l4.dZ.shape, l3.Z.shape)
    l3.dZ = np.dot(l4.W.T, l4.dZ) * ReLu_derivative(l3.Z)
    # print(l4.W.T.shape, l4.dZ.shape, l3.Z.shape)
    l3.dW = np.dot(l3.dZ, l2.A.T) / m
    # print(l3.dZ.shape, X.T.shape)
    l3.db = np.sum(l3.dZ,axis=1) / m

    l2.dZ = np.dot(l3.W.T, l3.dZ) * ReLu_derivative(l2.Z)
    l2.dW = np.dot(l2.dZ, X.T) / m
    l2.db = np.sum(l2.dZ,axis=1) / m

    # l1.dZ = np.dot(l2.W.T, l2.dZ) * ReLu_derivative(l1.A)
    # l1.dW = np.dot(l1.dZ, X.T) / m
    # l1.db = np.sum(l1.dZ,axis=1) / m


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


l1 = layers(784) # Input Layer
l2 = layers(10) # Hidden Layer
l3 = layers(10) # Hidden Layer
l4 = layers(10) # Output Layer

l1.A = X

print(l1.A.shape)

l2.weights(l1.layerSize)
l3.weights(l2.layerSize)
l4.weights(l3.layerSize)


# forwardPropogation(l1,l2,l3,l4,m)

# print(f"Z2: {l2.Z.shape} A2: {l2.A.shape}\nZ3: {l3.Z.shape} A3: {l3.A.shape}\nZ4: {l4.Z.shape} A4: {l4.A.shape}\n")

# BackwardPropogation(l1,l2,l3,l4,Y,m)

# print(f"dZ2: {l2.dZ.shape} dW2: {l2.dW.shape} db2: {l2.db.shape}\ndZ3: {l3.dZ.shape} dW3: {l3.dW.shape} db3: {l3.db.shape}\ndZ4: {l4.dZ.shape} dW4: {l4.dW.shape} db4: {l4.db.shape}\n")

# print(l3.W.shape, l3.dW.shape)


L = 0.15
for ipoch in range(600):
    

    forwardPropogation(l1,l2,l3,l4,m)
    BackwardPropogation(l1,l2,l3,l4,Y,m)

    gradientDescent(l1,l2,l3,l4,L)
    if ipoch % 50 == 0:
        print(ipoch)
        print("Accuracy: ", get_accuracy(get_predictions(l4.A), Y))

print("Accuracy: ", get_accuracy(get_predictions(l4.A), Y))


with open('parameters.json','r') as file:
    Js_Object = js.load(file)


with open('parameters.json','w') as file:
    Js_Object['W2'] = l2.W.tolist()
    Js_Object['B2'] = l2.b.tolist()
    Js_Object['W3'] = l3.W.tolist()
    Js_Object['B3'] = l3.b.tolist()
    Js_Object['W4'] = l4.W.tolist()
    Js_Object['B4'] = l4.b.tolist()
    js.dump(Js_Object,file)



def one_hot(Y,n):
    one_hot_Y = []
    for i in Y:
        arr = []
        for j in range(n):
            if j == i:
                arr.append(1)
            else:
                arr.append(0)
        one_hot_Y.append(arr)
    return np.array(one_hot_Y)

# one_hot_Y = one_hot(Y,10).T

