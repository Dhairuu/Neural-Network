import pandas as pd
import numpy as np
import json as js
import math as mt
from matplotlib import pyplot as plt

TestData = pd.read_csv('test.csv')
TestData = np.array(TestData)
TestData = TestData.T
m,n = TestData.shape
# print(n)
Test_X = TestData[0:n]
# Y = TestData[0]
Test_X = Test_X / 100

print(Test_X.shape)

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
        # self.init()

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
    tempZ = np.array(Z)
    tempZ = tempZ.T
    res = []
    for row in tempZ:
        sum = sumEXP(row)
        arr = []
        for element in row:
            arr.append(round(mt.exp(element) / sum, 2))
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

def get_predictions(A):
    return np.argmax(A,0)

def get_accuracy(predictions, Y):
    print(predictions,Y)
    return np.sum(predictions == Y) / Y.size

l1 = layers(784) # Input Layer
l2 = layers(10) # Hidden Layer
l3 = layers(10) # Hidden Layer
l4 = layers(10) # Output Layer

l1.A = Test_X

l2.weights(l1.layerSize)
l3.weights(l2.layerSize)
l4.weights(l3.layerSize)

with open ('parameters.json','r') as file:
    Js_Object = js.load(file)

l2.W = np.array(Js_Object['W2'])
l2.b = np.array(Js_Object['B2'])

l3.W = np.array(Js_Object['W3'])
l3.b = np.array(Js_Object['B3'])

l4.W = np.array(Js_Object['W4'])
l4.b = np.array(Js_Object['B4'])

forwardPropogation(l1,l2,l3,l4,n)


Result = get_predictions(l4.A)

index = input("Enter index: ")
index = int(index)
print(f"Prediction: {Result[index]}")

currentImage = Test_X[:, index, None]
currentImage = currentImage.reshape((28,28)) * 255

plt.gray()
plt.imshow(currentImage)
plt.show()


# err = 0
# for i in range(m):
    # if(Result[i] != Y[i]):
        # err+=1

# print(err/m *100)