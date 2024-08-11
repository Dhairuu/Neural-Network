from Utils import *

TestData = pd.read_csv('test.csv')
TestData = np.array(TestData)
TestData = TestData.T
m,n = TestData.shape

TestData_X = TestData[0:n]
TestData_X = TestData_X / 100

l1 = layers(784) # Input Layer
l2 = layers(10) # Hidden Layer
l3 = layers(10) # Hidden Layer
l4 = layers(10) # Output Layer

l2.weights(l1.layerSize)
l3.weights(l2.layerSize)
l4.weights(l3.layerSize)

l1.A = TestData_X

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

currentImage = TestData_X[:, index, None]
currentImage = currentImage.reshape((28,28)) * 255

plt.gray()
plt.imshow(currentImage)
plt.show()