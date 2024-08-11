from Utils import *

data = pd.read_csv('train.csv')
data = np.array(data)
data_train = data.T
m,n = data.shape
X = data_train[1:n]
Y = data_train[0]
X = X /100


l1 = layers(784) # Input Layer
l2 = layers(10) # Hidden Layer
l3 = layers(10) # Hidden Layer
l4 = layers(10) # Output Layer

l2.weights(l1.layerSize)
l3.weights(l2.layerSize)
l4.weights(l3.layerSize)

l1.A = X

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