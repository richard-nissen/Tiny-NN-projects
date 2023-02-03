import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x                                         # X input
        self.weights1   = np.random.rand(self.input.shape[1],4)     # Random starting weights
        self.weights2   = np.random.rand(4,1)                       # Random starting weights
        self.y          = y                                         # Y output
        self.output     = np.zeros(y.shape)                         # Output

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))                                     # Activation function

    def sigmoid_derivative(self, x):
        return x * (1 - x)                                          # Derivative of sigmoid function

    
    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))       # Calculate layer 1
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))      # Calculate output

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output))) 
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def loss(self):
        return np.sum((self.y - self.output)**2)

    def predict(self, x):
        layer1 = self.sigmoid(np.dot(x, self.weights1))
        output = self.sigmoid(np.dot(layer1, self.weights2))
        return output

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)
    loss = []


    for i in range(1500):
        nn.feedforward()
        nn.backprop()
        loss.append(nn.loss())
    print(nn.output)

    plt.plot(loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.show()
