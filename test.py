# -*- coding: utf-8 -*-

import numpy as np

alpha=0.1
bias=-0.2

def step(n):
    return int(n>0)
class Model:
    def propagation(self, x):
        print('forward propagation')
        net_input = np.dot(x, self.weights) + bias
        output = np.vectorize(step)(net_input)
        print('output =', output.flatten())
        return output

    def back(self, x, y, output):
        print('backward propagation')
        error = y - output
        delta = np.dot(x.T, error) * alpha
        print("delta:",delta)
        self.weights += delta
        print("weights:", self.weights.flatten())

    def train(self, x, y, epochs):
        print('training')
        self.weights = np.array([[0.3],[-0.1]])
        for i in range(epochs):
            print("="*10,"epoch:", i, "="*10)
            output = self.propagation(x)
            self.back(x, y, output)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
andY = np.array([[0],[0],[0],[1]])
orY = np.array([[0],[1],[1],[1]])
# xorY = np.array([[0],[1],[1],[0]])

epochs=20

m = Model()

# Y = andY
Y = orY
# Y = xorY

m.train(X, Y, epochs)
