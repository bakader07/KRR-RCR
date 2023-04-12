# -*- coding: utf-8 -*-

import numpy as np

class Layer:
    def __init__(self, nodes, shape=1, activation='linear'):
        self.activation = activation
        self.nodes = np.array([self.node(shape) for i in range(nodes)])
        print('Layer nodes:')
        print(self.nodes)

    def node(self, shape):
        return np.random.rand(shape)


class NeuralNet:
    def __init__(self):
        self.weights = []
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, epochs):
        for i in range(epochs):
            print('Epoch:', i+1)
        self.input = x
        # self.weights[0] = np.random.rand(self.input.shape[1],4)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def predict(self, input):
        print(input)
