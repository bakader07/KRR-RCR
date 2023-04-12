# -*- coding: utf-8 -*-

import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return np.array(result)

    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # loss
                err += self.loss(y_train[j], output)

                # backward
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # average error
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
