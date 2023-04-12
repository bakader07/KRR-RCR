# -*- coding: utf-8 -*-

import numpy as np

from model import NeuralNetwork
from layers import Layer, ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

X = np.array([[[0,0]],[[0,1]],[[1,0]],[[1,1]]])
Y = np.array([[[0]],[[1]],[[1]],[[0]]])

model = NeuralNetwork()

model.add(Layer(2, 3))
model.add(ActivationLayer(tanh, tanh_prime))

model.add(Layer(3, 1))
model.add(ActivationLayer(tanh, tanh_prime))

model.use(mse, mse_prime)
model.fit(X, Y, 1000, 0.01)

out = model.predict(X)
# to not print in scientific notation
np.set_printoptions(suppress = True)
print(out)