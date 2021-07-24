import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_vec(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x.clip(min=0)


def relu_derivative(x):
    d_x = np.copy(x)
    d_x[d_x <= 0] = 0
    d_x[d_x > 0] = 1
    return d_x


def softmax(z):
    exponents = np.exp(z - np.max(z))
    exp_sum = np.sum(exponents)
    return exponents / exp_sum