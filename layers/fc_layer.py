import numpy as np
from utils import activation_functions


def generate_weight(n_units_current_layer, n_units_prev_layer, initialization_tech='xavier'):
    if initialization_tech == 'he':
        return np.random.randn(n_units_current_layer, n_units_prev_layer) * np.sqrt(2 / n_units_prev_layer)
    return np.random.randn(n_units_current_layer, n_units_prev_layer) * np.sqrt(1 / n_units_prev_layer)


def generate_bias(n_units):
    bias = np.ones(n_units)
    return bias / 100


class FC_Layer:
    def __init__(self, n_units, n_units_prev_layer, pre_trained=False, activation_fn='relu'):
        self.__n_units = n_units
        self.__activation_fn = activation_fn
        self.z = None
        self.__input_ = None
        if not pre_trained:
            self.__weight = generate_weight(n_units, n_units_prev_layer)
            self.__bias = generate_bias(n_units)

    def get_weight(self):
        return self.__weight

    def get_bias(self):
        return self.__bias

    def set_weight(self, weight):
        assert len(weight) == self.__n_units
        self.__weight = weight

    def set_bias(self, bias):
        self.__bias = bias

    def get_n_units(self):
        return self.__n_units

    def get_activation_fn(self):
        return self.__activation_fn

    def get_output_for_fv(self):
        return self.z

    def forward(self, input_):
        self.__input_ = np.copy(input_)
        if input_.shape[0] != self.__weight[0].shape[0]:
            self.__input_ = input_.flatten()
        self.z = np.matmul(self.__input_, self.__weight.transpose()) + self.__bias
        if self.__activation_fn == 'relu':
            activation_neuron = activation_functions.relu(self.z)
        elif self.__activation_fn == 'softmax':
            activation_neuron = activation_functions.softmax(self.z)
        else:
            assert False  # no other activation function is supported
        return activation_neuron

    def backward(self, current_layer_output_err, learning_rate, adjust_weight=True):
        current_layer_err = current_layer_output_err
        if self.__activation_fn != 'softmax':  # if True => this is the final layer
            current_layer_err = current_layer_output_err * (activation_functions.relu_derivative(self.z)
                                                            if self.__activation_fn == 'relu' else 1)
        d_input_err = np.matmul(current_layer_err.reshape(1, len(self.__weight)), self.__weight, dtype=np.float128)

        if adjust_weight:
            self.__weight -= learning_rate * np.matmul(current_layer_err.reshape(len(self.__weight), 1),
                                                       self.__input_.reshape(1, len(self.__weight[0])),
                                                       dtype=np.float128)
            self.__bias -= current_layer_err.squeeze() * learning_rate
        return d_input_err
