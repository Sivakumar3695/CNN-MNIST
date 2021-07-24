import numpy as np
from utils import activation_functions
import math


def slice_convert_sub_arr_to_matrix(data, filter_width, filter_breadth, stripe, operation):
    data_width = len(data[0]) if operation != 'full_convolution' else len(data[0][0])
    data_breadth = len(data[0][0]) if operation != 'full_convolution' else len(data[0][0][0])

    if operation == 'pooling':
        conv_output_w = int(data_width / filter_width)
        conv_output_b = int(data_breadth / filter_breadth)
    else:
        conv_output_w = data_width - filter_width + 1
        conv_output_b = (data_width - filter_width + 1)

    h_axis_calc = np.repeat(np.arange(filter_width), filter_width).reshape(-1, 1) \
                  + np.repeat(np.arange(conv_output_w) * stripe, conv_output_w).reshape(1, -1)
    v_axis_calc = np.tile(np.arange(filter_width), filter_breadth).reshape(-1, 1) \
                  + np.tile(np.arange(conv_output_w) * stripe, conv_output_b).reshape(1, -1)
    if operation == 'full_convolution':
        return_matrix_t = data[:, :, h_axis_calc, v_axis_calc]
        return_matrix_t = np.concatenate(return_matrix_t, axis=1)[:]
    elif operation == 'd_filter_convolution' or operation == 'pooling':
        return_matrix_t = data[:, h_axis_calc, v_axis_calc]
    else:
        return_matrix_t = np.concatenate(data[:, h_axis_calc, v_axis_calc])[:]
    return return_matrix_t


def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    n, c, h, w = x_shape
    assert (h + 2 * padding - field_height) % stride == 0
    assert (w + 2 * padding - field_height) % stride == 0
    out_height = (h + 2 * padding - field_height) / stride + 1
    out_width = (w + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * c)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(c, dtype='int32'), field_height * field_width).reshape(-1, 1)
    return k, i, j


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    n, c, h, w = x_shape
    h_padded, w_padded = h + 2 * padding, w + 2 * padding
    x_padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(c * field_height * field_width, -1, n)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


class Convolve:
    def __init__(self, prev_layer_obj, n_channel, filter_l, stride=1, padding=0, pre_trained=False,
                 initialization='he'):
        self.__filter_l = filter_l
        self.__n_channel = n_channel
        self.__stride = stride
        self.__padding = padding
        self.__dim = math.floor(((prev_layer_obj.get_dim() + (2 * padding) - filter_l) / stride)) + 1
        self.__vec_op_matrix = None
        self.z = None
        self.shape = None
        if not pre_trained:
            self.__filter = self.generate_filter(prev_layer_obj, initialization)
            self.__bias = self.generate_bias()

    def get_weight(self):
        return self.__filter

    def get_activation_fn(self):
        return 'relu'

    def get_output_for_fv(self):
        return self.z

    def get_bias(self):
        return self.__bias

    def set_weight(self, filter_):
        assert len(filter_) == self.__n_channel and len(filter_[0][0]) == self.__filter_l
        self.__filter = filter_

    def set_bias(self, bias):
        self.__bias = bias

    def get_n_channel(self):
        return self.__n_channel

    def get_n_units(self):
        return self.__dim * self.__dim * self.__n_channel

    def get_dim(self):
        return self.__dim

    def generate_filter(self, prev_layer_obj, initialization_tech='he'):
        n_channel_prev_layer = prev_layer_obj.get_n_channel()
        n_units_prev_layer = prev_layer_obj.get_n_units()
        if initialization_tech == 'xavier':
            np.random.randn(self.__n_channel, n_channel_prev_layer, self.__filter_l, self.__filter_l) \
            * np.sqrt(1 / n_units_prev_layer)
        return np.random.randn(self.__n_channel, n_channel_prev_layer, self.__filter_l, self.__filter_l) \
               * np.sqrt(2 / n_units_prev_layer)

    def generate_bias(self):
        bias = np.ones(self.__n_channel)
        return bias / 100

    def forward(self, data):
        data_channel_len = len(data)
        data_width = len(data[0])
        data_breadth = len(data[0][0])
        self.shape = (data_channel_len, data_width, data_breadth)

        filter_channel_len = len(self.__filter)
        filter_width = len(self.__filter[0][0])
        filter_breadth = len(self.__filter[0][0][0])

        self.__vec_op_matrix = slice_convert_sub_arr_to_matrix(data, filter_width, filter_breadth, self.__stride,
                                                               'convolution')

        vectorized_filter = np.zeros((filter_width * filter_breadth * data_channel_len, filter_channel_len))
        for i in range(filter_channel_len):
            vectorized_filter[:, i] = self.__filter[i].reshape(1, filter_width * filter_breadth * data_channel_len)
        out = (np.matmul(vectorized_filter.transpose(), self.__vec_op_matrix, dtype=np.float128)) \
              + self.__bias.reshape(filter_channel_len, 1)
        self.z = out.reshape(filter_channel_len, data_width - filter_width + 1, data_breadth - filter_breadth + 1)
        activated_out = activation_functions.relu(self.z)
        return activated_out

    def backward(self, error_current_layer_output, learning_rate, is_prev_layer_orig_img=False, adjust_weight=True):
        current_layer_err = error_current_layer_output * activation_functions.relu_derivative(self.z)

        filter_w = len(self.__filter[0][0])
        filter_b = len(self.__filter[0][0][0])

        err_current_layer_flat = current_layer_err.transpose(0, 1, 2).reshape(len(self.__filter), -1)

        d_input_neuron_err = None
        if not is_prev_layer_orig_img:
            weight_flat = self.__filter.reshape(len(self.__filter), -1)
            d_input_neuron_err = weight_flat.T @ err_current_layer_flat
            shape = (1, self.shape[0], self.shape[1], self.shape[2])
            d_input_neuron_err = col2im_indices(d_input_neuron_err, shape, filter_w, filter_b, 0, 1).squeeze()

        if adjust_weight:
            filter_adjuster = err_current_layer_flat @ self.__vec_op_matrix.T
            filter_adjuster = filter_adjuster.reshape(self.__filter.shape)
            self.__filter -= learning_rate * filter_adjuster

            self.__bias -= learning_rate * (np.sum(current_layer_err.reshape(len(current_layer_err),
                                                                             current_layer_err[0].size), axis=1))

        return d_input_neuron_err
