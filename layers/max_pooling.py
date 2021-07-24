import math
import numpy as np


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


class MaxPooling:
    def __init__(self, prev_layer_obj, filter_l, padding='same'):
        self.__filter_l = filter_l
        self.__n_channel = prev_layer_obj.get_n_channel()
        self.__padding = padding
        self.__input = None
        self.__output = None
        self.__pad_width = 0
        self.__pad_breadth = 0
        self.__dim = math.floor(prev_layer_obj.get_dim() / filter_l) if padding == 'valid' \
            else math.ceil(prev_layer_obj.get_dim() / filter_l)

    def get_weight(self):
        return None

    def get_bias(self):
        return None

    def get_n_channel(self):
        return self.__n_channel

    def get_n_units(self):
        return self.__dim * self.__dim * self.__n_channel

    def get_dim(self):
        return self.__dim

    def get_output_for_fv(self):
        return self.__output

    def forward(self, input_):
        self.__pad_width = 0
        self.__pad_breadth = 0
        if self.__padding == 'same':
            if len(input_[0]) % self.__filter_l != 0:
                self.__pad_width = self.__filter_l - (len(input_[0]) % self.__filter_l)
            if len(input_[0][0]) % self.__filter_l != 0:
                self.__pad_breadth = self.__filter_l - (len(input_[0][0]) % self.__filter_l)
        padded_data = np.pad(input_, [(0, 0), (0, self.__pad_width), (0, self.__pad_breadth)], mode='constant')
        self.__input = padded_data

        data_channel_len = len(padded_data)
        data_width = len(padded_data[0])
        data_breadth = len(padded_data[0][0])

        output_width = int(data_width / self.__filter_l)
        output_breadth = int(data_breadth / self.__filter_l)

        out = slice_convert_sub_arr_to_matrix(padded_data, self.__filter_l, self.__filter_l,
                                              self.__filter_l, 'pooling').max(1)
        self.__output = out.reshape(data_channel_len, output_width, output_breadth)
        return self.__output

    def backward(self, err_current_layer_output, lr, adjust_weight=True):
        err_current_layer_output = err_current_layer_output.reshape(self.__n_channel,
                                                                    self.__dim,
                                                                    self.__dim)
        d_input_neuron_err = np.equal(self.__input, self.__output
                                      .repeat(self.__filter_l, axis=1)
                                      .repeat(self.__filter_l, axis=2)).astype(int) \
                             * err_current_layer_output \
                                 .repeat(self.__filter_l, axis=1) \
                                 .repeat(self.__filter_l, axis=2)
        row_pad = self.__pad_width
        col_pad = self.__pad_breadth
        while row_pad > 0:
            d_input_neuron_err = d_input_neuron_err[:, :-1, :]
            row_pad -= 1
        while col_pad > 0:
            d_input_neuron_err = np.delete(d_input_neuron_err, -1, 2)
            col_pad -= 1
        return d_input_neuron_err
