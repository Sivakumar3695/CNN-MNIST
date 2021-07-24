import numpy as np
from layers.convolve import Convolve
from layers.max_pooling import MaxPooling
from layers.fc_layer import FC_Layer as FCLayer
from layers.input_img import InputImage
import os
import matplotlib.pyplot as plt


def loss_calculation(output, predicted):
    error_predicted = predicted - output
    loss = np.sum(-1 * np.multiply(output, np.log(predicted + 1e-15), dtype=np.float128))
    return error_predicted, loss


def generate_bar_graph(data1):
    # set width of bar
    bar_width = 0.5
    fig = plt.subplots(figsize=(16, 4))
    plt.rcParams['font.size'] = 5
    plt.rcParams['text.color'] = '#8d8c8c'
    # Set position of bar on X axis
    bar_height = np.arange(len(data1))
    # Make the plot
    plt.bar(bar_height, data1, color='r', width=bar_width)

    # Adding x_ticks
    plt.xlabel('Layer no.', fontsize=7)
    plt.ylabel('Time taken (s)', fontsize=7)
    plt.xticks([r for r in range(len(data1))],
               [x + 1 for x in range(len(data1))])

    plt.tick_params(
        axis='x', labelsize=5, length=0, width=0,
        labelcolor='#8d8c8c'
    )
    plt.tick_params(
        axis='y', labelsize=5, length=0, width=0,
        labelcolor='#8d8c8c'
    )
    plt.savefig("./feature_visualize/output/mean_activation_bar.jpg", dpi=300)
    plt.close()


class Model:
    def __init__(self, sample_input_img, model_str=None, pre_trained=False):
        if model_str is None:
            model_str = 'C(8,5)-P2-C(16,3)-P2-FC128-FC10'
        # print(model_str)
        self.model = model_str
        self.pre_trained = pre_trained
        self.obj_list = self.get_obj_list(sample_input_img)
        if pre_trained:
            self.load_weight()

    def get_obj_list(self, sample_input_img):
        obj_list = [InputImage(sample_input_img)]
        layer_list = self.model.split(sep='-')
        for i in range(len(layer_list)):
            layer = layer_list[i]
            prev_obj = obj_list[len(obj_list) - 1]
            if layer[0] == 'C':
                details = layer[layer.find("(") + 1:layer.find(")")].split(',')
                obj_list.append(Convolve(prev_obj, int(details[0]), int(details[1]), pre_trained=self.pre_trained))
            elif layer[0] == 'P':
                obj_list.append(MaxPooling(prev_obj, int(layer[1])))
            else:
                fc_units = int(layer.replace('FC', ''))
                activation_fn = 'softmax' if i == (len(layer_list) - 1) else 'relu'
                obj_list.append(FCLayer(fc_units, prev_obj.get_n_units(), activation_fn=activation_fn,
                                        pre_trained=self.pre_trained))
        return obj_list

    def predict(self, input_):
        prediction = self.__forward(input_)
        # print(prediction)
        predicted_n = prediction.argmax()
        percent = prediction.max()
        return predicted_n, percent

    def train(self, input_, output, lr):
        predicted_output = self.__forward(input_)
        prediction_err, loss = loss_calculation(output, predicted_output)
        self.__backward(prediction_err, lr)
        return loss

    def evaluate(self, x_test, y_test, print_detailed_failure=False):
        t_positive = 0
        f_positive = 0
        f_negative = 0
        digit_failure_count = np.zeros(10)
        for epoch in range(1):
            for i in range(len(x_test)):
                data = x_test[i]
                output = y_test[i]

                prediction = self.__forward(data)
                print("***********************************************************")
                print("ACTUAL: ", output)
                print("PREDICTED: ", prediction)
                print("***********************************************************")
                if output.argmax() == prediction.argmax():
                    t_positive += 1
                else:
                    digit_failure_count[output.argmax()] += 1
                    f_positive += 1
                    f_negative += 1

        t_negative = (len(prediction) - 1) * t_positive + (len(prediction) - 2) * f_positive
        if print_detailed_failure:
            print('Detailed Failure data:')
            print(digit_failure_count)
        return [t_positive, t_negative, f_positive, f_negative]

    def __forward(self, input_):
        next_layer_input = np.copy(input_)
        for i in range(1, len(self.obj_list)):
            layer_obj = self.obj_list[i]
            output = layer_obj.forward(next_layer_input)
            next_layer_input = output
        return output

    def __backward(self, prediction_err, lr):
        layer_err = np.copy(prediction_err)
        for i in range(len(self.obj_list) - 1, 0, -1):
            layer_obj = self.obj_list[i]
            if i > 1:
                prev_layer_output_err = layer_obj.backward(layer_err, lr)
                layer_err = prev_layer_output_err
            else:
                layer_obj.backward(layer_err, lr, is_prev_layer_orig_img=True)
                return

    def save_weight(self):
        if not os.path.exists('./weight_output/'):
            os.mkdir('./weight_output')
        file_name = 'weight_details' + self.model + '.npz'
        file_path = './weight_output/' + file_name
        if os.path.exists(file_path):
            os.remove(file_path)
        weight_dict = dict()
        for i in range(1, len(self.obj_list)):
            layer = self.obj_list[i]
            weight = layer.get_weight()
            bias = layer.get_bias()
            if weight is not None:
                weight_dict['weight_' + str(i)] = weight
                weight_dict['bias_' + str(i)] = bias
        np.savez(file_path, **weight_dict)

    def load_weight(self):
        if not os.path.exists('./weight_output'):
            os.mkdir('./weight_output')
        file_name = 'weight_details' + self.model + '.npz'
        file_path = './weight_output/' + file_name

        if not os.path.exists(file_path):
            print("No trained weight for the given model!!")
            raise Exception("No trained weight for the given model!!")
        loaded_weight = np.load(file_path)
        for i in range(1, len(self.obj_list)):
            if loaded_weight.get('weight_' + str(i)) is not None:
                layer = self.obj_list[i]
                layer.set_weight(loaded_weight['weight_' + str(i)])
                layer.set_bias(loaded_weight['bias_' + str(i)])

    def feature_visualize_till(self, input_, layer_n, layer_channel=None):
        next_layer_input = np.copy(input_)
        for i in range(1, layer_n + 1):
            layer = self.obj_list[i]
            next_layer_input = layer.forward(next_layer_input)
            output_for_fv = next_layer_input  # layer.get_output_for_fv()
        if layer_channel is None:
            err = output_for_fv * 1
            activation_unit_sum = np.sum(output_for_fv)
        else:
            activation_unit_sum = np.sum(output_for_fv[layer_channel])
            err_filter = np.zeros(output_for_fv.shape)
            err_filter[layer_channel] += 1
            err = output_for_fv * err_filter
        # print(activation_unit_sum)
        if 0 < activation_unit_sum <= 1:
            lr = 0.5
            sign = 1 if activation_unit_sum > 0 else -1
            while sign * lr * activation_unit_sum <= 0.001:
                lr = lr * 10
        else:
            lr = 0.05
        for i in range(layer_n, 0, -1):
            layer = self.obj_list[i]
            err = layer.backward(err, lr, adjust_weight=False)
        input_ = input_ + lr * err
        return input_, True if activation_unit_sum >= 10000 or activation_unit_sum == 0 else False

    def get_fv_data_for_given_img(self, img, layer_check=6):
        next_layer_input = np.copy(img)
        mean_activation_values = []
        layer = []
        channel = []
        for i in range(1, len(self.obj_list)):
            layer_obj = self.obj_list[i]
            output = layer_obj.forward(next_layer_input)
            if i == layer_check:
                if len(output.shape) == 3:
                    total_n_val = len(output[0]) * len(output[0][0])
                else:
                    total_n_val = len(output)
                for j in range(len(output)):
                    mean_activation_values.append(np.sum(output[j]) / total_n_val)
                    layer.append(i)
                    channel.append(j)
            next_layer_input = output
        generate_bar_graph(mean_activation_values)
        print(output)
        print('Prediction:' + str(output.argmax()))
        return [mean_activation_values, layer, channel]
