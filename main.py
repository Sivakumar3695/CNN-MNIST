import os
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from time import time
import matplotlib.pyplot as plt
import statistics
from model import Model
from utils import preprocess_custom, feature_visualization


def train(model_cnn, x_train, y_train, lr):
    itr_data = []
    mean_loss = []
    median_loss = []
    print('NOTE: Training will be done with SGD for all 60000 data in Training dataset and '
          'evaluation will be done for all 10000 data in test dataset')
    epoch = int(input('Enter the number of training epoch:'))
    for epoch in range(epoch):
        itr_data.append(epoch + 1)
        loss_data = []
        for i in range(60000):
            data = x_train[i]
            output = y_train[i]

            loss = model_cnn.train(data, output, lr)
            print(str(epoch) + ":" + str(i) + ":Loss:" + str(loss))
            loss_data.append(loss)

        mean_loss.append(statistics.mean(loss_data))
        median_loss.append(statistics.median(loss_data))

    print_itr_loss_data(itr_data, mean_loss, median_loss)
    plot_graph(itr_data, mean_loss, median_loss)
    return loss


def plot_graph(itr_data, mean_loss, median_loss):
    fig = plt.figure(figsize=[8, 4], facecolor='white', constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(itr_data, mean_loss, 'ro-', linewidth=0.25, label='Mean')
    ax.plot(itr_data, median_loss, 'ko-', linewidth=0.25, label='Median')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.savefig('loss_graph.png', dpi=300)


def print_itr_loss_data(itr_data, mean_loss, median_loss):
    print('Itr data:')
    print(itr_data)
    print('Loss mean data')
    print(mean_loss)
    print('Median Loss')
    print(median_loss)


def print_evaluation_data(loss, confusion_matrix, start_time):
    t_positive = confusion_matrix[0]
    t_negative = confusion_matrix[1]
    f_positive = confusion_matrix[2]
    f_negative = confusion_matrix[3]

    if loss is not None:
        print("Final Loss: " + str(loss))
    print("Accuracy: " + str((t_positive + t_negative) / (t_positive + t_negative + f_positive + f_negative)))
    print("Precision: " + str(t_positive / (t_positive + f_positive)))
    print("Total time taken:" + str(time() * 1000 - start_time))


def evaluate_test_data_on_pre_trained(model_str=None):
    if model_str is None:
        print('Please provide a model to proceed!!')
        return
    # initialization
    start_time = time() * 1000
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test / 255
    y_test = np_utils.to_categorical(y_test)

    model_cnn = Model(x_test[0], pre_trained=True, model_str=model_str)
    confusion_matrix = model_cnn.evaluate(x_test, y_test, True)

    # stats print
    print_evaluation_data(None, confusion_matrix, start_time)


def train_and_evaluate_model(model_str):
    # initialization
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    # learning_rate = 0.0001
    learning_rate = float(input("Enter the learning rate:"))

    start_time = time() * 1000

    # train
    model_cnn = Model(x_train[0], model_str=model_str)
    loss = train(model_cnn, x_train, y_train, learning_rate)
    print("Training Done!!")

    # evaluation on test data
    confusion_matrix = model_cnn.evaluate(x_test, y_test)

    np.set_printoptions(threshold=np.inf)

    # save weights so as to use the trained model in future
    model_cnn.save_weight()

    # stats print
    print_evaluation_data(loss, confusion_matrix, start_time)


def predict_custom_images(model_str=None):
    if model_str is None:
        print('Please provide a model to proceed!!')
        return
    # initialization
    sample_img_dim = np.zeros((1, 28, 28))  # default MNIST dimension
    # model_cnn = Model(sample_img_dim, pre_trained=True)
    model_cnn = Model(sample_img_dim, pre_trained=True, model_str=model_str)

    '''
    load images from custom_test_images and do pre-processing as per the below information

    Important MNIST Information:
        1. All images are size normalized to fit in a 20x20 pixel box
        2. Digits are centered in a 28x28 image using the center of mass
        3. Digits are written in white color in black background
    '''
    images, image_files = preprocess_custom.get_pre_processed_images()

    # prediction for given custom images
    for i in range(len(images)):
        image_ = images[i]
        image_ = image_.reshape(1, 28, 28)
        predicted_n, percent = model_cnn.predict(image_)
        print("Actual Number:" + image_files[i] +
              ', Predicted:' + str(predicted_n) + ' with ' + str("{:.2%}".format(percent)) + ' confidence')


def cnn_functions():
    models = []
    i = 1
    for model_name in os.listdir('./weight_output'):
        model_name = model_name.replace('weight_details', '')
        model_name = model_name.replace('.npz', '')
        models.append(model_name)
        print(str(i) + '. ' + model_name)
        i += 1
    print(str(i) + '. To train a new model')
    user_option = int(input("Enter your option:"))
    if user_option < i:
        if user_option == 0:
            print('Please provide a model to proceed!!')
            return
        model_str = models[user_option - 1]
        print('1. Feature Visualization for the given image\n2. Feature Visualization for a given layer and channel\n'
              '3. Predict custom images\n4. Evaluate of MNIST test data on pre-trained model\n')
        user_option = int(input('Enter your option:'))
        if user_option == 1:
            feature_visualization.generate_feature_visualize_for_given_image(model_str)
        elif user_option == 2:
            layer_n = int(input('Enter the layer for feature visualization:'))
            layer_channel = int(input('Enter the channel for feature visualization:'))
            layer_channel = None if layer_channel == -1 else layer_channel
            feature_visualization.generate_feature_visualize(layer_n, model_str, layer_channel)
        elif user_option == 3:
            predict_custom_images(model_str)
        elif user_option == 4:
            evaluate_test_data_on_pre_trained(model_str)
        else:
            print('Not a valid option!!')
            return
    elif user_option == i:
        print('NOTE: If the model entered is invalid, training might fail')
        model_str = str(input('Enter the new model:'))
        if not model_str.endswith('-FC10'):
            print('Please enter a valid model.'
                  'The final layer should be FULLY CONNECTED with 10 units representing the output')
            return
        train_and_evaluate_model(model_str)


if __name__ == "__main__":
    cnn_functions()

