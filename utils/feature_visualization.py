import os
import cv2
import numpy as np
from model import Model


def generate_feature_visualize_for_given_image(model_str, n_fz_images=5):
    """
    :param model_str: a model using which the feature visualize images are generated.
                        Please make sure the model is trained before using this method
    :param n_fz_images: Number of feature visualize images required.

    Please make sure to put the input image in the folder mentioned below
    Folder: './feature_visualize/input.jpg'
    Input image format: jpg

    Following images will be generated:
    1. mean_activation_bar.jpg in the folder ./feature_visualize/output
    2. 'n_fz_images' number of feature visualize images in the folder ./feature_visualize/output
    3. collages_feature_visualize.jpg in the folder ./feature_visualize/output
        - for better comparison of the input image and the feature visualized image to draw conclusion
          on how the algorithm made a decision for the given input image
        - colored number shown in the collage image are just for comparison.
          Prediction and procession are done with the original image with black background and white text

    NOTE: Everytime this function is called, './feature_visualize/output' folder will be deleted and newly created.

    :return:
    """
    input_path = './feature_visualize/input.jpg'
    img = cv2.imread(input_path, 0)
    if img is None:
        print("Please place the input image in this place :=> ./feature_visualize/input.jpg")
        return
    img = img.reshape(1, 28, 28)
    model_cnn = Model(img, pre_trained=True, model_str=model_str)
    if os.path.exists('./feature_visualize/output'):
        os.system('rm -rf ./feature_visualize/output')
    os.mkdir('./feature_visualize/output')
    activation_unit_data = model_cnn.get_fv_data_for_given_img(img, layer_check=8)

    mean_activation_data = np.copy(activation_unit_data[0])
    layer_arr = activation_unit_data[1]
    channel_arr = activation_unit_data[2]

    max_channel_itr = n_fz_images
    while max_channel_itr > 0:
        current_max_arg = mean_activation_data.argmax()
        layer_to_be_checked = layer_arr[current_max_arg]
        channel_to_be_checked = channel_arr[current_max_arg]
        print('Layer:' + str(layer_to_be_checked) + ', Channel:' + str(channel_to_be_checked) +
              ', Val:' + str(mean_activation_data.max()))

        mean_activation_data[current_max_arg] = 0

        generate_feature_visualize(layer_to_be_checked, model_str, channel_to_be_checked,
                                   (n_fz_images - max_channel_itr + 1))
        max_channel_itr -= 1

    if not os.path.exists('./feature_visualize/output'):
        print('No Visualize output exists!')
        return

    colored_input = cv2.imread(input_path, 0)
    colored_input[colored_input > 75] = 75
    for x in range(0, 28, 9):
        cv2.line(colored_input, (x, 0), (x, 27), (100, 0, 0), 1, 1)
    for x in range(0, 28, 9):
        cv2.line(colored_input, (0, x), (27, x), (100, 0, 0), 1, 1)
    # cv2.imwrite('./feature_visualize/output/colored_input_img.png', colored_input.astype(float))
    combined_output = np.copy(colored_input)
    for i in range(n_fz_images):
        path = './feature_visualize/output/feature_visualize_' + str(i+1) + '.png'
        img_obj = cv2.imread(path, 0)
        blend_img = cv2.addWeighted(colored_input, 0.5, img_obj, 0.5, 0.0)
        combined_output = np.concatenate((combined_output, blend_img), axis=1)

    cv2.imwrite('./feature_visualize/output/collages_feature_visualize.jpg', combined_output)


def generate_feature_visualize(layer_n, model_str, channel_n=None, vis_imn_n=1):
    out_dim = 28
    img = np.uint8(np.random.uniform(150, 151, (out_dim, out_dim, 1)))  # generate random image
    if not os.path.exists('./feature_visualize/output/'):
        os.mkdir('./feature_visualize/output/')
    img = img.reshape(1, out_dim, out_dim)
    feature_img = np.copy(img / 255)

    model_cnn = Model(feature_img, pre_trained=True, model_str=model_str)
    for i in range(4000):
        feature_img, stop_itr = model_cnn.feature_visualize_till(feature_img, layer_n, layer_channel=channel_n)
        if stop_itr:
            print('Reached')
            break

    feature_img[feature_img < 0] = 0
    cv2.imwrite('./feature_visualize/output/feature_visualize_' + str(vis_imn_n) + '.png', feature_img
                .reshape(out_dim, out_dim, 1).astype(int))
