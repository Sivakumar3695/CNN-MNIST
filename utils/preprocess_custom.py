import math
import cv2
import numpy as np
import os
from scipy import ndimage
import  matplotlib.pyplot as plt


def get_pre_processed_images():
    images = []
    image_file_names = []
    os.system('rm -rf ./custom_test_images/resized && mkdir -p ./custom_test_images/resized')
    for filename in os.listdir('./custom_test_images'):
        image_ = cv2.imread('./custom_test_images/' + filename, 0)
        if image_ is not None:
            img = 255 - image_
            pixel_mean = np.mean(img)
            threshold = 0
            if pixel_mean > 150:
                threshold = 200
            # elif pixel_mean > 110:
            elif pixel_mean > 100:
                threshold = 190
                # threshold = 190
            elif pixel_mean > 80:
                threshold = 150
            else:
                threshold = 95
            img[img > threshold] = 255
            img[img < threshold] = 0
            # test[test < 220] = 0

            # img = cv2.threshold(image_, threshold, 255, cv2.THRESH_BINARY)[1]
            # if img.min() == 255:
            #     img = cv2.threshold(image_, 150, 255, cv2.THRESH_BINARY)[1]
            # print(filename + ' ' + str(np.sum(image_)))

            # cv2.imwrite('./custom_test_images/resized/' + filename, img)
            # print(filename + ' : ' + str(pixel_mean))
            # continue

            img = crop_img_to_take_main_content(img, 1020)

            if img.shape[0] < 5 or img.shape[1] < 5:
                img = cv2.threshold(image_, 150, 255, cv2.THRESH_BINARY)[1]
                img = 255 - img

                img = crop_img_to_take_main_content(img, 500)

            img = resize_to_20px(img)
            img = fit_20px_to_28px(img)

            # shift image so that the main content will be centered
            shift_x, shift_y = get_best_shift(img)
            img = shift(img, shift_x, shift_y)

            cv2.imwrite('./custom_test_images/resized/' + filename, img)

            images.append(img / 255)
            image_file_names.append(filename)

    return images, image_file_names


def crop_img_to_take_main_content(img, pixel_summation_threshold):
    cy, cx = ndimage.measurements.center_of_mass(img)
    row_top = int(cy)
    row_bottom = int(cy)

    col_left = int(cx)
    col_right = int(cx)
    prev_sum = np.sum(img[row_top, :])
    while np.sum(img[row_top, :]) != 0 and (np.sum(img[row_top, :]) / prev_sum) > 0.3:
        prev_sum = np.sum(img[row_top, :])
        row_top = row_top - 1
    prev_sum = np.sum(img[row_bottom, :])
    while np.sum(img[row_bottom, :]) != 0 and (np.sum(img[row_bottom, :])) / prev_sum > 0.3:
        prev_sum = np.sum(img[row_bottom, :])
        row_bottom = row_bottom + 1
    prev_sum = np.sum(img[:, col_left])
    while np.sum(img[:, col_left]) != 0 and np.sum(img[:, col_left]) / prev_sum > 0.3:
        prev_sum = np.sum(img[:, col_left])
        col_left = col_left - 1
    prev_sum = np.sum(img[:, col_right])
    while np.sum(img[:, col_right]) != 0 and np.sum(img[:, col_right]) / prev_sum > 0.3:
        prev_sum = np.sum(img[:, col_right])
        col_right = col_right + 1

    del_bottom = (len(img) - row_bottom) * -1
    del_right = (len(img[0]) - col_right) * -1
    img = img[row_top:]
    img = img[:del_bottom]
    img = img[:, col_left:]
    img = img[:, :del_right]

    # while np.sum(img[0]) <= pixel_summation_threshold and img.shape[0] >= 5:
    #     img = img[1:]
    #
    # while np.sum(img[:, 0]) <= pixel_summation_threshold and img.shape[1] >= 5:
    #     img = np.delete(img, 0, 1)
    #
    # while np.sum(img[-1]) <= pixel_summation_threshold and img.shape[0] >= 5:
    #     img = img[:-1]
    #
    # while np.sum(img[:, -1]) <= pixel_summation_threshold and img.shape[1] >= 5:
    #     img = np.delete(img, -1, 1)

    return img


def fit_20px_to_28px(img):
    rows, cols = img.shape
    column_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    row_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (row_padding, column_padding), 'constant')
    return img


def resize_to_20px(img):
    rows, cols = img.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))
    return img


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    shift_matrix = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, shift_matrix, (cols, rows))
    return shifted
