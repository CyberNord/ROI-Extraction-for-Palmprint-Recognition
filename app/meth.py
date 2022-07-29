import copy

import cv2
import numpy as np
from numpy import array

from app.constants import OTSU_LOWER, OTSU_HIGHER, PIXEL_OFFSET, PIXEL_OFFSET_NEG


# ### Preprocessing ###
# OTSU
def otsu(img):
    return cv2.threshold(img, OTSU_LOWER, OTSU_HIGHER, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# adds a frame to given picture
def add_frame(frame_size: int, img, value: array):
    return cv2.copyMakeBorder(
        img,
        top=frame_size,
        bottom=frame_size,
        left=frame_size,
        right=frame_size,
        borderType=cv2.BORDER_CONSTANT,
        value=value
    )


# ### Valley Point Detection ###
# roll
def move_matrix_left(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET_NEG, 1)


def move_matrix_right(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET, 1)


def move_matrix_up(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET_NEG, 0)


def move_matrix_down(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET, 0)


def move_matrix(binary_arr: array, pixel_offset: int, axis: int):
    print('pixel offset: ' + str(pixel_offset) + 'axis: ' + str(axis))
    roll = np.roll(np.copy(binary_arr), pixel_offset, axis=axis)
    roll[binary_arr > 0] = 0
    return roll


# logical conjunction of TPDTR
def logical_conjunction(left: array, right: array, up: array):
    translate_left = translate_to(left, 1)
    translate_right = translate_to(right, 1)
    translate_up = translate_to(up, 1)
    # multiplied = translate_up & translate_right & translate_left
    multiplied = translate_left*translate_right*translate_up
    return translate_to(multiplied, 255)


def translate_to(matrix: array, to: int):
    matrix[matrix > 0] = to
    return matrix
