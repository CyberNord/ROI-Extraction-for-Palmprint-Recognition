import copy

import cv2
import numpy as np
from numpy import array

from app.constants import OTSU_LOWER, OTSU_HIGHER, PIXEL_OFFSET, PIXEL_OFFSET_NEG, VALLEY_GAP_OFFSET, A_HORIZONTAL, \
    A_VERTICAL, M_CALCULATION, M_VISIBLE


# ### Preprocessing ###
# OTSU
def otsu(img):
    return cv2.threshold(img, OTSU_LOWER, OTSU_HIGHER, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# adds a frame to given picture
# the offset is chosen to match the main shifting offset
def add_frame(img, value: array):
    return cv2.copyMakeBorder(
        img,
        top=PIXEL_OFFSET,
        bottom=PIXEL_OFFSET,
        left=PIXEL_OFFSET,
        right=PIXEL_OFFSET,
        borderType=cv2.BORDER_CONSTANT,
        value=value
    )


# ### Valley Point Detection ###
# roll
def move_matrix_left(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET_NEG, A_HORIZONTAL)


def move_matrix_right(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET, A_HORIZONTAL)


def move_matrix_up(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET_NEG, A_VERTICAL)


def move_matrix(binary_arr: array, pixel_offset: int, axis: int):
    print('pixel offset: ' + str(pixel_offset) + 'axis: ' + str(axis))
    roll = np.roll(np.copy(binary_arr), pixel_offset, axis=axis)
    roll[binary_arr > 0] = 0
    return roll


# logical conjunction of TPDTR
def logical_conjunction(left: array, right: array, up: array):
    translate_left = translate_to(left, M_CALCULATION)
    translate_right = translate_to(right, M_CALCULATION)
    translate_up = translate_to(up, M_CALCULATION)
    multiplied = translate_left * translate_right * translate_up
    return translate_to(multiplied, M_VISIBLE)


def translate_to(matrix: array, to: int):
    matrix[matrix > 0] = to
    return matrix


# this function will take in the conjunction of TPDTR,
# move it down &conjunct it with the binary hand
# it will return the hand valley points
def get_valley_points(conj: array, binary_arr: array):
    roll = np.roll(conj, VALLEY_GAP_OFFSET, axis=A_VERTICAL)
    ret = translate_to(roll, 1) * translate_to(binary_arr, M_CALCULATION)
    return translate_to(ret, M_VISIBLE)
