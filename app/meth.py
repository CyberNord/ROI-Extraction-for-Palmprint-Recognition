import cv2
import numpy as np
from numpy import array
from numpy.ma import copy

from app.constants import OTSU_LOWER, OTSU_HIGHER, PIXEL_OFFSET, PIXEL_OFFSET_NEG


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


# roll
def move_matrix_left(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET, 1)


def move_matrix_right(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET_NEG, 1)


def move_matrix_up(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET, 0)


def move_matrix_down(binary_arr: array):
    return move_matrix(binary_arr, PIXEL_OFFSET_NEG, 0)


def move_matrix(binary_arr: array, pixel_offset: int, axis: int):
    roll = np.roll(binary_arr, pixel_offset, axis=axis)
    binary_arr[roll > 0] = 0
    return binary_arr
