import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import array

from app.constants import OTSU_LOWER, OTSU_HIGHER, PIXEL_OFFSET, PIXEL_OFFSET_NEG, VALLEY_GAP_OFFSET, A_HORIZONTAL, \
    A_VERTICAL, M_CALCULATION, M_VISIBLE, V_ALPHA, V_BETA, V_GAMMA, V_CIRCLE_COLOR, V_CHECKPOINT_COLOR, V_TEST_COLOR


# ### Preprocessing ###
# OTSU
def otsu(img):
    return cv2.threshold(img, OTSU_LOWER, OTSU_HIGHER, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# adds a frame to given picture (causes Errors)
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


def out_of_bonds(coordinates, height, width):
    if 0 <= coordinates[0] <= height - 1 and 0 < coordinates[1] < width - 1:
        return False
    return True


# Four-point check
# top is black & right, bottom, left are white
def get_cond1(binary_arr: array, c: tuple, height: int, width: int):
    # clockwise top, right, bottom, left
    arr = ((c[0], c[1] - V_ALPHA),
           (c[0] + V_ALPHA, c[1]),
           (c[0], c[1] + V_ALPHA),
           (c[0] - V_ALPHA, c[1]))
    for coordinates in arr:
        if out_of_bonds(coordinates, height, width):
            return False

    # if binary_arr[arr[0]] == 0 and binary_arr[arr[1]] == 255 and binary_arr[arr[2]] == 255 and binary_arr[arr[3]] == 255:
    if binary_arr[arr[0]] == 0 :
        # print(f'Center={c}, up={binary_arr[arr[0]]},right={binary_arr[arr[1]]},down={binary_arr[arr[2]]},left={binary_arr[arr[3]]}')
        return True
    # print(f'up={binary_arr[arr[0]]},right={binary_arr[arr[1]]},down={binary_arr[arr[2]]},left={binary_arr[arr[3]]}')
    return False


def get_cond2(binary_arr: array, c: tuple, height: int, width: int):
    radius = (V_ALPHA + V_BETA)
    diag = int(radius * 0.7)

    # clockwise 8-positions
    # top, top right, right, bottom right, bottom, bottom left, left, top left
    arr = ((c[0], c[1] - radius),
           (c[0] + diag, c[1] - diag),
           (c[0] + radius, c[1]),
           (c[0] + diag, c[1] + diag),
           (c[0], c[1] + radius),
           (c[0] - diag, c[1] + diag),
           (c[0] - radius, c[1]),
           (c[0] - diag, c[1] - diag))

    count = 0
    for coordinates in arr:
        if out_of_bonds(coordinates, height, width):
            return False
        if binary_arr[coordinates] == 0:
            count += 1

    if 0 < count <= 4:
        return True
    else:
        return False


def get_cond3(binary_arr: array, c: tuple, height: int, width: int):
    radius = (V_ALPHA + V_BETA + V_GAMMA)
    diag = int(radius * 0.7)
    off_16_a = int(radius * 0.9)
    off_16_b = int(radius * 0.4)

    # clockwise 16-positions (x-pos)
    # top, x1, top right, x2, right, x3, bottom right, x4, bottom, x5, bottom left, x6, left, x7, top left, x8
    arr = ((c[0], c[1] - radius),
           (c[0] + off_16_b, c[1] - off_16_a),
           (c[0] + diag, c[1] - diag),
           (c[0] + off_16_a, c[1] - off_16_b),
           (c[0] + radius, c[1]),
           (c[0] + off_16_a, c[1] + off_16_b),
           (c[0] + diag, c[1] + diag),
           (c[0] + off_16_b, c[1] + off_16_a),
           (c[0], c[1] + radius),
           (c[0] - off_16_b, c[1] + off_16_a),
           (c[0] - diag, c[1] + diag),
           (c[0] - off_16_a, c[1] + off_16_b),
           (c[0] - radius, c[1]),
           (c[0] - off_16_a, c[1] - off_16_b),
           (c[0] - diag, c[1] - diag),
           (c[0] - off_16_b, c[1] - off_16_a))
    count = 0
    for coordinates in arr:
        if out_of_bonds(coordinates, height, width):
            return False
        if binary_arr[coordinates] == 0:
            count += 1

    if 0 < count <= 7:
        return True
    else:
        return False


# draw conditions (visualisation)
def draw_cond_1(c: tuple, img_gray: array):
    img = draw_4(c, img_gray, V_ALPHA)

    return img


def draw_cond_2(c: tuple, img_gray: array):
    a_b = V_ALPHA + V_BETA
    img = draw_8(c, img_gray, a_b)
    return img


def draw_cond_3(c: tuple, img_gray: array):
    a_b_c = V_ALPHA + V_BETA + V_GAMMA
    img = draw_16(c, img_gray, a_b_c)
    return img


def draw_4(c: tuple, img_gray: array, radius: int):
    cv2.circle(img_gray, c, radius, V_CIRCLE_COLOR)
    cv2.circle(img_gray, (c[0], c[1] - radius), 3, V_CHECKPOINT_COLOR)  # top
    cv2.circle(img_gray, (c[0] + radius, c[1]), 3, V_CHECKPOINT_COLOR)  # right
    cv2.circle(img_gray, (c[0], c[1] + radius), 3, V_CHECKPOINT_COLOR)  # bottom
    cv2.circle(img_gray, (c[0] - radius, c[1]), 3, V_CHECKPOINT_COLOR)  # left
    return img_gray


def draw_8(c: tuple, img_gray: array, radius: int):
    img = draw_4(c, img_gray, radius)
    a_bK = int(radius * 0.7)

    cv2.circle(img, (c[0] + a_bK, c[1] + a_bK), 3, V_CHECKPOINT_COLOR)  # bottom right
    cv2.circle(img, (c[0] - a_bK, c[1] - a_bK), 3, V_CHECKPOINT_COLOR)  # top left
    cv2.circle(img, (c[0] + a_bK, c[1] - a_bK), 3, V_CHECKPOINT_COLOR)  # top right
    cv2.circle(img, (c[0] - a_bK, c[1] + a_bK), 3, V_CHECKPOINT_COLOR)  # bottom left

    return img


def draw_16(c: tuple, img_gray: array, radius: int):
    img = draw_8(c, img_gray, radius)
    off_16_a = int(radius * 0.9)
    off_16_b = int(radius * 0.4)

    cv2.circle(img_gray, (c[0] + off_16_b, c[1] - off_16_a), 3, V_CHECKPOINT_COLOR)  # x1
    cv2.circle(img_gray, (c[0] + off_16_b, c[1] + off_16_a), 3, V_CHECKPOINT_COLOR)  # x4
    cv2.circle(img_gray, (c[0] - off_16_b, c[1] - off_16_a), 3, V_CHECKPOINT_COLOR)  # x8
    cv2.circle(img_gray, (c[0] - off_16_b, c[1] + off_16_a), 3, V_CHECKPOINT_COLOR)  # x5

    cv2.circle(img_gray, (c[0] + off_16_a, c[1] - off_16_b), 3, V_CHECKPOINT_COLOR)  # x2
    cv2.circle(img_gray, (c[0] + off_16_a, c[1] + off_16_b), 3, V_CHECKPOINT_COLOR)  # x3
    cv2.circle(img_gray, (c[0] - off_16_a, c[1] - off_16_b), 3, V_CHECKPOINT_COLOR)  # x7
    cv2.circle(img_gray, (c[0] - off_16_a, c[1] + off_16_b), 3, V_CHECKPOINT_COLOR)  # x6

    return img
