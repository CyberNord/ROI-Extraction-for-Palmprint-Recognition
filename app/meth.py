import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import array

from app.settings import OTSU_LOWER, OTSU_HIGHER, PIXEL_OFFSET, PIXEL_OFFSET_NEG, VALLEY_GAP_OFFSET, \
    V_ALPHA, V_BETA, V_GAMMA, YCrCb_SKIN_LOWER, YCrCb_SKIN_HIGHER, DEBUG_PICTURES

# Color values (BGR)

BLACK = (0, 0, 0)
DARK = (1, 1, 1)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (255, 0, 255)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)

# Axis
A_HORIZONTAL = 1
A_VERTICAL = 0

# Matrix values
M_VISIBLE = 255
M_CALCULATION = 1


# ### Preprocessing ###

# OTSU (Mode 3)
def otsu(img: array):
    return cv2.threshold(img, OTSU_LOWER, OTSU_HIGHER, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Masking (Mode 2)
def mask(img_ycrcb: array):
    lower = np.array(YCrCb_SKIN_LOWER, np.uint8)
    upper = np.array(YCrCb_SKIN_HIGHER, np.uint8)

    mask = cv2.inRange(img_ycrcb, lower, upper)
    _, black_and_white = cv2.threshold(mask, 127, 255, 0)

    black_and_white = cv2.dilate(black_and_white, None, iterations=3)
    black_and_white = cv2.erode(black_and_white, None, iterations=3)

    return black_and_white


# cb_cr  (Mode 1)
def cb_cr(img_ycrcb: array, sigma: array):

    # # exp[ -0.5 (cb_cr - mu) sigma^-1 (cb_cr - mu)^T]
    mu = np.mean(img_ycrcb, (0, 1))
    sig_1 = np.linalg.inv(sigma)

    print('Processing picture:', end=' ')

    for idx_y, line in enumerate(img_ycrcb):
        for idx_x, pixel in enumerate(line):
            y, cr, cb = pixel

            cb_cr_mu = np.subtract((cr, cb), (mu[1], mu[2]))
            cb_cr_mu_t = cb_cr_mu.transpose()

            mul = np.matmul(np.matmul(cb_cr_mu, sig_1), cb_cr_mu_t)
            exp = np.exp(-0.5 * mul)

            if exp > 0.5:
                img_ycrcb[idx_y, idx_x, :] = 255
            else:
                img_ycrcb[idx_y, idx_x, :] = 0

            if idx_y % 200 == 0 and idx_x % 500 == 0:
                print('.', end='')

    print(' ,done')
    plt.imshow(img_ycrcb, cmap='gray')
    plt.show()

    return img_ycrcb


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
# move it down & conjunct it with the binary hand
# it will return the hand valley points
def get_valley_points(conj: array, binary_arr: array):
    roll = np.roll(conj, VALLEY_GAP_OFFSET, axis=A_VERTICAL)
    ret = translate_to(roll, 1) * translate_to(binary_arr, M_CALCULATION)
    return translate_to(ret, M_VISIBLE)


def out_of_bonds(coordinates, height, width):
    if 0 <= coordinates[0] <= height - 1 and 0 < coordinates[1] < width - 1:
        return False
    return True


# Condition1: four-point check
# radius = Alpha
# top is black & right, bottom, left are white
def get_cond1(binary_arr: array, c: tuple, height: int, width: int):
    # 4-positions
    # top, right, bottom, left
    arr = ((c[0], c[1] - V_ALPHA),
           (c[0] + V_ALPHA, c[1]),
           (c[0], c[1] + V_ALPHA),
           (c[0] - V_ALPHA, c[1]))  # top
    for coordinates in arr:
        if out_of_bonds(coordinates, height, width):
            return False
    if binary_arr[arr[3]] == 0 \
            and binary_arr[arr[0]] == 255 and binary_arr[arr[1]] == 255 and binary_arr[arr[2]] == 255:
        return True
    return False


# Condition2: eight-point check
# radius = Alpha + Beta
# at least 1 and not more than 4 white pixels
def get_cond2(binary_arr: array, c: tuple, height: int, width: int):
    radius = (V_ALPHA + V_BETA)
    diag = int(radius * 0.7)
    # 8-positions
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
    if 1 <= count <= 4:
        return True
    else:
        return False


# Condition3: sixteen-point check
# radius = Alpha + Beta + Gamma
# at least 1 and not more than 7 white pixels
def get_cond3(binary_arr: array, c: tuple, height: int, width: int):
    radius = (V_ALPHA + V_BETA + V_GAMMA)
    diag = int(radius * 0.7)
    off_16_a = int(radius * 0.9)
    off_16_b = int(radius * 0.4)

    # 16-positions (x-pos)
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

    if 1 <= count <= 7:
        return True
    else:
        return False


# draw hand valleys (visualisation)
def draw_circle(c: tuple, img_gray: array, color=RED):
    cv2.circle(img_gray, c, 0, color, 3)
    cv2.circle(img_gray, c, 10, color)
    return img_gray


def draw_points(centroids: array, img: array, color=RED, debug=True):
    p = 1
    for c in centroids:
        (x, y) = c[0], c[1] - V_ALPHA
        cv2.circle(img, c, 0, color, 6)
        if debug:
            img = cv2.putText(img, f'P{p}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if (p == 2 or p == 4) and DEBUG_PICTURES:
                print(f'coordinates of p{p}:({c}) at array position centroids[{p - 1}]')
        p += 1
    return img


def rotate(center, points, angle):
    result = []

    for point in points:
        new_pt = (point[0] - center[0], point[1] - center[1])

        # ( x' * cos(alpha) ) - ( y' * sin(alpha)) + off = x
        x = (new_pt[1] * np.cos(angle)) - (new_pt[0] * np.sin(angle)) + center[1]

        # ( x' * sin(alpha) ) + ( y' * cos(alpha)) + off = y
        y = (new_pt[1] * np.sin(angle)) + (new_pt[0] * np.cos(angle)) + center[0]

        r = (round(y), round(x))
        result.append(r)

    return result


def calculate_roi_params(p2: tuple, p4: tuple):
    distance = int(math.dist(p2, p4))
    off = round(distance * 0.2)
    q1 = (p2[0], p2[1] + off)
    q3 = (p4[0], p4[1] + distance + off)

    return distance, off, q1, q3


def draw_roi(img: array, p2: tuple, p4: tuple, debug=False, color=RED, thickness=2):
    distance, off, q1, q3 = calculate_roi_params(p2, p4)
    if debug:
        q4 = (p4[0], p4[1] + off)
        q2 = (p2[0], p4[1] + distance + off)
        img = cv2.rectangle(img, p2, q4, YELLOW, thickness)
        img = cv2.circle(img, p2, 0, PINK, 5)
        img = cv2.circle(img, p4, 0, PINK, 5)
        img = cv2.circle(img, p4, 0, PINK, 5)
        img = cv2.circle(img, q1, 0, BLUE, 10)
        img = cv2.circle(img, q2, 0, GREEN, 10)
        img = cv2.circle(img, q3, 0, GREEN, 10)
        img = cv2.circle(img, q4, 0, BLUE, 10)

    return cv2.rectangle(img, q1, q3, color, thickness)


def cut_roi(img: array, p2: tuple, p4: tuple, right_hand):
    distance, off, q1, q3 = calculate_roi_params(p2, p4)

    if off == 0 or distance == 0:
        return None

    if not right_hand:
        q4 = (p4[0], p4[1] + off)
        q2 = (p2[0], p4[1] + distance + off)
        if q4[0] < 0 or q4[1] < 0 or q2[0] < 0 or q2[1] < 0:
            return None
        return img[q4[1]:q2[1], q4[0]:q2[0]]

    if q1[0] < 0 or q1[1] < 0 or q3[0] < 0 or q3[1] < 0:
        return None
    return img[q1[1]:q3[1], q1[0]:q3[0]]


def is_right_hand(sorted_list: array):

    # left
    d_x1_x2 = math.dist(sorted_list[0], sorted_list[1])
    d_x1_x3 = math.dist(sorted_list[0], sorted_list[2])
    avg_l = (d_x1_x2 + d_x1_x3) / 2

    # right
    d_x4_x2 = math.dist(sorted_list[3], sorted_list[1])
    d_x4_x3 = math.dist(sorted_list[3], sorted_list[2])
    avg_r = (d_x4_x2 + d_x4_x3) / 2

    if avg_r > avg_l:
        return False
    else:
        return True
