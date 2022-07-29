import copy
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from app.meth import add_frame, otsu, move_matrix_right, move_matrix_left, move_matrix_up, logical_conjunction, \
    get_valley_points

path = os.path.join("db\\casia\\small", "*.*")
# path = os.path.join("db\\11kHands\\small", "*.*")
cv_img = []
path_list = glob.glob(path)
# print(path_list)

for file in path_list:
    # Read in pic & rotate
    image = cv2.imread(file)
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    # plt.imshow(image)
    # plt.show()

    # add frame
    img_frame = add_frame(image, [0, 0, 0])
    height, width, _ = img_frame.shape
    print('Height= ' + str(height) + ' Width= ' + str(width))
    # plt.imshow(img_frame)
    # plt.show()

    # Gray
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)

    # Otsu
    ret, img_otsu = otsu(img_gray)
    # plt.imshow(img_otsu)
    # plt.show()

    # Valley Point Detection Based on TPDTR
    # roll to right
    bin_hand_r = move_matrix_right(np.copy(img_otsu))
    # plt.imshow(bin_hand_r)
    # plt.show()

    # roll to left
    bin_hand_l = move_matrix_left(np.copy(img_otsu))
    # plt.imshow(bin_hand_l)
    # plt.show()

    # roll to up
    bin_hand_u = move_matrix_up(np.copy(img_otsu))
    # plt.imshow(bin_hand_u)
    # plt.show()

    # logical conjunction
    conj = logical_conjunction(bin_hand_l, bin_hand_r, bin_hand_u)
    # plt.imshow(conj)
    # plt.show()

    # Translate down & get valley Points
    valleys = get_valley_points(np.copy(conj), np.copy(img_otsu))
    plt.imshow(valleys)
    plt.show()

    # ------------------------------
    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image

    erode = cv2.erode(valleys, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=2)

    plt.imshow(dilate)
    plt.show()

    moments = []
    centroids = []
    contours0, _ = cv2.findContours(dilate.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for cnt in contours0:
        moments.append(cv2.moments(cnt))

    print("moments= " + str(moments))

    for m in moments:
        centroids.append((int(round(m['m10'] / m['m00'])), int(round(m['m01'] / m['m00']))))

    print("centroids= " + str(centroids))

    for c in centroids:
        cv2.circle(img_frame, c, 10, (255, 0, 0))

    plt.imshow(img_frame)
    plt.show()
    print('-------------next Image-------------')

print('fin')
