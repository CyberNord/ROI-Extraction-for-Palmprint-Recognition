import copy
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from app.meth import add_frame, otsu, move_matrix_right, move_matrix_left, move_matrix_up, logical_conjunction

path = os.path.join("db\\casia\\small", "*.*")
# path = os.path.join("db\\11kHands\\small", "*.*")
print(path)
cv_img = []
path_list = glob.glob(path)
print(path_list)

for file in path_list:
    # Read in pic & rotate
    image = cv2.imread(file)
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    height, width, channels = image.shape
    print(height, width)
    # plt.imshow(image)
    # plt.show()

    # add frame
    img_frame = add_frame(30, image, [0, 0, 0])
    # plt.imshow(img_frame)
    # plt.show()

    # Gray
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(img_gray)  # array
    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)

    # Otsu
    ret, img_otsu = otsu(img_gray)
    plt.imshow(img_otsu)
    plt.show()
    # plt.imshow(img_otsu)
    # plt.show()

    # Valley Point Detection Based on TPDTR
    # roll to right
    bin_hand_r = move_matrix_right(np.copy(img_otsu))
    plt.imshow(bin_hand_r)
    plt.show()

    # roll to left
    bin_hand_l = move_matrix_left(copy.deepcopy(img_otsu))
    plt.imshow(bin_hand_l)
    plt.show()

    # roll to up
    bin_hand_u = move_matrix_up(copy.deepcopy(img_otsu))
    plt.imshow(bin_hand_u)
    plt.show()

    # logical conjunction
    conj = logical_conjunction(bin_hand_l, bin_hand_r, bin_hand_u)
    plt.imshow(conj)
    plt.show()

# destroy
# cv2.waitKey(0)
# cv2.destroyAllWindows()
