import copy
import glob
import os
import cv2
import numpy as np

from app.meth import add_frame, otsu, move_matrix_right, move_matrix_left, move_matrix_up

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
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)

    # Gray
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(img)  # array
    # cv2.imshow('gray', img)
    # cv2.waitKey(0)

    # Otsu
    ret, img_otsu = otsu(img)
    cv2.imshow('Otsu Thresholding', img_otsu)
    cv2.waitKey(0)

    # add frame
    border = add_frame(30, img_otsu, [0, 0, 0])
    # cv2.imshow('border', border)
    # cv2.waitKey(0)

    # roll to right
    roll_r = move_matrix_right(copy.deepcopy(border))
    r = copy.deepcopy(border)
    r[roll_r > 0] = 0
    cv2.imshow('right', roll_r)
    cv2.waitKey(0)

    # roll to left
    roll_l = move_matrix_left(copy.deepcopy(border))
    left = copy.deepcopy(border)
    left[roll_l > 0] = 0
    cv2.imshow('left', roll_l)
    cv2.waitKey(0)

    # roll to up
    roll_u = move_matrix_up(copy.deepcopy(border))
    up = copy.deepcopy(border)
    up[roll_u > 0] = 0
    cv2.imshow('up', roll_u)
    cv2.waitKey(0)


    # should be the same
    # cv2.imshow('Otsu Thresholding', border)
    # cv2.waitKey(0)

# destroy
# cv2.waitKey(0)
# cv2.destroyAllWindows()
