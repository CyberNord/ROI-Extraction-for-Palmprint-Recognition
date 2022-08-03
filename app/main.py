import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from app.meth import add_frame, otsu, move_matrix_right, move_matrix_left, move_matrix_up, logical_conjunction, \
    get_valley_points, get_cond1, get_cond2, get_cond3

path = os.path.join("db\\casia\\small", "*.*")
# path = os.path.join("db\\11kHands\\small", "*.*")
cv_img = []
path_list = glob.glob(path)
# print(path_list)


for file in path_list:
    # Read in pic & rotate
    image = cv2.imread(file)
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    # image = np.flip(image, axis=1)
    # plt.imshow(image)
    # plt.show()

    # add frame
    # img_frame = add_frame(image, [0, 0, 0])
    # height, width, _ = img_frame.shape
    # print('Height= ' + str(height) + ' Width= ' + str(width))
    # plt.imshow(img_frame)
    # plt.show()

    # Gray
    height, width, _ = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    # plt.imshow(valleys)
    # plt.show()

    # ------------------------------
    # # perform a series of erosion and dilation to remove
    # # any small blobs of noise from the threshold image
    # valleys = cv2.erode(valleys, None, iterations=1)
    # valleys = cv2.dilate(valleys, None, iterations=1)
    # plt.imshow(dilate)
    # plt.show()

    # ------------------------------
    valley_blobs = np.copy(valleys)
    check = np.copy(image)
    plt.imshow(valleys, cmap='gray')
    plt.show()
    counter = 0
    for i in range(len(valleys)):
        for j in range(len(valleys[i])):
            coord = (i, j)
            if valleys[coord] == 255:
                check[coord] = (255, 0, 0)
                if get_cond1(img_otsu, coord, height, width):
                    check[coord] = (0, 255, 0)
                    # if get_cond2(img_otsu, coord, height, width):
                    #     if get_cond3(img_otsu, coord, height, width):
                    #         if 1 == 2:
                    #             print("yes")
                    #     else:
                    #         # counter += 1
                    #         # print(f'pre:{valley_blobs[coord]}')
                    #         valley_blobs[coord] = 0
                    #         # print(f'past:{valley_blobs[coord]}')
                    # else:
                    #     # counter += 1
                    #     # print(f'pre:{valley_blobs[coord]}')
                    #     valley_blobs[coord] = 0
                    #     # print(f'past:{valley_blobs[coord]}')
                else:
                    counter += 1
                    print(f'pre:{valley_blobs[coord]}')
                    valley_blobs[coord] = 0
                    print(f'past:{valley_blobs[coord]}')

    print(counter)

    plt.imshow(check)
    plt.show()
    # plt.imshow(valley_blobs)
    # plt.show()
    #
    # valley_blobs = cv2.dilate(valley_blobs, None, iterations=1)
    # plt.imshow(valley_blobs, cmap='gray')
    # plt.show()

    # moments = []
    # centroids = []
    # contours0, _ = cv2.findContours(valleys.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #
    # for cnt in contours0:
    #     moments.append(cv2.moments(cnt))
    #
    # print("moments= " + str(moments))
    #
    # for m in moments:
    #     centroids.append((int(round(m['m10'] / m['m00'])), int(round(m['m01'] / m['m00']))))
    #
    # print("centroids= " + str(centroids))
    #
    # img_4_px = np.copy(image)
    # img_8_px = np.copy(image)
    # img_16_px = np.copy(image)
    # cond1 = 0
    # cond2 = 0
    # cond3 = 0
    # for c in centroids:
    #     print(f'c={c}')
    #     img_4_px = draw_cond_1(c, img_4_px)
    #     print(f'before:{img_otsu[(78, 323)]}')
    #     cond1 = get_cond1(c, np.copy(img_otsu))
    #
    #     # img_8_px = draw_cond_2(c, img_8_px)
    #     # img_16_px = draw_cond_3(c, img_16_px)
    #
    # print(cond1)
    # plt.imshow(img_4_px)
    # plt.show()
    # plt.imshow(img_otsu)
    # plt.show()

    # plt.imshow(img_8_px)
    # plt.show()
    # plt.imshow(img_16_px)
    # plt.show()
    print('-------------next Image-------------')

print('fin')
