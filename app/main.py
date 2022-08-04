import glob
import math
import os
import cv2
import imutils as imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw

from app.meth import otsu, move_matrix_right, move_matrix_left, move_matrix_up, logical_conjunction, \
    get_valley_points, get_cond1, get_cond2, get_cond3, draw_circle, draw_points, rotate, get_slope

path = os.path.join("db\\casia\\small2", "*.*")
# path = os.path.join("db\\casia\\samples_1", "*.*")
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
    center = (int((width-1)/2), int((height-1)/2))
    print(f'height={height},width={width}, center={center}')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()

    # Otsu
    ret, img_otsu = otsu(img_gray)
    # plt.imshow(img_otsu, cmap='gray')
    # plt.show()

    # Valley Point Detection Based on TPDTR
    # roll to right
    bin_hand_r = move_matrix_right(np.copy(img_otsu))
    # plt.imshow(bin_hand_r, cmap='gray')
    # plt.show()

    # roll to left
    bin_hand_l = move_matrix_left(np.copy(img_otsu))
    # plt.imshow(bin_hand_l, cmap='gray')
    # plt.show()

    # roll to up
    bin_hand_u = move_matrix_up(np.copy(img_otsu))
    # plt.imshow(bin_hand_u, cmap='gray')
    # plt.show()

    # logical conjunction
    conj = logical_conjunction(bin_hand_l, bin_hand_r, bin_hand_u)
    # plt.imshow(conj, cmap='gray')
    # plt.show()

    # Translate down & get valley Points
    valleys = get_valley_points(np.copy(conj), np.copy(img_otsu))
    # plt.imshow(valleys, cmap='gray')
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
    # plt.imshow(valleys, cmap='gray')
    # plt.show()
    counter = 0
    for i in range(len(valleys)):
        for j in range(len(valleys[i])):
            coord = (i, j)
            if valleys[coord] == 255:
                check[coord] = (255, 0, 0)
                if get_cond1(img_otsu, coord, height, width):
                    if get_cond2(img_otsu, coord, height, width):
                        if get_cond3(img_otsu, coord, height, width):
                            check[coord] = (0, 255, 0)
                        else:
                            # counter += 1
                            # print(f'pre:{valley_blobs[coord]}')
                            valley_blobs[coord] = 0
                            # print(f'past:{valley_blobs[coord]}')
                    else:
                        # counter += 1
                        # print(f'pre:{valley_blobs[coord]}')
                        valley_blobs[coord] = 0
                        # print(f'past:{valley_blobs[coord]}')
                else:
                    # counter += 1
                    # print(f'pre:{valley_blobs[coord]}')
                    valley_blobs[coord] = 0
                    # print(f'past:{valley_blobs[coord]}')

    # print(counter)

    # visualizes the pixels that were removed & the one that stay
    # plt.imshow(check)
    # plt.show()

    # valley blobs
    # plt.imshow(valley_blobs, cmap='gray')
    # plt.show()

    moments = []
    centroids = []

    itr = 0
    search_c = True
    while search_c and itr < 5:
        itr += 1

        contours0, _ = cv2.findContours(valley_blobs.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in contours0:
            moments.append(cv2.moments(cnt))
            # print("moments= " + str(moments))

        for m in moments:
            divisor: float
            if m['m00'] == 0:
                divisor = 0.1
            else:
                divisor = m['m00']
            centroids.append((int(round(m['m10'] / divisor)), int(round(m['m01'] / divisor))))
        if len(centroids) == 4:
            search_c = False
        else:
            # if itr == 1:
            #     valley_blobs = cv2.erode(valley_blobs, None, iterations=1)
            moments = []
            centroids = []
            valley_blobs = cv2.dilate(valley_blobs, None, iterations=1)
            # plt.imshow(valley_blobs)
            # plt.show()

    print(f"centroids={centroids} - iterations:{itr}")

    valley_centroids = np.copy(image)
    valley_points = np.copy(image)
    for c in centroids:
        valley_centroids = draw_circle(c, valley_centroids)

    # plt.imshow(valley_centroids)
    # plt.show()

    # sort list by y-value
    sorted_list = sorted(centroids, key=lambda y: y[0])
    print(f'sorted list: {sorted_list}')

    # distinguish between left and right hand
    right_Hand = True
    if sorted_list[0][1] < sorted_list[3][1]:
        right_Hand = False
        sorted_list.reverse()
    valley_points = draw_points(sorted_list, valley_points)

    # plt.imshow(valley_points)
    # plt.show()

    slope = int(get_slope(sorted_list[1], sorted_list[3]))
    angle = math.degrees(math.atan2(sorted_list[3][1] - sorted_list[1][1], sorted_list[3][0] - sorted_list[1][0]))
    m = int(math.dist(sorted_list[1], sorted_list[3]))

    # plt.imshow(valley_points)
    # plt.show()

    if right_Hand:
        output_image = imutils.rotate(valley_points, angle=angle)
        rotated_coordinates = rotate(center, sorted_list, math.radians(slope))
        output_image = draw_points(rotated_coordinates, output_image, (255, 200, 0), False)
    else:
        output_image = imutils.rotate(valley_points, angle=180+angle)

    cv2.circle(output_image, center, 0, (255, 0, 255), 5)  # center point
    # plt.imshow(output_image)
    # plt.show()

    # valley_points = draw_points(sorted_list, output_image, (0, 255, 255), False)
    # plt.imshow(valley_points)
    # plt.show()



    plt.imshow(output_image)
    plt.show()

    print('-------------next Image-------------')

print('fin')
