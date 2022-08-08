import csv
import glob
import math
import os
import cv2
import imutils as imutils
import numpy as np
from datetime import datetime

from app.meth import otsu, move_matrix_right, move_matrix_left, move_matrix_up, logical_conjunction, \
    get_valley_points, get_cond1, get_cond2, get_cond3, draw_circle, draw_points, rotate, draw_roi, cut_roi


# path = os.path.join("db\\casia\\small", "*.*")
# path = os.path.join("db\\casia\\sample_10x2_01", "*.*")
path = os.path.join("db\\casia\\samples_312x2_02", "*.*")
# path = os.path.join("db\\casia\\samples_3x2", "*.*")
# path = os.path.join("db\\11kHands\\small", "*.*")

success_counter = 0
failure_counter = 0
log = str(path) + '\n-----------------------------\n'

path_list = glob.glob(path)

folder_out = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
hand_no = 1

for file in path_list:

    out = 'D:\\Datengrab\\BA_workspace\\out\\' + folder_out + '\\' + str(hand_no)
    if not os.path.exists(out):
        os.makedirs(out)

    # Read in pic & rotate
    image = cv2.imread(file)
    image = cv2.rotate(cv2.imread(file), cv2.cv2.ROTATE_90_CLOCKWISE)

    file_name = os.path.basename(file)
    file_name = file_name[:-4]

    cv2.imwrite(out + '\\00_' + file_name + '.jpg', image)


    # Gray
    height, width, _ = image.shape
    center = (int(width / 2), int(height / 2))
    print(f'height={height},width={width}, center={center}')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out + '\\01_gray.png', img_gray)

    # Otsu
    ret, img_otsu = otsu(img_gray)
    cv2.imwrite(out + '\\02_img_otsu.png', img_otsu)

    # Valley Point Detection Based on TPDTR
    # roll to right
    bin_hand_r = move_matrix_right(np.copy(img_otsu))
    cv2.imwrite(out + '\\03_bin_hand_r.png', bin_hand_r)
    # plt.imshow(bin_hand_r, cmap='gray')
    # plt.show()

    # roll to left
    bin_hand_l = move_matrix_left(np.copy(img_otsu))
    cv2.imwrite(out + '\\04_bin_hand_l.png', bin_hand_l)
    # plt.imshow(bin_hand_l, cmap='gray')
    # plt.show()

    # roll to up
    bin_hand_u = move_matrix_up(np.copy(img_otsu))
    cv2.imwrite(out + '\\05_bin_hand_u.png', bin_hand_u)
    # plt.imshow(bin_hand_u, cmap='gray')
    # plt.show()

    # logical conjunction
    conj = logical_conjunction(bin_hand_l, bin_hand_r, bin_hand_u)
    cv2.imwrite(out + '\\06_Conjuction.png', conj)
    # plt.imshow(conj, cmap='gray')
    # plt.show()

    # Translate down & get valley Points
    valleys = get_valley_points(np.copy(conj), np.copy(img_otsu))
    cv2.imwrite(out + '\\07_valleys.png', valleys)
    # plt.imshow(valleys, cmap='gray')
    # plt.show()

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
    cv2.imwrite(out + '\\08_Cut-out-Visualised.png', check)
    # plt.imshow(check)
    # plt.show()

    # valley blobs
    cv2.imwrite(out + '\\09_Valley-blobs.png', valley_blobs)
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
            cv2.imwrite(out + '\\10_Valley-blobs-dilated.png', valley_blobs)
            # plt.imshow(valley_blobs)
            # plt.show()

    print(f"centroids={centroids} - iterations:{itr}")

    if len(centroids) == 4:

        valley_centroids = np.copy(image)
        valley_points = np.copy(image)
        for c in centroids:
            valley_centroids = draw_circle(c, valley_centroids)

        cv2.imwrite(out + '\\11_valley_centroids.png', valley_centroids)
        # plt.imshow(valley_centroids)
        # plt.show()

        # sort list by y-value
        sorted_list = sorted(centroids, key=lambda y: y[0])
        print(f'sorted list: {sorted_list}')

        # distinguish between left and right hand
        right_hand = True
        if sorted_list[0][1] < sorted_list[3][1]:
            right_hand = False
            sorted_list.reverse()
        valley_points = draw_points(sorted_list, valley_points)
        cv2.imwrite(out + '\\12_valley-points.png', valley_points)
        # plt.imshow(valley_points)
        # plt.show()

        # x4-x2,y4-y2
        angle = math.degrees(math.atan2(sorted_list[3][1] - sorted_list[1][1], sorted_list[3][0] - sorted_list[1][0]))

        # slope = int(get_slope(sorted_list[1], sorted_list[3]))
        # distance = int(math.dist(sorted_list[1], sorted_list[3]))

        if not right_hand:
            angle += 180

        output_image = imutils.rotate(valley_points, angle=angle)       # for visualisation
        image = imutils.rotate(image, angle=angle)                      # for cutting ROI

        cv2.circle(output_image, center, 0, (255, 0, 255), 5)  # center point
        a = angle * np.pi / 180

        rotated_coordinates = rotate(center, sorted_list, a)
        output_image = draw_points(rotated_coordinates, output_image, (255, 250, 0), False)

        cv2.imwrite(out + '\\13_Rotate-image.png', output_image)
        # plt.imshow(output_image)
        # plt.show()

        roi_vis = draw_roi(np.copy(output_image), rotated_coordinates[1], rotated_coordinates[3], True)
        cv2.imwrite(out + '\\14_ROI_Visualized.png', roi_vis)
        # plt.imshow(roi_vis)
        # plt.show()

        roi = cut_roi(np.copy(image), rotated_coordinates[1], rotated_coordinates[3], right_hand)
        if roi is not None:
            cv2.imwrite(out + '\\15_ROI__' + str(hand_no) + '__.png', roi)
        else:
            failure_counter += 1
            log += f'Failure drawing ROI at hand {hand_no}  {file_name}\n'

        success_counter += 1
        # plt.imshow(roi)
        # plt.show()
    else:
        failure_counter += 1
        log += f'Failure at hand {hand_no}  {file_name}\n'

    print(f'-------------next hand{hand_no}-------------')
    hand_no += 1

print('fin')
total = success_counter + failure_counter
print(f'Total analysed pictures: {total}\nsuccess={success_counter} failures={failure_counter}\n')
log += '\n\nTotal analysed pictures: {total}\nsuccess={succes_counter} failures={failure_counter}\n'

fp = open('log.txt', 'w')
fp.write(log)
fp.close()
