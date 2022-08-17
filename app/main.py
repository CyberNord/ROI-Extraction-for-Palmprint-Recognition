import csv
import glob
import math
import os
import cv2
import imutils as imutils
import numpy as np
from datetime import datetime

from app.constants import ERODE_STEP, OUTPUT_FOLDER, ALTERNATE_HAND_DETECTION, SKIN_SAMPLE
from app.meth import otsu, move_matrix_right, move_matrix_left, move_matrix_up, logical_conjunction, \
    get_valley_points, get_cond1, get_cond2, get_cond3, draw_circle, draw_points, rotate, draw_roi, \
    cut_roi, is_right_hand, mask, cb_cr

# path = os.path.join("db\\casia\\test_17_alpha", "*.*")
# path = os.path.join("db\\11kHands\\samples_25", "*.*")
path = os.path.join("db\\own", "*.*")

# 1: 11k hands Mask, 2: 11k hands cbcr, 3: casia & own
mode = 3
success_counter = 0
failure_counter = 0
log = str(path) + '\n-----------------------------\n'
path_list = glob.glob(path)
folder_out = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
hand_no = 1

if mode == 2:
    skin_path = os.path.join(SKIN_SAMPLE)

    skin_s = cv2.imread(skin_path)
    skin_s = cv2.cvtColor(skin_s, cv2.COLOR_BGR2YCR_CB)
    cov = np.delete(skin_s, 0, 2)
    cov = np.transpose(cov, (0, 2, 1))
    cov = np.reshape(cov, (2, -1))
    cov = np.cov(cov)

for file in path_list:

    out = OUTPUT_FOLDER + folder_out + '\\' + str(hand_no)
    if not os.path.exists(out):
        os.makedirs(out)

    # Read in pic & rotate
    # image = cv2.rotate(cv2.imread(file), cv2.cv2.ROTATE_180)
    # image = cv2.rotate(cv2.imread(file), cv2.cv2.ROTATE_90_CLOCKWISE)
    image = cv2.imread(file)

    if mode == 1:
        # Gray
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        cv2.imwrite(out + '\\01_YCrCb.png', ycrcb)

        # Masking
        img_bin = mask(ycrcb)
        # cv2.imwrite(out + '\\02_img_otsu.png', img_bin)

    elif mode == 2:

        # Gray
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        cv2.imwrite(out + '\\01_YCrCb.png', ycrcb)

        img_bin = cb_cr(ycrcb, cov)
        cv2.imwrite(out + '\\02_BW.png', img_bin)

    else:
        # Gray
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out + '\\01_gray.png', img_gray)

        # Otsu - CASIA
        _, img_bin = otsu(img_gray)
        cv2.imwrite(out + '\\02_img_otsu.png', img_bin)

    # get filename without extension
    file_name = os.path.basename(file)
    file_name = file_name[:-4]

    height, width, _ = image.shape
    center = (int(width / 2), int(height / 2))
    print(f'height={height},width={width}, center={center}')
    cv2.imwrite(out + '\\00_' + file_name + '.jpg', image)

    # Valley Point Detection Based on TPDTR
    # roll to right
    bin_hand_r = move_matrix_right(np.copy(img_bin))
    cv2.imwrite(out + '\\03_bin_hand_r.png', bin_hand_r)

    # roll to left
    bin_hand_l = move_matrix_left(np.copy(img_bin))
    cv2.imwrite(out + '\\04_bin_hand_l.png', bin_hand_l)

    # roll to up
    bin_hand_u = move_matrix_up(np.copy(img_bin))
    cv2.imwrite(out + '\\05_bin_hand_u.png', bin_hand_u)

    # logical conjunction
    conj = logical_conjunction(bin_hand_l, bin_hand_r, bin_hand_u)
    cv2.imwrite(out + '\\06_Conjuction.png', conj)

    # Translate down & get valley Points
    valleys = get_valley_points(np.copy(conj), np.copy(img_bin))
    cv2.imwrite(out + '\\07_valleys.png', valleys)

    valley_blobs = np.copy(valleys)
    check = np.copy(image)
    for i in range(len(valleys)):
        for j in range(len(valleys[i])):
            coord = (i, j)
            if valleys[coord] == 255:
                check[coord] = (255, 0, 0)
                if get_cond1(img_bin, coord, height, width):
                    if get_cond2(img_bin, coord, height, width):
                        if get_cond3(img_bin, coord, height, width):
                            check[coord] = (0, 255, 0)
                        else:
                            valley_blobs[coord] = 0
                    else:
                        valley_blobs[coord] = 0
                else:
                    valley_blobs[coord] = 0

    # visualizes the pixels that were removed & the one that stay
    cv2.imwrite(out + '\\08_Cut-out-Visualised.png', check)

    # valley blobs
    cv2.imwrite(out + '\\09_Valley-blobs.png', valley_blobs)

    moments = []
    centroids = []

    itr = 0
    search_c = True
    while search_c and itr < 7:
        itr += 1

        contours0, _ = cv2.findContours(valley_blobs.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in contours0:
            moments.append(cv2.moments(cnt))

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
            if itr == 1 and ERODE_STEP:
                valley_blobs = cv2.erode(valley_blobs, None, iterations=1)
            moments = []
            centroids = []
            valley_blobs = cv2.dilate(valley_blobs, None, iterations=1)
            cv2.imwrite(out + '\\10_Valley-blobs-dilated.png', valley_blobs)

    print(f"centroids={centroids} - iterations:{itr}")

    if len(centroids) == 4:

        valley_centroids = np.copy(image)
        valley_points = np.copy(image)
        for c in centroids:
            valley_centroids = draw_circle(c, valley_centroids)

        cv2.imwrite(out + '\\11_valley_centroids.png', valley_centroids)

        # sort list by y-value
        sorted_list = sorted(centroids, key=lambda y: y[0])
        print(f'sorted list: {sorted_list}')

        # distinguish between left and right hand
        if ALTERNATE_HAND_DETECTION:
            right_hand = is_right_hand(sorted_list)
        else:
            # method proposed by paper
            right_hand = False
            if sorted_list[3][1] > sorted_list[0][1] and sorted_list[3][1] > sorted_list[1][1] and sorted_list[3][1] > sorted_list[2][1]:
                right_hand = False
                sorted_list.reverse()
            elif sorted_list[0][1] > sorted_list[1][1] and sorted_list[0][1] > sorted_list[2][1] and sorted_list[0][1] > sorted_list[3][1]:
                right_hand = False

        if not right_hand:
            sorted_list.reverse()

        valley_points = draw_points(sorted_list, valley_points)
        cv2.imwrite(out + '\\12_valley-points.png', valley_points)

        # x4-x2,y4-y2
        angle = math.degrees(math.atan2(sorted_list[3][1] - sorted_list[1][1], sorted_list[3][0] - sorted_list[1][0]))

        if not right_hand:
            angle += 180

        output_image = imutils.rotate(valley_points, angle=angle)       # for visualisation
        image = imutils.rotate(image, angle=angle)                      # for cutting ROI

        cv2.circle(output_image, center, 0, (255, 0, 255), 5)           # center point
        a = angle * np.pi / 180

        rotated_coordinates = rotate(center, sorted_list, a)
        output_image = draw_points(rotated_coordinates, output_image, (255, 250, 0), False)

        cv2.imwrite(out + '\\13_Rotate-image.png', output_image)

        roi_vis = draw_roi(np.copy(output_image), rotated_coordinates[1], rotated_coordinates[3], True)
        cv2.imwrite(out + '\\14_ROI_Visualized.png', roi_vis)

        roi = cut_roi(np.copy(image), rotated_coordinates[1], rotated_coordinates[3], right_hand)
        if roi is not None:
            cv2.imwrite(out + '\\15_ROI__' + str(hand_no) + '__.png', roi)
            success_counter += 1
        else:
            failure_counter += 1
            log += f'Failure drawing ROI (out of bounce) at hand {hand_no}  {file_name}\n'
    else:
        failure_counter += 1
        log += f' Not possible to find Centroids at {hand_no}  {file_name}\n'

    print(f'-------------next hand{hand_no}-------------')
    hand_no += 1

print('fin')
total = success_counter + failure_counter
print(f'Total analysed pictures: {total}\nsuccess={success_counter} failures={failure_counter}\n')
log += f'\n\nTotal analysed pictures: {total}\nsuccess={success_counter} failures={failure_counter}\n'

fp = open('D:\\Datengrab\\BA_workspace\\out\\' + folder_out + '\\log.txt', 'w')
fp.write(log)
fp.close()
