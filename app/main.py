import glob
import math
import os
import shutil

import cv2
import imutils as imutils
import numpy as np
from datetime import datetime

from app.constants import ERODE_STEP, OUTPUT_FOLDER, ALTERNATE_HAND_DETECTION, SKIN_SAMPLE, MODE, ROTATE, \
    DEBUG_PICTURES, DATABASE
from app.meth import otsu, move_matrix_right, move_matrix_left, move_matrix_up, logical_conjunction, \
    get_valley_points, get_cond1, get_cond2, get_cond3, draw_circle, draw_points, rotate, draw_roi, \
    cut_roi, is_right_hand, mask, cb_cr

path = os.path.join(DATABASE, "*.*")

mode = MODE
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
    cov_Y = np.delete(skin_s, 0, 2)
    cov_transpose = np.transpose(cov_Y, (0, 2, 1))
    cov_reshape = np.reshape(cov_transpose, (2, -1))
    cov = np.cov(cov_reshape)

for file in path_list:

    out = OUTPUT_FOLDER + folder_out + '\\' + str(hand_no)
    if not os.path.exists(out):
        os.makedirs(out)

    # Read in pic & rotate
    image = cv2.imread(file)

    if ROTATE == 90:
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif ROTATE == 180:
        image = cv2.rotate(image, cv2.cv2.ROTATE_180)
    elif ROTATE == 270:
        image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    # else no rotation

    if mode == 1:
        # 01 Gray - Colorspace
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        # 02 Masking
        img_bin = mask(ycrcb)

        if DEBUG_PICTURES:
            cv2.imwrite(out + '\\01_YCrCb.png', ycrcb)
            cv2.imwrite(out + '\\02_img_otsu.png', img_bin)

    elif mode == 2:
        # 01 YCrCb - Colorspace
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

        # 02 Binary YCrCb
        img_skin = cb_cr(ycrcb, cov)

        # 02 Translate to real BW - Colorspace
        img_bin = cv2.cvtColor(img_skin, cv2.COLOR_YCrCb2BGR)
        img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
        _, img_bin = otsu(img_bin)

        if DEBUG_PICTURES:
            cv2.imwrite(out + '\\01_YCrCb.png', ycrcb)
            cv2.imwrite(out + '\\02_1_BW_YCrCb.png', img_skin)
            cv2.imwrite(out + '\\02_2_BW.png', img_bin)

    else:
        # 01 Gray - Colorspace
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 02 Otsu - Algorithm
        _, img_bin = otsu(img_gray)

        if DEBUG_PICTURES:
            cv2.imwrite(out + '\\01_gray.png', img_gray)
            cv2.imwrite(out + '\\02_img_otsu.png', img_bin)

    # get filename without extension
    file_name = os.path.basename(file)
    file_name = file_name[:-4]

    height, width, _ = image.shape
    center = (int(width / 2), int(height / 2))

    if DEBUG_PICTURES:
        print(f'height={height},width={width}, center={center}')
        cv2.imwrite(out + '\\00_' + file_name + '.jpg', image)

    # Valley Point Detection Based on TPDTR
    # 03 roll to right
    bin_hand_r = move_matrix_right(np.copy(img_bin))

    # 04 roll to left
    bin_hand_l = move_matrix_left(np.copy(img_bin))

    # 05 roll to up
    bin_hand_u = move_matrix_up(np.copy(img_bin))

    # 06 logical conjunction
    conj = logical_conjunction(np.copy(bin_hand_l), np.copy(bin_hand_r), np.copy(bin_hand_u))

    # 07 Translate down & get valley Points
    valleys = get_valley_points(np.copy(conj), np.copy(img_bin))

    if DEBUG_PICTURES:
        cv2.imwrite(out + '\\03_bin_hand_r.png', bin_hand_r)
        cv2.imwrite(out + '\\04_bin_hand_l.png', bin_hand_l)
        cv2.imwrite(out + '\\05_bin_hand_u.png', bin_hand_u)
        cv2.imwrite(out + '\\06_Conjunction.png', conj)
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

    if DEBUG_PICTURES:
        # 08 visualizes the pixels that were removed & the one that stay
        cv2.imwrite(out + '\\08_Cut-out-Visualised.png', check)

        # 09 valley blobs
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

            if DEBUG_PICTURES:
                cv2.imwrite(out + '\\10_Valley-blobs-dilated.png', valley_blobs)

    print(f"centroids={centroids} - iterations:{itr}")

    if len(centroids) == 4:

        valley_centroids = np.copy(image)
        valley_points = np.copy(image)
        for c in centroids:
            valley_centroids = draw_circle(c, valley_centroids)

        if DEBUG_PICTURES:
            cv2.imwrite(out + '\\11_valley_centroids.png', valley_centroids)

        # sort list by y-value
        sorted_list = sorted(centroids, key=lambda y: y[0])
        print(f'Valley Coordinates: {sorted_list}')

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

        # 12 show valley points
        valley_points = draw_points(sorted_list, valley_points)

        if DEBUG_PICTURES:
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

        # 13 draw points (again) in rotated image
        output_image = draw_points(rotated_coordinates, output_image, (255, 250, 0), False)

        # 14 Visualised ROI
        roi_vis = draw_roi(np.copy(output_image), rotated_coordinates[1], rotated_coordinates[3], True)

        if DEBUG_PICTURES:
            cv2.imwrite(out + '\\13_Rotate-image.png', output_image)
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

# Write Log
log += f'\n\nTotal analysed pictures: {total}\nsuccess={success_counter} failures={failure_counter}\n'
fp = open(OUTPUT_FOLDER + folder_out + '\\log.txt', 'w')
fp.write(log)
fp.close()

# save settings too
shutil.copyfile('constants.py', OUTPUT_FOLDER + folder_out + '\\settings.txt')
