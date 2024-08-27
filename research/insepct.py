
# TODO: create a folder called "res" in the root directory and put all the images of O-rings in this folder

import os
import sys
import cv2 as cv
import time
import statistics as stats
import numpy as np
import matplotlib.pyplot as plt

# print the full array without truncation
np.set_printoptions(threshold=sys.maxsize)


# create an image histogram
def imhist(img):
    hist = np.zeros(256)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            hist[img[i, j]] += 1
    return hist


# compute a threshold value using histogram-based algorithm
def calculate_threshold_histogram(hist):
    # starting, ending and mid-point (threshold) index
    start_index = np.min(np.where(hist > 0))
    end_index = np.max(np.where(hist > 0))
    t = int(round((start_index + end_index) / 2))

    bg_weight = np.sum(hist[0:t + 1])  # background weight (left)
    fg_weight = np.sum(hist[t + 1:end_index + 1])  # foreground weight (right)

    while start_index < end_index:
        new_t = ((start_index + end_index) / 2)
        if bg_weight > fg_weight:  # if left side is heavier
            bg_weight -= hist[start_index]
            start_index += 1
            if new_t > t:  # re-center t
                bg_weight += hist[t+1]
                fg_weight -= hist[t+1]
                t += 1
        else:  # if right side is heavier
            fg_weight -= hist[end_index]
            end_index -= 1
            if new_t <= t:  # re-center t
                bg_weight -= hist[t]
                fg_weight += hist[t]
                t -= 1
    print('Threshold Value: %d' % (t - 20))
    return t - 20


# compute a new threshold value using clustering algorithm
def calculate_threshold_clustering(img, initial_t):
    c1, c2 = [], []

    # (1) segment the image using the initial threshold value to produce two clusters, C1 and C2
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] > initial_t:
                c1.append(img[i, j])
            if img[i, j] <= initial_t:
                c2.append(img[i, j])

    # (2) compute the average grey level of each cluster of pixels
    mean_c1 = stats.mean(c1)
    mean_c2 = stats.mean(c2)

    # (3) compute a new threshold value
    new_t = (float(mean_c1) + float(mean_c2)) / 2
    return new_t


# implement image thresholding using for loops
def apply_threshold(img, t):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] > t:
                img[i, j] = 0  # black (background)
            else:
                img[i, j] = 255  # white (foreground)
    return img


# repeatedly call the calculate_threshold(img, t) method until the difference between
# previous and current t is less than a predefined limit (0.1)
def thresholding(img, t):
    count = 1
    iteration = {'T%d' % count: float(format(t, '.1f'))}
    new_t = calculate_threshold_clustering(img, t)
    diff_t = abs(t - new_t)
    t_limit = 0.1

    while diff_t > t_limit:  # recalculate a value for t as the difference is not negligible
        temp_t = new_t
        new_t = calculate_threshold_clustering(img, temp_t)
        diff_t = abs(temp_t - new_t)
        count += 1
        iteration.update({'T%d' % count: new_t})
    else:  # implement image thresholding as the difference is negligible (close to 0)
        apply_threshold(img, new_t)
    print('Threshold Value: %s' % iteration)


# add pixels to the boundaries of the foreground object, i.e. enlarging
def dilate(img):
    # (1) pad image (2D array of pixels) with zeros
    img = np.pad(img, 0, 'constant')

    # (2) create a rectangular shape structuring element and fill it with zeros
    dilated_img = np.zeros((img.shape[0], img.shape[1]))

    # (3) loop through 220x220 pixels (48400)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # retrieve the sum of 8 neighboring pixels and the current pixel (3x3)
            dilated_img[i, j] = np.sum(img[i - 1:i + 2, j - 1:j + 2])

            if dilated_img[i, j] > 0:  # if not all grey levels are black (background)
                dilated_img[i, j] = 255  # white (foreground)
            else:
                dilated_img[i, j] = 0  # black (background)
    return dilated_img


# remove pixels from the boundaries of the foreground object, i.e. shrinking
def erode(img):
    # (1) pad image (2D array of pixels) with zeros
    img = np.pad(img, 0, 'constant')

    # (2) create a rectangular shape structuring element and fill it with zeros
    eroded_img = np.zeros((img.shape[0], img.shape[1]))

    # (3) loop through 220x220 pixels (48400)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            # retrieve the sum of 8 neighboring pixels and the current pixel (3x3)
            eroded_img[i, j] = np.sum(img[i - 1:i + 2, j - 1:j + 2])

            if eroded_img[i, j] < (255 * 9):  # if not all grey levels are white (foreground)
                eroded_img[i, j] = 0  # black (background)
            else:
                eroded_img[i, j] = 255  # white (foreground)
    return eroded_img


# implement Breadth First Search algorithm using a linked queue (FIFO)
# pixels are labelled before being enqueued (queue is used to check the neighbors of a pixel)
# all the neighboring pixels are labelled before moving on to the next pixel
def labelling(img):
    img = np.pad(img, 0, 'constant')
    labelled_img = np.zeros((img.shape[0], img.shape[1]))

    # (1) initialize a queue (list) and set "curr" to 100
    queue = []
    curr = 100

    # (2) labelling: check if the current pixel is a foreground pixel and if it's already labelled
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 255 and labelled_img[i, j] == 0:  # current foreground pixel
                labelled_img[i, j] = curr
                queue.append([i, j])  # enqueue current pixel

                # (3) dequeue a pixel (first in the queue) and look at its 4-connected pixels
                # repeat Step 2 for each neighboring pixel
                while len(queue) != 0:
                    [i, j] = queue.pop(0)  # pop the current pixel

                    if img[i - 1, j] == 255 and labelled_img[i - 1, j] == 0:  # left neighbor pixel
                        labelled_img[i - 1, j] = curr
                        queue.append([i - 1, j])

                    if img[i + 1, j] == 255 and labelled_img[i + 1, j] == 0:  # right neighbor pixel
                        labelled_img[i + 1, j] = curr
                        queue.append([i + 1, j])

                    if img[i, j - 1] == 255 and labelled_img[i, j - 1] == 0:  # top neighbor pixel
                        labelled_img[i, j - 1] = curr
                        queue.append([i, j - 1])

                    if img[i, j + 1] == 255 and labelled_img[i, j + 1] == 0:  # bottom neighbor pixel
                        labelled_img[i, j + 1] = curr
                        queue.append([i, j + 1])

                # (4) iterate Step 2 for the next pixel in the image
                # increment "curr" by 100 to distinguish between different regions (labels)
                curr += 100

            elif img[i, j] == 0:  # background pixel
                pass

    label = np.unique(labelled_img)
    print('Unique Labels: ', label.tolist())  # list all unique labels
    return label, labelled_img


# some of the larger broken pieces are still visible after applying the morphology operators
# remove broken pieces completely using the labels assigned
def remove_broken_piece(label, img, labelled_img):
    img = np.pad(img, 0, 'constant')
    print('O-ring\'s Label: %d' % label[1])

    if len(label) > 2:  # apart from the labels for background and foreground region
        print('Broken Pieces Exist: Yes')
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if labelled_img[i, j] > label[1]:  # label[1] = 100 (O-ring)
                    img[i, j] = 0  # black (background)
    else:
        print('Broken Pieces Exist: No')
    return img


# extract the contours of the labelled O-ring
def extract_contours(label, img, labelled_img):
    img = np.pad(img, 0, 'constant')
    contoured_img = np.zeros((img.shape[0], img.shape[1]))

    for i in range(0, labelled_img.shape[0]):
        for j in range(0, labelled_img.shape[1]):
            # it's an edge pixel if one of its 4-connected neighbors is a background pixel
            if labelled_image[i, j] == label[1]:  # pixels belonging to the O-ring
                if img[(i - 1), j] == 0 or img[(i + 1), j] == 0 or img[i, (j - 1)] == 0 or img[i, (j + 1)] == 0:
                    contoured_img[i, j] = 255  # white (outline)
                else:
                    contoured_img[i, j] = 0  # black (background)
    return contoured_img


# compute the coordinates (x1, y1, x2, y2) used to generate a bounding box around the labelled O-ring
def calculate_coordinate(img):
    # 2D array of length (220, 220)
    # start with high initial values (arbitrary) for min_x, min_y and they will converge slowly
    min_x, min_y, max_x, max_y = 250, 250, 0, 0

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == 255:  # edge pixel
                if min_x > j:   # row
                    min_x = j
                if max_x < j:
                    max_x = j

                if min_y > i:  # column
                    min_y = i
                if max_y < i:
                    max_y = i
    return min_x, min_y, max_x, max_y


# perform an analysis on the extracted region to classify the O-ring (pass/fail)
# algorithm: split the O-ring in half vertically (left and right side of the O-ring)
# in theory, a ring is symmetrical so the thickness on each side of the O-ring should be approximately equal
# thicknesses are the number of foreground pixels calculated on each side
# O-ring will be classified as "Fail" as long as it's chipped or broken
def analyse_ring(coord, img):
    # used the coordinate of the bounding box to compute the geometry properties of the O-ring
    x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]
    centroid = (x1 + x2) / 2, (y1 + y2) / 2  # center of the O-ring
    radius = (x2 - x1) / 2  # diameter divided by 2

    # (1) initialize the thicknesses on each side and the difference in thicknesses to 0
    thickness_left, thickness_right, normalized_diff = 0, 0, 0
    diff_limit = 0.04
    pass_image = True  # classification result

    # (2) perform a side-by-side analysis (progress from left to right for each row)
    # if a foreground pixel is found increment the corresponding thickness by 1
    for i in range(y1, y2+1):  # rows
        for j in range(x1, int(centroid[0]+1)):  # left side of the O-ring (x1 to centre)
            if img[i, j] == 255:
                thickness_left += 1

        for j in range(int(centroid[0]+1), x2+1):  # right side of the O-ring (centre to x2)
            if img[i, j] == 255:
                thickness_right += 1

        # (3) compute the average difference in thicknesses for each row
        # normalize the difference: divide by the radius to make every thickness to be exactly one pixel unit
        normalized_diff = abs(thickness_left - thickness_right) / (radius * 2)
        if round(normalized_diff, 3) <= diff_limit:  # move on to the next row if the difference is negligible
            thickness_left = 0
            thickness_right = 0
        else:  # the O-ring is classified as "Fail"
            pass_image = False
            break

    print('Centroid: %s' % (centroid,))
    print('Radius: %.1f' % radius)

    if pass_image:
        print('Result: O-ring is not flawed.')
        return 'PASS'
    else:
        print('Result: O-ring is flawed.')
        return 'FAIL'


# add colored annotations to the images
def annotate_image(img, text):
    colored_img = cv.cvtColor(np.uint8(img), cv.COLOR_GRAY2BGR)  # convert grayscale to RGB
    # params: image, text, coord, font, fontScale, color, thickness
    cv.putText(colored_img, text, (1, 210), cv.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
    return colored_img


# load all 15 images from resource folder
res = os.listdir('res')

for x in range(len(res)):
    original_image = cv.imread('res/Oring%d.jpg' % (x + 1), 0)  # specify 0 to read image in grayscale mode

    histogram = imhist(original_image)
    plt.plot(histogram)
    # TODO: uncomment this line of code to show the histogram
    # plt.show()

    # process each image
    before = time.time()
    print('----- Processed Image #%d -----' % (x + 1))

    # (1) Thresholding - compute a threshold value for each image using clustering algorithm (more accurate)
    thresholding(original_image, original_image.mean())  # use the average grey level as the initial threshold value

    # (1) Thresholding - compute a threshold value for each image using histogram-based algorithm (significantly faster)
    # TODO: uncomment this line of code to use the histogram-based algorithm (comment out the thresholding method above)
    # apply_threshold(original_image, calculate_threshold_histogram(histogram))

    # (2) Binary Morphology - perform dilation and erosion to fill interior holes
    dilated_image = dilate(original_image)  # fill interior holes
    eroded_image = erode(dilated_image)  # erode once to retain the original thickness of the O-ring

    # (3) Connected Component Labelling (CCL) - extract the foreground pixels, i.e. pixels belonging to the O-ring
    # assign labels to groups of pixels that are connected
    ring_label, labelled_image = labelling(eroded_image)
    # remove broken pieces using the labels assigned
    ring_image = remove_broken_piece(ring_label, eroded_image, labelled_image)

    # (4) extract the contours of the labelled O-ring
    contoured_image = extract_contours(ring_label, ring_image, labelled_image)

    # (5) generate a bounding box around the labelled O-ring
    coordinate = calculate_coordinate(contoured_image)  # x1, y1, x2, y2
    box_image = cv.rectangle(contoured_image, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), 255, 1)

    # (6) analyse the extracted O-ring region
    result = analyse_ring(coordinate, ring_image)

    # calculate the image processing time
    after = time.time()
    result = 'Processing Time: %.3fs | %s' % ((after - before), result)

    # add annotations to the images
    original_image = annotate_image(original_image, 'STEP 1: THRESHOLDED')
    dilated_image = annotate_image(dilated_image, 'STEP 2: DILATED')
    eroded_image = annotate_image(eroded_image, 'STEP 3: ERODED')
    labelled_image = annotate_image(labelled_image, 'STEP 4: LABELLED')
    ring_image = annotate_image(ring_image, 'STEP 5: EXTRACTED O-RING')
    box_image = annotate_image(box_image, result.upper())
    print(result + '\n')

    # display the output image in a single window
    output_image = np.concatenate((original_image, dilated_image, eroded_image,
                                   labelled_image, ring_image, box_image), axis=1)
    cv.imshow('Output Image #%d' % (x + 1), output_image)
    cv.waitKey(0)
    cv.destroyAllWindows()  # close any open windows

    x += 1  # loop through all images