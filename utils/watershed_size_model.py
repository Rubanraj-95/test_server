from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from PIL import Image
import glob
import numpy as np
from numpy import arange
import imutils
import cv2
import logging
import pandas as pd
import time as t
from datetime import datetime
import urllib.request
import os
import requests
import json

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

# output_images_path = "/home/outgrow-dluser/iGrade_uat/quick_imgpro/size_output_images/"
output_images_path = "size_output_images/"

ref_width = 17.2
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]: %(message)s')


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def black_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(f'{circles}')
    try:
        if circles is None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass

    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        logging.info(f"circles - {circles}")
        i = circles[0][0]
        logging.info(f'i {i}')
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        logging.info(f'{count} {image[i[1]][i[0]]}')
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]

    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []

    circles = circles[0]
    logging.info(f"circle_rad {circles[0][-1]}")
    pixel_dia = circles[0][-1] * 2
    logging.info(f"pixel_dia {pixel_dia}")

    pixelsPerMetric = pixel_dia / ref_width
    logging.info("pixelpermeteric {pixelsPerMetric}")
    image = cv2.imread(image_path)
    gray = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(gray, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue
        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f'{output_images_path}{str(ticket_id)}_{str(image_id)}.png')
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def other_small_comm_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 120
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(f'{circles}')
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        logging.info(f"circles {circles}")
        i = circles[0][0]
        logging.info(f'i {i}')
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        logging.info(f'{count} {image[i[1]][i[0]]}')
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    logging.info("circle_rad {circles[0][-1]}")
    pixel_dia = circles[0][-1] * 2
    logging.info(f"pixel_dia {pixel_dia}")

    pixelsPerMetric = pixel_dia / ref_width
    logging.info(f"pixel per meteric {pixelsPerMetric}")
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f'{output_images_path}{str(ticket_id)}_{str(image_id)}.png')
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


###
def gka_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 50
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()
    # Load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        logging.info(f"circles {circles}")
        i = circles[0][0]
        logging.info(f'i {i}')
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        logging.info(f'{count} {image[i[1]][i[0]]}')
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    logging.info(f"circle_rad {circles[0][-1]}")
    pixel_dia = circles[0][-1] * 2
    logging.info(f"pixel_dia {pixel_dia}")

    pixelsPerMetric = pixel_dia / ref_width
    logging.info(f"pixel per meteric {pixelsPerMetric}")

    image = cv2.imread(image_path)
    dst = cv2.GaussianBlur(image, (9, 9), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(dst, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    dilate = cv2.dilate(thresh, np.ones((3, 3)), iterations=1)

    thresh = dilate

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=30, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 35.0 or float(length) < 1.5:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f'{output_images_path}{str(ticket_id)}_{str(image_id)}.png')
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def white_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        logging.info(f"circles {circles}")
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        logging.info(f'{count} {image[i[1]][i[0]]}')
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    logging.info(f"circle_rad {circles[0][-1]}")
    pixel_dia = circles[0][-1] * 2
    logging.info(f"pixel_dia {pixel_dia}")

    pixelsPerMetric = pixel_dia / ref_width
    logging.info(f"pixel per meteric {pixelsPerMetric}")
    image = cv2.imread(image_path)
    img = cv2.bitwise_not(image)

    dst = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(dst, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate=cv2.erode(thresh,np.ones((3,3)),iteration=1)
    # thresh=dilate

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f'{output_images_path}{str(ticket_id)}_{str(image_id)}.png')
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def rice_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        logging.info(f"circles {circles}")
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        logging.info(f"{count} {image[i[1]][i[0]]}")
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    logging.info(f"circle_rad {circles[0][-1]}")
    pixel_dia = circles[0][-1] * 2
    logging.info(f"pixel_dia {pixel_dia}")

    pixelsPerMetric = pixel_dia / ref_width
    logging.info(f"pixel per meteric {pixelsPerMetric}")
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=14, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 1.5:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f"{output_images_path}{str(ticket_id)}_{str(image_id)}.png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def maize_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)

    try:
        if circles is None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        logging.info(f"circles {circles}")
        i = circles[0][0]

        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        logging.info(f"{count} {image[i[1]][i[0]]}")
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    logging.info(f"circle_rad {circles[0][-1]}")
    pixel_dia = circles[0][-1] * 2
    logging.info(f"pixel_dia {pixel_dia}")

    pixelsPerMetric = pixel_dia / ref_width
    logging.info(f"pixel per metric {pixelsPerMetric}")
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=24, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f"{output_images_path}{str(ticket_id)}_{str(image_id)}.png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def wheat_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixelpermeteric", pixelsPerMetric)
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=18, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f"{output_images_path}{str(ticket_id)}_{str(image_id)}.png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def green_gram_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixelpermeteric", pixelsPerMetric)
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=14, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 0.8:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f"{output_images_path}{str(ticket_id)}_{str(image_id)}.png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def green_gram_splits_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]

        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=14, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 1.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f"{output_images_path}{str(ticket_id)}_{str(image_id)}.png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def sesame_seeds_wg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixelpermeteric", pixelsPerMetric)
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    img = cv2.bitwise_not(image)

    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    print(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def niger_seeds_wg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixelpermeteric", pixelsPerMetric)
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)
    img = cv2.bitwise_not(image)

    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=15, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f'{output_images_path}{str(ticket_id)}_{str(image_id)}.png')
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def black_pepper_wg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []

    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    # gray = cv2.GaussianBlur(image, (7, 7), 0)

    img = cv2.bitwise_not(image)

    dst = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(dst, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    proc = thresh

    dilate = cv2.dilate(proc, np.ones((3, 3)), iterations=9)
    proc = dilate

    D = ndimage.distance_transform_edt(proc)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=proc)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=proc)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    print(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def kidney_beans_wg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]

        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    img = cv2.bitwise_not(image)

    dst = cv2.GaussianBlur(img, (9, 9), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(dst, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=85, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    print(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def ragi_wg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    img = cv2.bitwise_not(image)

    # dst = cv2.GaussianBlur(img, (9, 9),cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=13, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    print(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def red_gram_white_bg_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    gray = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(gray, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue
        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    print(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def cardamom_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    img = cv2.bitwise_not(image)

    dst = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(dst, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    erode = cv2.erode(thresh, np.ones((3, 3)), iterations=10)

    D = ndimage.distance_transform_edt(erode)
    localMax = peak_local_max(D, indices=False, min_distance=60, labels=erode)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=erode)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    print(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def pista_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    img = cv2.bitwise_not(image)

    dst = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(dst, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # erode=cv2.erode(thresh,np.ones((3,3)),iteration=10)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=85, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    print(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list


def raisin_size_detection(ticket_id, image_id, image_path, coin_diameter):
    ref_width = coin_diameter
    minDist = 100
    param1 = 30
    param2 = 50
    minRadius = 100
    maxRadius = 300
    image = cv2.imread(image_path)
    # image = cv2.medianBlur(image,5)
    # rows=max(image.shape)
    img_copy = image.copy()

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    logging.info(circles)
    try:
        if circles == None:
            logging.info("No circles detected..")
            return 0, {}, [], []

    except Exception as e:
        pass
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # circles= circles[0]
        print("circles", circles)
        i = circles[0][0]
        count += 1
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(count, image[i[1]][i[0]])
        cv2.putText(image, str(count), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)
        pixel = i[2]
    if len(circles) > 1:
        logging.info("More than one circles..")
        return 4, {}, [], []
    circles = circles[0]
    print("circle_rad", circles[0][-1])
    pixel_dia = circles[0][-1] * 2
    print("pixel_dia", pixel_dia)

    pixelsPerMetric = pixel_dia / ref_width
    print("pixel per metric", pixelsPerMetric)
    image = cv2.imread(image_path)
    img = cv2.bitwise_not(image)

    dst = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    shifted = cv2.pyrMeanShiftFiltering(dst, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    dilate = cv2.erode(thresh, np.ones((3, 3)), iteration=1)
    thresh = dilate

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=40, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

    out_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    l = []
    for label in np.unique(labels):
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        l.append(r)
        length = (2 * r) / pixelsPerMetric
        length = round(length, 2)
        if float(length) > 20.0 or float(length) < 2.0:
            continue

        cv2.circle(out_image, (circles[0][0], circles[0][1]), circles[0][2], (255, 255, 0), 9)
        cv2.circle(out_image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(out_image, "#{}".format(length), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                    2)
        cv2.putText(out_image, "Coin", (circles[0][0], circles[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 0), 8)

        single_grain_details = {"length": length, "width": length}
        length_list.append(length)
        width_list.append(length)
        final_data.append(single_grain_details)
    logging.info(f"{output_images_path}{str(ticket_id)}_{str(image_id)}.png")
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", out_image)

    return 3, final_data, length_list, width_list
