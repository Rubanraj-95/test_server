from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from PIL import Image
import glob
import numpy as np
from numpy import arange
import imutils
import cv2
import pandas as pd
import time as t
from datetime import datetime
import urllib.request
import os
import requests
import json
from utils import *
# from .imgpro_queries import *
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

# output_images_path = "/home/outgrow-dluser/iGrade_uat/quick_imgpro/size_output_images/"
output_images_path = "size_output_images/"
ref_width = 17.2


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def size_detection(ticket_id, image_id, image_path):
    rescale_images(image_path)
    image = cv2.imread(image_path)

    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 60, 10)
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)
    # cv2.imwrite(output_images_path+ str(ticket_id)+"_"+str(image_id) +"_canny.png", edged)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually

    counter = 0
    cnt = 1
    output_image = image.copy()
    final_data = []
    length_list = []
    width_list = []
    for c in cnts:
        # if the contour area is large, ignore it
        if cv2.contourArea(c) < 50 and counter:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4 and counter and cv2.contourArea(c) > 12000:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if abs(dA - dB) > 800:
            continue

        if pixelsPerMetric is None:
            if cv2.contourArea(c) < 1000:
                continue
            pixelsPerMetric = dB / ref_width
            # pixelsPerMetric = dB / args["width"]

            # compute the size of the object
        new = []
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        g_length = max(dimA, dimB)
        g_width = min(dimA, dimB)

        g_length_str = str(round(g_length, 2))
        g_width_str = str(round(g_width, 2))

        # print('#########',cnt,g_length)
        if cnt == 1 and g_length <= 10 and g_length <= 25:
            # error... update rd_status as 4
            update_rd_status(ticket_id, 4)
            print("Inside If Check")
            break
        else:
            # passed.. detected the black box
            # print("Inside Else Check")
            cnt = 0

        # if g_length>40:
        #    continue;

        # Draw the object sizes on the image
        if counter == 0:
            cv2.line(output_image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 0), 2)
            cv2.line(output_image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 0), 2)
            # cv2.putText(output_image, "{:.1f}mm".format(cv2.contourArea(c)),(int(trbrX + 10), int(trbrY)),
            # cv2.FONT_HERSHEY_SIMPLEX,0.95, (255, 255, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, (255, 0, 0), 2)
            counter = 1
            continue
        else:
            cv2.line(output_image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(output_image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
            # cv2.putText(output_image, "{:.1f}mm".format(cv2.contourArea(c)),(int(trbrX + 10), int(trbrY)),
            # cv2.FONT_HERSHEY_SIMPLEX,0.95, (255, 255, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, (255, 255, 0), 2)
            # insert size values into DB 

            single_grain_details = {"length": g_length_str, "width": g_width_str}
            length_list.append(g_length)
            width_list.append(g_width)
            final_data.append(single_grain_details)
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + ".png", output_image)

    # final_data = json.dumps(final_data, indent=1)
    # final_data = json.loads(final_data)
    # final_data = str(final_data).replace("'", '"')

    return final_data, length_list, width_list
