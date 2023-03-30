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

ref_width = 17.0
# output_images_path = "/home/outgrow-dluser/iGrade_uat/quick_imgpro/size_output_images/"
output_images_path = "size_output_images/"


def get_yintersection(a, b):  # (a,b) == ([ymin,ymax] of i,[ymin,ymax] of j)
    # find min(min_ay,min_by),  if got min_by
    # check if min_ay falls between min_by and max_by: if true **min_ay(start point)** -
    # find min(max_ay,max_by): if got max_by: **max_by(end point)** ,  else return 0
    # min_ay * max_by == IOU from y side alone.. and should see for x side to calculate area of IOU

    if a[0] <= b[0]:
        if b[0] <= a[1]:
            start_point = b[0]
            end_point = min(a[1], b[1])
            return end_point - start_point
        else:
            return 0
    else:  # b[0]>a[0]
        if a[0] <= b[1]:  # intersects and a[0] is strart point
            start_point = a[0]
            end_point = min(a[1], b[1])
            return end_point - start_point

        else:  # since both a[ymin] and a[ymax] falls away from b[ymin,ymax] and so didnt intersects.
            return 0


def get_xintersection(a, b):
    if a[0] <= b[0]:
        if b[0] <= a[1]:
            start_point = b[0]
            end_point = min(a[1], b[1])
            return end_point - start_point
        else:
            return 0
    else:  # b[0]>a[0]
        if a[0] <= b[1]:
            start_point = a[0]
            end_point = min(a[1], b[1])
            return end_point - start_point

        else:
            return 0
    return x_iou


def check_overlap(co_box, coordinates):
    for coordinate in coordinates:
        y_iou = get_yintersection([coordinate[0], coordinate[2]], [co_box[0], co_box[2]])
        if y_iou:  # not equal to 0
            x_iou = get_xintersection([coordinate[1], coordinate[3]], [co_box[1], co_box[3]])
            area_iou = y_iou * x_iou

            y_i = coordinate[2] - coordinate[0]
            x_i = coordinate[3] - coordinate[1]
            y_j = co_box[2] - co_box[0]
            x_j = co_box[3] - co_box[1]

            area_i = y_i * x_i
            area_j = y_j * x_j

            percent_wrt_i = int((area_iou / area_i) * 100)
            percent_wrt_j = int((area_iou / area_j) * 100)
            # print(area_iou,area_i,area_j,percent_wrt_i,percent_wrt_j)

            # if percent_wrt_i > 40 and percent_wrt_j >40 or area_iou > 0:
            # i has high confidence score that j in default, so ignoring threshold comparison
            # if not j in del_indexes:
            #    del_indexes.append(j)
            if percent_wrt_i > 50 and percent_wrt_j > 50:
                print(co_box, coordinate)
                return 1, coordinates
    return 0, coordinates


def get_coordinates(boxes, h, w):
    coordinates = []
    # ymin, xmin, ymax, xmax = box
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        ymin = int(h * ymin)
        xmin = int(w * xmin)
        ymax = int(h * ymax)
        xmax = int(w * xmax)
        coordinates.append([ymin, xmin, ymax, xmax])
    return coordinates


def get_height_width(image_path):
    img = cv2.imread(str(image_path))
    h, w, c = img.shape
    return h, w


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def rem_sizes(img_path, coordinates, ticket_id, image_id):
    image = cv2.imread(img_path)
    print(image.shape)
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 60, 10)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    affan = []
    counter = 0
    cnt = 1
    output_image = image.copy()
    size_boxes = []
    length_list = []
    final_data = []
    for c in cnts:
        # if the contour area is large, ignore it
        if cv2.contourArea(c) < 50 or cv2.contourArea(c) > 12000 and counter:
            continue
        x, y, h, w = cv2.boundingRect(c)
        if w > 600 or h > 600:
            continue
        co_box = [y, x, y + w, x + h]
        status = 0
        if coordinates != []:
            status, coordinates = check_overlap(co_box, coordinates)
        if status == 1:  # overlaped
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

        if abs(dA - dB) > 400:
            continue

        if pixelsPerMetric is None:
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
        if (cnt == 1 and g_length <= 10 and g_length <= 25):
            # error... update rd_status as 4
            update_rd_status(ticket_id, 4)
            print("Inside If Check")
            break;
        else:
            # passed.. detected the black box
            # print("Inside Else Check")
            cnt = 0
        print(counter, g_length)
        if g_length > 40:
            continue

        # Draw the object sizes on the image
        if counter == 0:
            cv2.line(output_image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 0), 2)
            cv2.line(output_image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 0), 2)
            # cv2.putText(output_image, "{:.1f}mm".format(cv2.contourArea(c)),(int(trbrX + 10),
            # int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.95, (255, 255, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 0, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, (255, 0, 0), 2)
            counter = 1
            continue

        else:
            cv2.line(output_image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
            cv2.line(output_image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
            # cv2.putText(output_image, "{:.1f}mm".format(cv2.contourArea(c)),(int(trbrX + 10), int(tr
            # brY)), cv2.FONT_HERSHEY_SIMPLEX,0.95, (255, 255, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 0), 2)
            cv2.putText(output_image, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, (255, 255, 0), 2)
            # insert size values into DB

            single_grain_details = {"length": g_length_str, "width": g_width_str}
            length_list.append(g_length)
            final_data.append(single_grain_details)
    if counter == 0:
        print("[Error] : Black box not detected... ")
        return None
    cv2.imwrite(output_images_path + str(ticket_id) + "_" + str(image_id) + "_waste_rem.png", output_image)

    return final_data, length_list
