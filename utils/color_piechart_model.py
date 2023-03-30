# import libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
from PIL import Image
from scipy.spatial import KDTree
from webcolors import css3_hex_to_names, hex_to_rgb
import pandas as pd
import requests
from time import time
import json


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def convert_rgb_to_names(rgb_tuple):
    # a dictionary of all the hex and their respective names in css3
    css3_db = css3_hex_to_names
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'


def get_colors(image, number_of_colors, show_chart):
    modified_image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
    start = time()
    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    a = Counter(labels)
    key_to_delete = max(a, key=lambda k: a[k])
    del a[key_to_delete]
    # sort to ensure correct color percentage
    a = dict(sorted(a.items()))
    # print(a)
    counts = {i: v for i, v in enumerate(a.values())}

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    color_names = {}
    counter = 0
    for i in range(len(hex_colors)):
        # print(hex_colors[i])
        x = convert_rgb_to_names(hex_to_rgb(hex_colors[i])).title()
        color_names[counter] = x
        counter = counter + 1

    hex_col = {}
    for pos, value in enumerate(hex_colors):
        hex_col[pos] = value

    total_sum = sum(counts.values())
    color_details = {}
    for key in color_names.keys():
        hexa_value = str(hex_col[key])
        color = {}
        color["color_name"] = str(color_names[key])
        color["percentage"] = (counts[key] / total_sum) * 100
        color_details[hexa_value] = color
        if color["percentage"] >= 80:
            x = color_details.pop(hexa_value)
    # if (show_chart):
    #    plt.figure(figsize = (8, 6))
    #    plt.pie(counts.values(),labels =color_names.values(),colors = hex_colors,autopct='%1.2f%%')
    #    plt.savefig('C:/Users/wfp117320/Desktop/b/'+'pie_chart'+'.jpg')

    return color_details


def remove_bg(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # color range set to remove
    white = (255, 255, 255)
    dark_grey = (80, 80, 80)

    mask = cv2.inRange(image, dark_grey, white)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result


def color_pie_chart(image_path):
    start = time()
    result = remove_bg(image_path)
    # plt.imshow(result)
    # plt.show()
    color_details = get_colors(result, 11, True)
    final_data = color_details
    final_data = json.dumps(final_data, indent=1)
    final_data = json.loads(final_data)
    final_data = str(final_data).replace("'", '"')
    return str(final_data)
