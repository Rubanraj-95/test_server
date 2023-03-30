# Import required packages
import numpy as np
from math import sqrt

interval = 0.2
range_thresh_percent = 0.20    # aka 15 percent / 15%


def size_uniformity(sizes):
    sizes.sort()
    size_count = len(sizes)
    ranges = np.arange(0.0, 40.0, interval)
    i = 0
    size_dict = {}

    for r in ranges:
        count = 0
        if i == size_count:
            size_dict[str(r)] = size_dict.get(str(r), 0) + count
            continue

        while sizes[i] < r:
            count += 1
            i += 1
            if i == size_count:
                break
        size_dict[str(r)] = size_dict.get(str(r), 0) + count
    
    count = size_count-i
    size_dict['21'] = size_dict.get(str('21'), 0) + count

    return size_dict


def normal_dist(y, x):
    # mean = 0
    # variance = 0
    # sd = 0
    fx = []
    for i in range(len(x)):
        fx.append(x[i]*y[i])
    sum_fx = sum(fx)
    sum_x = sum(x)
    mean = sum_fx / sum_x
    
    x_minus_xbar = []
    x_minus_xbar_square = []
    fx_minus_xbar_square = []
    for i in range(len(x)):
        x_minus_xbar.append(round(x[i]-mean, 2))
        x_minus_xbar_square.append(round(abs(x_minus_xbar[i]*x_minus_xbar[i]), 2))
        fx_minus_xbar_square.append(round(x[i]*x_minus_xbar_square[i], 2))

    variance = sum(fx_minus_xbar_square)/(sum_x-1)
    sd = sqrt(variance)
    return mean, variance, sd


def uniformity_percent(y, x, w_size, interval):
    tot_count = sum(x)
    max_count = -1
    i = 0
    j = interval
    range_i = i
    range_j = j
    while j < len(x):
        count = sum(x[i:j+1])
        if count > max_count:
            max_count = count
            range_i = i
            range_j = j
        i = i + 1
        j = j + 1
    
    percent = (max_count/tot_count)*100

    return y[range_i], y[range_j], percent


def size_uniformity_master_function(sizes):
    """

    :param sizes:
    :return:
    """

    if len(sizes) == 0:
        return {"mean": 0,
                "variance": 0,
                "standard_deviation": 0,
                "window_size": 0,
                "size_uniformity_percent": 0.0,
                "range_interval": "No objects detected.."}

    size_dict = size_uniformity(sizes)

    ranges = []
    for i in size_dict.keys():
        ranges.append(float(i))
    count = list(size_dict.values()) 
    mean, variance, sd = normal_dist(ranges[:-1], count[:-1])

    w_size = round(mean * range_thresh_percent, 1)
    # if int((w_size*10)%2) == 1:
    #    w_size = round(w_size + 0.1,1)
        
    skips = int(round((w_size / 0.1)) / 2)
    x, y, percent = uniformity_percent(ranges[:-1], count[:-1], w_size, skips-1)

    # plt.plot(ranges[:-1],count[:-1])
    # plt.show()

    x = round(x, 2)
    y = round(y, 2)
    
    final_data = {"mean": mean,
                  "variance": variance,
                  "standard_deviation": sd,
                  "window_size": w_size,
                  "size_uniformity_percent": percent,
                  "range_interval": str(x) + " mm - " + str(y) + " mm",
                  "range_from": str(x),
                  "range_to": str(y)}
    # final_data = json.dumps(final_data, indent=1)
    # final_data = json.loads(final_data)
    # final_data = str(final_data).replace("'", '"')
    return final_data
