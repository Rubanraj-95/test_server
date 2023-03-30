# Import required packages
import json
import logging
import os
import urllib.request
import time as t

from datetime import datetime
from flask import Flask, request
from config import *
from utils.quality_queries import *
from utils.range_model import *
from utils.size_uniformity_model import *


logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s]: %(message)s')

app = Flask(__name__)


def range_model(length_sizes_list):
    """
    Function to calculate the length range of all the grains.
    :param length_sizes_list: List of grains length to calculate the range.
    :return: Return the list of dictionaries with range details.
    """

    try:
        data = range_calculation(length_sizes_list)
        return 3, data

    except Exception as e:
        logging.error(f'Error in range model - {e}')
        return 4, []


def size_uniformity_model(sizes_list):
    """
    Function to uniformize the detected object size and calculate mean, variance,
    standard_deviation, window size, range intervel and ranges.
    :param sizes_list: List of detected object sizes.
    :return: Return the Dictionary of values.
    """

    try:
        data = size_uniformity_master_function(sizes_list)
        return 3, data
    except Exception as e:
        logging.error(f'Error in size uniformity model - {e}')
        return 4, {}


def club_range(range_data):
    checker = 0
    clubbed_percent = 0.0
    range_ = '0.0 - '
    count_ = 0
    exists = False
    while range_data[0]['range_percentage'] < range_clubbing_percent:
        exists = True
        if range_data[0]['range_percentage'] == 0.0 and checker == 0:
            range_ = range_data[0]['range'].split("-")[-1][1:] + ' - '
            del range_data[0]
        else:
            checker = 1
            clubbed_percent += range_data[0]['range_percentage']
            count_ += range_data[0]['count']
            del range_data[0]

        if not range_data:
            return [{'range': '0.0 - 60.0',
                     'count': 0,
                     'range_percentage': 0.0}]

    if exists:
        range_ = range_ + range_data[0]['range'].split("-")[0][:-1]
        dic = {'range': range_,
               'count': count_,
               'range_percentage': clubbed_percent}

        range_data.insert(0, dic)

    checker = 0
    clubbed_percent = 0.0
    range_ = ' - 60.0'
    count_ = 0
    exists = False
    while range_data[-1]['range_percentage'] < range_clubbing_percent:
        exists = True
        if range_data[-1]['range_percentage'] == 0.0 and checker == 0:
            del range_data[-1]
            range_ = ' - ' + range_data[-1]['range'].split("-")[-1][1:]

        else:
            checker = 1
            clubbed_percent += range_data[-1]['range_percentage']
            count_ += range_data[-1]['count']
            del range_data[-1]

    if exists:
        range_ = range_data[-1]['range'].split("-")[-1][1:] + range_
        dic = {'range': range_,
               'count': count_,
               'range_percentage': clubbed_percent}
        range_data.append(dic)
    logging.info(f'Clubbing Completed....{range_data}')
    return range_data


def find_width_min_max_avg(overall_size_data):
    """
    Function to calculate the average, minimum and maximum values of width.
    :param overall_size_data: Overall detected objects size data.
    :return: Return the minimum, maximum and average values of width.
    """

    if not overall_size_data:
        return 0.0, 0.0, 0.0

    mini = 10000.0
    maxi = 0.0
    sum = 0.0
    for i in overall_size_data:
        wid = float(i['width'])
        sum = sum + wid
        if wid > maxi:
            maxi = wid
        if wid < mini:
            mini = wid

    aver = sum / len(overall_size_data)
    logging.info(f"WIDTH :   min-", {mini}, '     maxi-', {maxi}, '    average-', {aver})
    return maxi, mini, aver


def find_length_min_max_avg(overall_size_data):
    """
    Function to calculate the average, minimum and maximum values of width.
    :param overall_size_data: Overall detected objects size data.
    :return: Return the minimum, maximum and average values of width.
    """

    if not overall_size_data:
        return 0.0, 0.0, 0.0

    mini = 10000.0
    maxi = 0.0
    sum = 0.0
    for i in overall_size_data:
        leng = float(i['length'])
        sum = sum + leng
        if leng > maxi:
            maxi = leng
        if leng < mini:
            mini = leng

    aver = sum / len(overall_size_data)
    logging.info("LENGTH :  min-", {mini}, '     maxi-', {maxi}, '    average-', {aver})
    return maxi, mini, aver


def quality_yolo_models(ticket_id, image_id, category_id, commodity_id, variety_id, sub_variety_id,
                        total_wt, url_image_path, coin_diameter, server):
    """
    Function to call the YOLO model to detect the object and calculate the size of objects.
    :param ticket_id: Ticket ID of this request.
    :param image_id: Image ID of this request.
    :param category_id: Category ID of this request.
    :param commodity_id: Commodity ID of this request.
    :param variety_id: Variety ID of this request.
    :param sub_variety_id: Sub Variety ID of this request.
    :param total_wt: Total weight entered by the user.
    :param url_image_path: Image URL path to download the request image.
    :param coin_diameter: Coin Diameter entered by the user.
    :param server: Name of the server handling the request.
    :return: Return the prediction and size data of the request.
    """

    try:
        logging.info(f"Quality Classification model started..")
        commodity_details = category_id + '_' + commodity_id + '_' + variety_id + '_' + sub_variety_id

        logging.info(f'Commodity_details - {commodity_details}')
        port_no = yolo_port_no_map[commodity_details][0]
        url = yolo_port_no_map[commodity_details][1]

        logging.info(f'{url}{port_no}')
        my_url = url + port_no + '/v1.0_predict_quality_yolo_models'
        logging.info(f'{my_url}')

        form_data = {"ticket_id": ticket_id,
                     "image_id": image_id,
                     "category_id": category_id,
                     "commodity_id": commodity_id,
                     "variety_id": variety_id,
                     "sub_variety_id": sub_variety_id,
                     "image_total_weight": total_wt,
                     "url_image_path": url_image_path,
                     "coin_diameter": coin_diameter,
                     "server": server}

        get_data = requests.post(my_url, form_data)
        data = json.loads(get_data.text)

        logging.info(f'{data}')
        if data["status"] == 3:
            return data["status"], data["data"], data["db_class_counts"], data["db_class_wt"], \
                data["db_classname_map"], data["coordinates"]
        else:
            return 4, {}, {}, {}, {}, []

    except Exception as e:
        logging.error(f'Error in Yolo model - {e}')
        return 4, {}, {}, {}, {}, []


# Flask API for Defect Analysis
@app.route('/v1.0_quality_yolo_models', methods=['GET', 'POST'])
def quality_yolo():
    """
    Flask API to get ticket_id, user_id, server name from request.
    Load the yolo model based on commodity and process the predicted model results.
    Form the result data in a json format and update the date-base with result.
    :return: Return the message with status code.
    """

    start = t.time()
    if request.method == 'POST':
        # Get the request details
        ticket_id = request.form['ticket_id']
        user_id = request.form['user_id']
        server = request.form['server']

        mydb, connected = db_connect_quality(server)

        # Get the details of the request ticket using ticket_id
        ticket_details = get_ticketid_details_quality(ticket_id)[0]
        if not ticket_details:
            logging.error(f'Requested ticket id is not created in DB !!!')
            return {"message": "ticket-id : " + str(ticket_id) + " not exists", "status": "0"}

        logging.info(f'Ticket Details - {ticket_details}')

        ticket_id = str(ticket_details[0])
        category_id = str(ticket_details[1])
        commodity_id = str(ticket_details[2])
        variety_id = str(ticket_details[3])
        sub_variety_id = str(ticket_details[4])
        coin_diameter = str(ticket_details[7])

        logging.info(f'coin_diameter - {coin_diameter}')

        if ticket_details[5] is None:
            image_wise_data = {}
            combined_data = {}
        else:
            image_wise_data = json.loads(str(ticket_details[5]))
            combined_data = json.loads(str(ticket_details[6]))

        if image_wise_data is not None:
            logging.error(f'image_wise_data - combined_data - {image_wise_data} - {combined_data}')

        # Get the image details from database
        image_details = get_image_details_quality(ticket_id)
        if not image_details:
            logging.error(f'No images present for the ticket id: {ticket_id}')
            return {"message": "No Images details for the ticket-id : " + str(ticket_id), "status": "0"}

        invalid_wt_check = 0
        overall_quality_count = {}
        overall_quality_wt = {}
        combined_actual_total_weight = 0.0

        overall_length_sizes_list = []
        overall_size_data = []
        db_classname_map = {}
        db_class_counts = {}

        for row in image_details:
            image_id = str(row[0])
            url_image_path = str(row[1])
            imgwise_total_wt = str(row[2])
            combined_actual_total_weight = combined_actual_total_weight + float(imgwise_total_wt)

            quality_status, quality_data, db_class_counts, db_class_wt, db_classname_map, coordinates = quality_yolo_models(
                ticket_id,
                image_id,
                category_id,
                commodity_id,
                variety_id,
                sub_variety_id,
                imgwise_total_wt,
                url_image_path,
                coin_diameter,
                server)

            quality_res_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logging.info(f"Quality Data : {quality_data}")

            if image_wise_data is None or image_wise_data == {} or image_wise_data == 0:
                image_wise_data = {}
                image_id_data = {"quality_data": json.loads(quality_data),
                                 "quality_status": quality_status,
                                 "quality_res_time": quality_res_time,
                                 "actual_image_weight": imgwise_total_wt}
                image_wise_data[image_id] = image_id_data
            else:
                if str(image_id) in image_wise_data.keys():
                    image_wise_data[image_id]["quality_data"] = quality_data
                    image_wise_data[image_id]["quality_status"] = quality_status
                    image_wise_data[image_id]["quality_res_time"] = quality_res_time
                    image_wise_data[image_id]["actual_image_weight"] = imgwise_total_wt
                else:
                    image_id_data = {"quality_data": json.loads(quality_data),
                                     "quality_status": quality_status,
                                     "quality_res_time": quality_res_time,
                                     "actual_image_weight": imgwise_total_wt}

                    image_wise_data[image_id] = image_id_data

            overall_length_sizes_list.extend(coordinates[1])
            overall_size_data.extend(coordinates[0])

            if quality_status == 3:
                for k in db_class_counts.keys():
                    overall_quality_count[k] = overall_quality_count.get(k, 0) + db_class_counts[k]
                    if db_class_wt == {}:
                        invalid_wt_check = 1
                        overall_quality_wt[k] = overall_quality_wt.get(k, 0) + 0.0
                    else:
                        overall_quality_wt[k] = overall_quality_wt.get(k, 0) + db_class_wt[k]
            else:
                logging.error(f"Quality status not 3")

        overall_quality_data = []
        tot_count = sum(overall_quality_count.values())
        tot_wt = sum(overall_quality_wt.values())
        for key in overall_quality_count.keys():
            if tot_count == 0:
                count_percent = 0.0
            else:
                count_percent = float((db_class_counts[key] / tot_count) * 100)

            if int(tot_wt) == 0:
                wt_percent = 0.0
            else:
                wt_percent = float((overall_quality_wt[key] / tot_wt) * 100)

            quality_param_name = db_classname_map[key]
            overall_quality_data.append({"quality_parameter_id": key,
                                         "quality_parameter_name": db_classname_map[key],
                                         "count": overall_quality_count[key],
                                         "count_percentage": count_percent,
                                         "weight": overall_quality_wt[key],
                                         "weight_percentage": wt_percent})

        if combined_data is None or combined_data == 0:
            combined_data = {}

        # Calculate the range using length list
        range_status, overall_range_data = range_model(overall_length_sizes_list)

        # Calculating the size and uniformize using length list
        su_status, overall_su_data = size_uniformity_model(overall_length_sizes_list)

        # Club the ranges
        overall_clubbed_range_data = club_range(overall_range_data)

        # Calculate the max length, min length, overall average length
        combined_data["overall_max_length"], combined_data["overall_min_length"], combined_data[
            "overall_avg_lenght"] = find_length_min_max_avg(overall_size_data)

        # Calculate the max width, min width, overall average width
        combined_data["overall_max_width"], combined_data["overall_min_width"], combined_data[
            "overall_avg_width"] = find_width_min_max_avg(overall_size_data)

        combined_data["overall_quality_data"] = overall_quality_data
        combined_data["combined_actual_image_weight"] = combined_actual_total_weight
        combined_data["overall_total_count"] = tot_count

        combined_data["overall_size_data"] = overall_size_data
        combined_data["overall_range_data"] = overall_clubbed_range_data
        combined_data["overall_size_uniformity_data"] = overall_su_data

        image_wise_data = json.dumps(image_wise_data, indent=1)
        image_wise_data = json.loads(image_wise_data)
        image_wise_data = str(image_wise_data).replace("'", '"')

        combined_data = json.dumps(combined_data, indent=1)
        combined_data = json.loads(combined_data)
        combined_data = str(combined_data).replace("'", '"')
        logging.info(f'combined_data - {combined_data}')
        logging.info(f'image_wise_data - {image_wise_data}')

        update_quality_data(ticket_id, image_wise_data, combined_data)
        mydb.close()
        end = t.time()
        total_time = end - start
        logging.info(f"Total Time taken by Quality model: {total_time:.2f} sec")
        if int(invalid_wt_check) == 1:
            return {"message": "Invalid weight for any one of the images.. : " + str(ticket_id), "status": "1"}
        else:
            return {"message": "Quality process done for the ticket id : " + str(ticket_id), "status": "1"}
    else:
        return {"message": "Invalid request method 'GET'", "status": "1"}


if __name__ == '__main__':
    app.run(port=port_num)
