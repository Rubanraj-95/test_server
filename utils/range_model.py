
import numpy as np

range_interval = 0.5
start_range = 0.0
end_range = 40.0


def range_calculation(sizes_list):
    """

    :param sizes_list:
    :return:
    """

    sizes_list.sort()
    total_size_count = len(sizes_list)
    final_data = []
    ranges = np.arange(start_range, end_range + range_interval, range_interval)
    s_i = 0
    max_s_i = total_size_count - 1
    for i in range(len(ranges) - 1):
        count = 0
        if s_i != -1 and s_i <= max_s_i:
            # (s_i <= max_s_i and sizes_list[s_i] >= ranges[i] and sizes_list[s_i] < ranges[i + 1]):
            while s_i <= max_s_i and ranges[i] <= sizes_list[s_i] < ranges[i + 1]:
                count += 1
                s_i += 1
        range_value = str(round(ranges[i], 2)) + " - " + str(round(ranges[i + 1], 2))
        percent = 0.0

        if total_size_count != 0:
            percent = (count / total_size_count) * 100
            percent = round(percent, 2)

        print(range_value, count, percent)
        single_range_details = {"range": range_value, "count": count, "range_percentage": percent}
        final_data.append(single_range_details)
    # final_data = json.dumps(final_data, indent=1)
    # final_data = json.loads(final_data)
    # final_data = str(final_data).replace("'", '"')
    return final_data


def grade_range(sizes_list, range_grade_details):
    """

    :param sizes_list:
    :param range_grade_details:
    :return:
    """

    igrade_names = ['Small', 'Regular', 'Bold']
    igrade_names_id = {'Small': '1', 'Regular': '2', 'Bold': '3'}
    names_i = 0
    sizes_list.sort()
    total_size_count = len(sizes_list)
    final_data = []
    small_s, small_e, regular_s, regular_e, bold_s, bold_t = float(range_grade_details["small_s"]), float(
        range_grade_details["small_t"]), float(range_grade_details["regular_s"]), float(
        range_grade_details["regular_t"]), float(range_grade_details["bold_s"]), float(range_grade_details["bold_t"])

    ranges = [small_s, small_e, regular_s, regular_e, bold_s, bold_t]

    s_i = 0
    max_s_i = total_size_count - 1
    # print("\nmax_s_i ---------------- ",max_s_i)
    bold_count = 0
    others_count = 0
    for i in range(len(ranges) - 1):
        count = 0
        if s_i != -1 and s_i <= max_s_i:
            # for loop continues if range value of the next is same as the current
            if ranges[i] == ranges[i + 1] and s_i <= max_s_i:
                continue

            while s_i <= max_s_i and ranges[i] <= sizes_list[s_i] < ranges[i + 1]:
                count += 1
                s_i += 1
        range_value = str(round(ranges[i], 2)) + " mm - " + str(round(ranges[i + 1], 2)) + " mm"
        percent = 0.0
        if total_size_count != 0:
            percent = (count / total_size_count) * 100
            percent = round(percent, 2)
        # print(range_value,count,percent)
        single_range_details = {"range": range_value, "count": count, "range_percentage": percent,
                                "range_grade_type": igrade_names[names_i], "range_from": str(round(ranges[i], 2)),
                                "range_to": str(round(ranges[i + 1], 2)),
                                "grade_id": igrade_names_id[igrade_names[names_i]]}
        if igrade_names[names_i] == 'Bold':
            bold_count += count
        else:
            others_count += count
        names_i += 1
        final_data.append(single_range_details)

    return final_data, bold_count, others_count
