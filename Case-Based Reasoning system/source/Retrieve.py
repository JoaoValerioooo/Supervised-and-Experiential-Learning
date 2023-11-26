import os
import numpy as np
import pandas as pd
import json


def calculate_distance(case1, case2, attr_dict):
    """
    Calculates the distance between two cases. For quantitative values, it uses the normalized
    Euclidean distance. For qualitative values, it uses 1 for the same value, 0 otherwise.
    For cyclical values such as months, closer months has closer distance
    """
    distance = 0

    for attr, value in case1.items():
        if case2.get(attr) != None and (attr_dict.get(attr)==None or (attr_dict.get(attr) and attr_dict.get(attr)["type"] != "meta_data")):
            if case2[attr] == value:  # for qualitative attributes
                distance += 0
            elif attr_dict.get(attr) and attr_dict.get(attr)["type"] == "range":  # for quantitative attributes
                # Normalize the values before calculating the distance
                # value1 = (value - attr_dict[attr]['min']) / (attr_dict[attr]['max'] - attr_dict[attr]['min'])
                # value2 = (case2[attr] - attr_dict[attr]['min']) / (attr_dict[attr]['max'] - attr_dict[attr]['min'])

                # distance += (value1 - value2) ** 2 # Euclidean
                distance += abs(value - case2[attr]) / (abs(value) + abs(case2[attr]))  # Canberra

            elif attr_dict.get(attr) and attr_dict.get(attr)[
                "type"] == "fixed":  # for fixed values, such as number of stars in hotel
                if attr_dict[attr]["val"].get(case2[attr]) and attr_dict[attr]["val"].get(value):
                    minimum = min(attr_dict[attr]["val"].values())
                    maximum = max(attr_dict[attr]["val"].values())
                    value1 = (attr_dict[attr]["val"][value] - minimum) / (maximum - minimum)
                    value2 = (attr_dict[attr]["val"][case2[attr]] - minimum) / (maximum - minimum)

                    distance += (value1 - value2) ** 2
                else:
                    distance += 1

            elif attr_dict.get(attr) and attr_dict.get(attr)["type"] == "cyclic":  # for cyclic values such as months
                maximum = max(attr_dict[attr]["val"].values())
                dist_1 = abs(attr_dict[attr]["val"][value] - attr_dict[attr]["val"][case2[attr]])
                dist_2 = min(attr_dict[attr]["val"][value], attr_dict[attr]["val"][case2[attr]]) + maximum - \
                         max(attr_dict[attr]["val"][value], attr_dict[attr]["val"][case2[attr]])

                normalized = min(dist_1, dist_2) / (maximum / 2)
                distance += normalized ** 2
            else:
                distance += 1

    return np.sqrt(distance)


def retrieve(case_base, new_case, data_folder, cases_to_retrive=1):
    """
    Retrieves the most similar case to new_case from case_base.

    Parameters:
    - case_base: A pandas DataFrame where each row represents a case.
    - new_case: A pandas Series representing the new case.

    Returns:
    - The index of the most similar case to new_case from case_base.
    - The calculated distance between the cases
    """
    attr_dict = get_attr_dict(data_folder)

    # Check if the new case has a value larger or smaller than those in the attribute  file, if so update the file
    for attr in new_case.index:
        if attr != 'num_acceptance' and attr != 'num_reject':
            if new_case[attr] and attr_dict.get(attr) and attr_dict.get(attr)["type"] == "range":
                if new_case[attr] < attr_dict[attr]['min']:
                    attr_dict[attr]['min'] = new_case[attr]
                elif new_case[attr] > attr_dict[attr]['max']:
                    attr_dict[attr]['max'] = new_case[attr]

    # with open(file_name, 'w') as f:
    #    json.dump(attr_dict, f)

    # Calculate the distance between the new case and all cases in the case base
    distances = case_base.apply(calculate_distance, axis=1, case2=new_case, attr_dict=attr_dict)
    distances.sort_values(inplace=True)

    # Get the index of the most similar case
    most_similar_case_indexes = distances.index[0:int(cases_to_retrive)]

    # Return the most similar case
    return most_similar_case_indexes, distances[distances.index[0:int(cases_to_retrive)]]


def get_attr_dict(data_folder):
    file_name = data_folder + "/attr_info.json"
    # Check if the min_max file exists, if not create one
    if not os.path.exists(file_name):
        print("File json file doesn't exist")
        return None
    else:
        with open(file_name, 'r') as f:
            attr_dict = json.load(f)
            return attr_dict
