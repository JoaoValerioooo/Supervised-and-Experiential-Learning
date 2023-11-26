import json
import pandas as pd
import os

indexes=[ "holiday-type", "price", "num-persons", "region", "transportation", "duration", "season", "accomodation", "hotel", "num_acceptance", "num_rejected"]

def null_adaptation(new_case, most_similar_case):    
    adapted_case = []

    for index in indexes:
        if index in new_case:
            adapted_case.append(new_case[index])
        else:
            adapted_case.append(most_similar_case[index])

    return pd.Series(adapted_case, index=indexes)


def weighted_adaptation(new_case, most_similar_cases):
    # Get back the attributes description
    attr_dict = '\{\}'
    # file_name = os.path.dirname(os.getcwd()) + "/data/attr_info.json"
    file_name = os.getcwd() + "/data/attr_info.json"
    # Check if the min_max file exists, if not create one
    if not os.path.exists(file_name):
        print("File json file doesn't exist")
    else:
        with open(file_name, 'r') as f:
            attr_dict = json.load(f)

    change_selected_region = True
    adapted_case = []

    ### Rules
    # Filter by a given region or the most common one in the most_similar_cases
    region = most_similar_cases['region'].value_counts().idxmax()
    if 'region' in new_case and new_case['region'] is not None :
        # Validate if there are rows matching the selected region, if not keep the most common one
        if len(most_similar_cases[most_similar_cases['region'] == new_case['region']]) > 0:
            region = new_case['region']
            change_selected_region = False


    most_similar_cases = most_similar_cases[most_similar_cases['region'] == region]

    for index in indexes:

        if index == 'region' and change_selected_region:
            adapted_case.append(most_similar_cases[index].value_counts().idxmax())

        elif index == 'num_acceptance':
            adapted_case.append(1)

        elif index == 'num_reject':
            adapted_case.append(0)

        elif index in new_case and new_case[index] is not None :
            adapted_case.append(new_case[index])
            
        else:
            if attr_dict.get(index)is not None and attr_dict.get(index)["type"] == "range": # for quantitative attributes
                adapted_case.append(most_similar_cases[index].mean())
            else:
                adapted_case.append(most_similar_cases[index].value_counts().idxmax())
    return pd.Series(adapted_case, index=indexes)
