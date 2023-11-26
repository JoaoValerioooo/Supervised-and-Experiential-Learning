import pandas as pd
from Retrieve import retrieve

def add_new_case(case_base, new_case, data_folder, threshold = 0.3):
    distance = retrieve(case_base, new_case, data_folder, 1)[1].values[0]
    if distance > threshold:
        case_base = pd.concat([case_base, new_case.to_frame().T], ignore_index=True)
        print("New case added successfully!")
        case_base = forgetting(case_base)
    else:
        print("New case too similar to be added")
    return  case_base


def forgetting(case_base, threshold=0.5):
    """
    When a new case is added, the case base should forget
    the case with lowest utility score.
    """
    # print(case_base)
    case_base_utilities = case_base.copy()
    total_successes = case_base_utilities['num_acceptance'].sum()
    total_failures = case_base_utilities['num_rejected'].sum()
    case_base_utilities['utility'] = ((case_base_utilities['num_acceptance'] / total_successes - case_base_utilities['num_rejected'] / total_failures) + 1) / 2
    print(case_base_utilities)
    case_base_utilities['utility'] = pd.to_numeric(case_base_utilities['utility'])
    index_min = case_base_utilities['utility'].idxmin()
    if case_base_utilities.loc[index_min, 'utility'] < threshold:
        print(f"Forgetting case with index {index_min}, which had an utility of {case_base_utilities.loc[index_min, 'utility']}")
        case_base = case_base.drop(index_min)
    return case_base
