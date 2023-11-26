
from Retrieve import retrieve
import os
import time
import numpy as np
import pandas as pd
from preprocessing import travel_dataset_xls_preprocessing
from Adaptation import null_adaptation, weighted_adaptation

#root = '/Users/clararivadulla/Repositories/SEL-PW3'
# root = os.path.dirname(os.getcwd())
root = os.getcwd()
travel_dataset_xls_preprocessing(root)
data_folder = f"{root}/data"
CB = pd.read_csv(f"{data_folder}/travel.csv")

# new_case = pd.Series([ "Bathing", 2499, 2, "Egypt", "Car", 14.0, "April", "TwoStars", "Hotel White House, Egypt"],
#                      index=[ "holiday-type", "price", "num-persons", "region", "transportation", "duration", "season", "accomodation", "hotel"])
new_case = pd.Series([ 2000, 4, "Egypt", "Car", 14.0, "April", "TwoStars"],
                     index=[ "price", "num-persons", "region", "transportation", "duration", "season", "accomodation"])
# Retrieve the most similar case
start_time = time.time()
most_similar_cases, distances = retrieve(CB, new_case, data_folder)
null_adapted_case = null_adaptation(new_case, CB.loc[most_similar_cases[0]])
weighted_adaptation_case = weighted_adaptation(new_case, CB.loc[most_similar_cases])
end_time = time.time()

print("------ Most similar case ------")
print(CB.loc[most_similar_cases[0]])
print("----- Null adapted case -----")
print(null_adapted_case)
print("----- Weight adapted case -----")
print(weighted_adaptation_case)
print("------ ----------------- ------")
print("Time taken:", end_time - start_time, "seconds")
print("distance:", distances[0])