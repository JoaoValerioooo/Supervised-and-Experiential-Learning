import Learning
from preprocessing import travel_dataset_xls_preprocessing
from Retrieve import retrieve
from Adaptation import weighted_adaptation
from Learning import add_new_case
from IPython.display import display
import time
import customtkinter
import pandas as pd
import numpy as np
import os
import time
from tkinter import messagebox

root = os.getcwd()
#root = os.path.dirname(os.getcwd())
data_folder = f"{root}/data"

try:
    CB = pd.read_csv(f"{data_folder}/travel.csv")
except (FileNotFoundError, PermissionError):
    print("Impossible to find CVS case base, reading from XLS.")
    travel_dataset_xls_preprocessing(root)
    CB = pd.read_csv(f"{data_folder}/travel.csv")

print(CB)

holiday_types = CB['holiday-type'].unique()
holiday_types = np.append(holiday_types, 'Other')

regions = CB['region'].unique()
regions = np.append(regions, 'Other')

transportations = CB['transportation'].unique()
transportations = np.append(transportations, 'Other')

months = CB['season'].unique()

accomodations = CB['accomodation'].unique()

hotels = CB['hotel'].unique()
hotels = np.append(hotels, 'Other')

app = customtkinter.CTk()
app.title("Travel Planner")
app.geometry("510x405")


def show_suggested_case_for_evaluation(suggested_case, most_similar_cases, CB, app):
    toplevel = customtkinter.CTkToplevel(app)
    toplevel.title("Travel Planner - Suggestion (Adapted case)")
    toplevel.geometry("510x405")
    toplevel.attributes('-topmost', 'true')

    customtkinter.CTkLabel(toplevel, text=f"Holiday type:", fg_color="transparent").grid(row=0, column=0, padx=10,
                                                                                         pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['holiday-type']}", fg_color="transparent").grid(row=0,
                                                                                                            column=1,
                                                                                                            padx=10,
                                                                                                            pady=5)

    customtkinter.CTkLabel(toplevel, text=f"price:", fg_color="transparent").grid(row=1, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['price']:.2f}", fg_color="transparent").grid(row=1,
                                                                                                         column=1,
                                                                                                         padx=10,
                                                                                                         pady=5)

    customtkinter.CTkLabel(toplevel, text=f"num-persons:", fg_color="transparent").grid(row=2, column=0, padx=10,
                                                                                        pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['num-persons']:.0f}", fg_color="transparent").grid(row=2,
                                                                                                               column=1,
                                                                                                               padx=10,
                                                                                                               pady=5)

    customtkinter.CTkLabel(toplevel, text=f"region:", fg_color="transparent").grid(row=3, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['region']}", fg_color="transparent").grid(row=3, column=1,
                                                                                                      padx=10, pady=5)

    customtkinter.CTkLabel(toplevel, text=f"transportation:", fg_color="transparent").grid(row=4, column=0, padx=10,
                                                                                           pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['transportation']}", fg_color="transparent").grid(row=4,
                                                                                                              column=1,
                                                                                                              padx=10,
                                                                                                              pady=5)

    customtkinter.CTkLabel(toplevel, text=f"duration:", fg_color="transparent").grid(row=5, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['duration']:.0f}", fg_color="transparent").grid(row=5,
                                                                                                            column=1,
                                                                                                            padx=10,
                                                                                                            pady=5)

    customtkinter.CTkLabel(toplevel, text=f"season:", fg_color="transparent").grid(row=6, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['season']}", fg_color="transparent").grid(row=6, column=1,
                                                                                                      padx=10, pady=5)

    customtkinter.CTkLabel(toplevel, text=f"accomodation:", fg_color="transparent").grid(row=7, column=0, padx=10,
                                                                                         pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['accomodation']}", fg_color="transparent").grid(row=7,
                                                                                                            column=1,
                                                                                                            padx=10,
                                                                                                            pady=5)

    customtkinter.CTkLabel(toplevel, text=f"hotel:", fg_color="transparent").grid(row=8, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['hotel']}", fg_color="transparent").grid(row=8, column=1,
                                                                                                     padx=10, pady=5)

    customtkinter.CTkButton(toplevel, text="Accept",
                            command=lambda: btn_accept_callback(suggested_case, most_similar_cases, CB, toplevel)).grid(
        row=9, column=0,
        padx=10, pady=15)
    customtkinter.CTkButton(toplevel, text="Reject",
                            command=lambda: btn_reject_callback(most_similar_cases, CB, toplevel)).grid(row=9,
                                                                                                        column=1,
                                                                                                        padx=10,
                                                                                                        pady=15)


def show_most_similar_case_for_evaluation(CB, new_case, most_similar_cases, app):
    toplevel = customtkinter.CTkToplevel(app)
    toplevel.title("Travel Planner - Suggestion (Most similar case)")
    toplevel.geometry("510x405")
    toplevel.attributes('-topmost', 'true')

    idx = most_similar_cases[0]
    suggested_case = CB.loc[idx]

    customtkinter.CTkLabel(toplevel, text=f"Holiday type:", fg_color="transparent").grid(row=0, column=0, padx=10,
                                                                                         pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['holiday-type']}", fg_color="transparent").grid(row=0,
                                                                                                            column=1,
                                                                                                            padx=10,
                                                                                                            pady=5)

    customtkinter.CTkLabel(toplevel, text=f"price:", fg_color="transparent").grid(row=1, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['price']:.2f}", fg_color="transparent").grid(row=1,
                                                                                                         column=1,
                                                                                                         padx=10,
                                                                                                         pady=5)

    customtkinter.CTkLabel(toplevel, text=f"num-persons:", fg_color="transparent").grid(row=2, column=0, padx=10,
                                                                                        pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['num-persons']:.0f}", fg_color="transparent").grid(row=2,
                                                                                                               column=1,
                                                                                                               padx=10,
                                                                                                               pady=5)

    customtkinter.CTkLabel(toplevel, text=f"region:", fg_color="transparent").grid(row=3, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['region']}", fg_color="transparent").grid(row=3, column=1,
                                                                                                      padx=10, pady=5)

    customtkinter.CTkLabel(toplevel, text=f"transportation:", fg_color="transparent").grid(row=4, column=0, padx=10,
                                                                                           pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['transportation']}", fg_color="transparent").grid(row=4,
                                                                                                              column=1,
                                                                                                              padx=10,
                                                                                                              pady=5)

    customtkinter.CTkLabel(toplevel, text=f"duration:", fg_color="transparent").grid(row=5, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['duration']:.0f}", fg_color="transparent").grid(row=5,
                                                                                                            column=1,
                                                                                                            padx=10,
                                                                                                            pady=5)

    customtkinter.CTkLabel(toplevel, text=f"season:", fg_color="transparent").grid(row=6, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['season']}", fg_color="transparent").grid(row=6, column=1,
                                                                                                      padx=10, pady=5)

    customtkinter.CTkLabel(toplevel, text=f"accomodation:", fg_color="transparent").grid(row=7, column=0, padx=10,
                                                                                         pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['accomodation']}", fg_color="transparent").grid(row=7,
                                                                                                            column=1,
                                                                                                            padx=10,
                                                                                                            pady=5)

    customtkinter.CTkLabel(toplevel, text=f"hotel:", fg_color="transparent").grid(row=8, column=0, padx=10, pady=5)
    customtkinter.CTkLabel(toplevel, text=f"{suggested_case['hotel']}", fg_color="transparent").grid(row=8, column=1,
                                                                                                     padx=10, pady=5)

    customtkinter.CTkButton(toplevel, text="Accept", command=lambda: btn_accept_similar_callback(idx, toplevel)).grid(
        row=9, column=0, padx=10, pady=15)
    customtkinter.CTkButton(toplevel, text="Reject",
                            command=lambda: btn_reject_similar_callback(idx, CB, new_case, most_similar_cases, toplevel,
                                                                        app)).grid(row=9, column=1, padx=10, pady=15)
    return None


def button_callback():
    print('Searching for a trip that matches your preferences...')

    selected_holiday_type = holidaymenu.get()
    if selected_holiday_type == "Other": selected_holiday_type = other_holiday.get()
    if selected_holiday_type == "": selected_holiday_type = None
    print(f'Holiday Type: {selected_holiday_type}')

    selected_price = slider.get()
    if selected_price == 0: selected_price = None
    print(f'Price: {selected_price}')

    selected_num_pers = num_pers.get()
    if selected_num_pers == "":
        selected_num_pers = None
    else:
        selected_num_pers = int(selected_num_pers)
    print(f'Number of persons: {selected_num_pers}')

    selected_region = regionmenu.get()
    if selected_region == "Other": selected_region = other_region.get()
    if selected_region == "": selected_region = None
    print(f'Region: {selected_region}')

    selected_transportation = transportationmenu.get()
    if selected_transportation == "Other": selected_transportation = other_transportation.get()
    if selected_transportation == "": selected_transportation = None
    print(f'Transportation: {selected_transportation}')

    selected_duration = duration.get()
    if selected_duration == "":
        selected_duration = None
    else:
        selected_duration = int(selected_duration)
    print(f'Duration (in days): {selected_duration}')

    selected_month = monthmenu.get()
    print(f'Month: {selected_month}')

    selected_accomodation = accomodationmenu.get()
    print(f'Accomodation: {selected_accomodation}')

    selected_hotel = hotelmenu.get()
    if selected_hotel == "Other": selected_hotel = other_hotel.get()
    if selected_hotel == "": selected_hotel = None
    print(f'Hotel: {selected_hotel}')
    print()

    new_case = pd.Series(
        [selected_holiday_type, selected_price, selected_num_pers, selected_region, selected_transportation,
         selected_duration, selected_month, selected_accomodation, selected_hotel, 0, 0],
        index=["holiday-type", "price", "num-persons", "region", "transportation", "duration", "season", "accomodation",
               "hotel", "num_acceptance", "num_rejected"])

    start_time = time.time()
    most_similar_cases, distances = retrieve(CB, new_case, data_folder, 7)
    suggested_case = weighted_adaptation(new_case, CB.loc[most_similar_cases])
    end_time = time.time()
    # print(f"Most similar case found in {end_time - start_time} seconds with {np.array(distances)[0]} of distance:")
    # show_most_similar_case_for_evaluation(CB, new_case, most_similar_cases, app)

    show_suggested_case_for_evaluation(suggested_case, most_similar_cases, CB, app)
    return None


def btn_accept_similar_callback(index, master):
    # Increase the number of the times the cases used for generate the adaptation are accepted
    CB.loc[index, 'num_acceptance'] += 1
    # Close the popup
    master.destroy()
    master.update()
    return None


def btn_reject_similar_callback(index, CB, new_case, most_similar_cases, toplevel, app):
    # Increase the number of the times the cases used for generate the adaptation are rejected
    CB.loc[index, 'num_rejected'] += 1
    # Close the popup
    toplevel.destroy()
    toplevel.update()
    # Show an adapted case to the user to be validated
    suggested_case = weighted_adaptation(new_case, CB.loc[most_similar_cases])
    show_suggested_case_for_evaluation(suggested_case, most_similar_cases, CB, app)
    return None


def btn_accept_callback(new_case, most_similar_cases, CB, master):
    # Increase the number of the times the cases used for generate the adaptation are accepted
    CB.loc[most_similar_cases, 'num_acceptance'] += 1
    # A new case is added
    Learning.add_new_case(CB, new_case, data_folder)
    #CB = add_new_case(CB, new_case)
    # Feedback to the user
    #show_success_message("New case added", master)
    # Close the popup
    master.destroy()
    master.update()
    return None


def btn_reject_callback(most_similar_cases, CB, master):
    # Increase the number of the times the cases used for generate the adaptation are rejected
    CB.loc[most_similar_cases, 'num_rejected'] += 1
    # Close the popup
    master.destroy()
    master.update()
    return None


def slider_event(value):
    current_price.configure(text=f"{value}")


def holidaymenu_callback(choice):
    if choice == 'Other':
        other_holiday.grid(row=0, column=2, padx=0, pady=5)
        selected_holiday_type = other_holiday.get()
    else:
        other_holiday.grid_forget()


def regionmenu_callback(choice):
    if choice == 'Other':
        other_region.grid(row=3, column=2, padx=0, pady=5)
    else:
        other_region.grid_forget()


def transportationmenu_callback(choice):
    if choice == 'Other':
        other_transportation.grid(row=4, column=2, padx=0, pady=5)
    else:
        other_transportation.grid_forget()


def hotelmenu_callback(choice):
    if choice == 'Other':
        other_hotel.grid(row=8, column=2, padx=0, pady=5)
    else:
        other_hotel.grid_forget()


def show_success_message(msg, master):
    popup = customtkinter.CTkToplevel(master)
    popup.title("Travel Planner - Success message")
    popup.geometry("400x100")
    popup.attributes('-topmost', 'true')
    customtkinter.CTkLabel(popup, text=f"{msg}", fg_color="transparent").grid(row=0, column=0, padx=10, pady=10)
    time.sleep(5)
    return None


def show_error_message(msg, master):
    popup = customtkinter.CTkToplevel(master)
    popup.title("Travel Planner - Error message")
    popup.geometry("400x100")
    popup.attributes('-topmost', 'true')
    customtkinter.CTkLabel(popup, text=f"{msg}", fg_color="transparent").grid(row=0, column=0, padx=10, pady=10)
    time.sleep(5)
    return None


def btn_add_case_callback():
    selected_holiday_type = holidaymenu.get()
    if selected_holiday_type == "Other": selected_holiday_type = other_holiday.get()
    if selected_holiday_type == "": return show_error_message("'Holiday type' is mandatory field", app)
    print(f'Holiday Type: {selected_holiday_type}')

    selected_price = slider.get()
    if selected_price == 0: return show_error_message("'Price' is mandatory field", app)
    print(f'Price: {selected_price}')

    selected_num_pers = num_pers.get()
    if selected_num_pers == "":
        return show_error_message("'Number of persons' is mandatory field", app)
    else:
        selected_num_pers = int(selected_num_pers)
    print(f'Number of persons: {selected_num_pers}')

    selected_region = regionmenu.get()
    if selected_region == "Other": selected_region = other_region.get()
    if selected_region == "": return show_error_message("'Region' is mandatory field", app)
    print(f'Region: {selected_region}')

    selected_transportation = transportationmenu.get()
    if selected_transportation == "Other": selected_transportation = other_transportation.get()
    if selected_transportation == "": return show_error_message("'Mean of transportation' is mandatory field", app)
    print(f'Transportation: {selected_transportation}')

    selected_duration = duration.get()
    if selected_duration == "":
        return show_error_message("'Duration' is mandatory field", app)
    else:
        selected_duration = int(selected_duration)
    print(f'Duration (in days): {selected_duration}')

    selected_month = monthmenu.get()
    print(f'Month: {selected_month}')

    selected_accomodation = accomodationmenu.get()
    print(f'Accomodation: {selected_accomodation}')

    selected_hotel = hotelmenu.get()
    if selected_hotel == "Other": selected_hotel = other_hotel.get()
    if selected_hotel == "": return show_error_message("'Hotel' is mandatory field", app)
    print(f'Hotel: {selected_hotel}')
    print()

    dialog = customtkinter.CTkInputDialog(
        text="If you are an expert and you are sure about adding a new case type 'Expert':",
        title="Travel Planner - Add new case")
    if dialog.get_input() == 'Expert':
        new_case = pd.Series(
            [selected_holiday_type, selected_price, selected_num_pers, selected_region, selected_transportation,
             selected_duration, selected_month, selected_accomodation, selected_hotel, 1, 0],
            index=["holiday-type", "price", "num-persons", "region", "transportation", "duration",
                   "season", "accomodation", "hotel", "num_acceptance", "num_rejected"])
        global CB
        CB = add_new_case(CB, new_case, data_folder, 0)

    return None


# HOLIDAY TYPE
holiday_label = customtkinter.CTkLabel(app, text="Holiday type:", fg_color="transparent")
holiday_label.grid(row=0, column=0, padx=10, pady=10)
holidaymenu = customtkinter.CTkOptionMenu(app, values=holiday_types,
                                          command=holidaymenu_callback)
holidaymenu.grid(row=0, column=1, padx=0, pady=10)
other_holiday = customtkinter.CTkEntry(app, placeholder_text="")

# PRICE
slider = customtkinter.CTkSlider(app, from_=0, to=10000, command=slider_event)
slider.set(0)
price_label = customtkinter.CTkLabel(app, text=f"Price:", fg_color="transparent")
price_label.grid(row=1, column=0, padx=10, pady=5)
slider.grid(row=1, column=1, padx=0, pady=5)
current_price = customtkinter.CTkLabel(app, text=f"0", fg_color="transparent")
current_price.grid(row=1, column=2, padx=0, pady=5)

# NUMBER OF PERSONS
# TODO: Check that this input is a number
num_pers_label = customtkinter.CTkLabel(app, text="Number of persons:", fg_color="transparent")
num_pers_label.grid(row=2, column=0, padx=10, pady=5)
num_pers = customtkinter.CTkEntry(app, placeholder_text="2")
num_pers.grid(row=2, column=1, padx=0, pady=5)

# REGION
region_label = customtkinter.CTkLabel(app, text="Region:", fg_color="transparent")
region_label.grid(row=3, column=0, padx=10, pady=5)
regionmenu = customtkinter.CTkOptionMenu(app, values=regions,
                                         command=regionmenu_callback)
regionmenu.grid(row=3, column=1, padx=0, pady=5)
other_region = customtkinter.CTkEntry(app, placeholder_text="")

# TRANSPORTATION
transportation_label = customtkinter.CTkLabel(app, text="Transportation:", fg_color="transparent")
transportation_label.grid(row=4, column=0, padx=10, pady=5)
transportationmenu = customtkinter.CTkOptionMenu(app, values=transportations,
                                                 command=transportationmenu_callback)
transportationmenu.grid(row=4, column=1, padx=0, pady=5)
other_transportation = customtkinter.CTkEntry(app, placeholder_text="")

# DURATION
duration_label = customtkinter.CTkLabel(app, text="Duration in days:", fg_color="transparent")
duration_label.grid(row=5, column=0, padx=10, pady=5)
duration = customtkinter.CTkEntry(app, placeholder_text="5")
duration.grid(row=5, column=1, padx=0, pady=5)

# MONTH
month_label = customtkinter.CTkLabel(app, text="Month:", fg_color="transparent")
month_label.grid(row=6, column=0, padx=10, pady=5)
monthmenu = customtkinter.CTkOptionMenu(app, values=months)
monthmenu.grid(row=6, column=1, padx=0, pady=5)
month_region = customtkinter.CTkEntry(app, placeholder_text="")

# ACCOMODATION
accomodation_label = customtkinter.CTkLabel(app, text="Accomodation:", fg_color="transparent")
accomodation_label.grid(row=7, column=0, padx=10, pady=5)
accomodationmenu = customtkinter.CTkOptionMenu(app, values=accomodations)
accomodationmenu.grid(row=7, column=1, padx=0, pady=5)

# HOTEL
hotel_label = customtkinter.CTkLabel(app, text="Hotel:", fg_color="transparent")
hotel_label.grid(row=8, column=0, padx=10, pady=5)
hotelmenu = customtkinter.CTkOptionMenu(app, values=hotels,
                                        command=hotelmenu_callback)
hotelmenu.grid(row=8, column=1, padx=0, pady=5)
hotelmenu.set(None)
other_hotel = customtkinter.CTkEntry(app, placeholder_text="")

button = customtkinter.CTkButton(app, text="Find me a trip", command=button_callback)
button.grid(row=9, column=0, padx=10, pady=15)

button = customtkinter.CTkButton(app, text="Add new case", command=btn_add_case_callback)
button.grid(row=9, column=1, padx=10, pady=15)


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit? (The changes in the case base will be stored)"):
        print(CB)
        CB.to_csv(f"{data_folder}/travel.csv", index=False)
        app.destroy()


app.protocol("WM_DELETE_WINDOW", on_closing)

app.mainloop()
