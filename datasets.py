#!/usr/bin/env python3

import pandas as pd
import datetime


def get_dataset_df():
    df = pd.read_csv("./datasets/kitchen_resource_consumption/dataset.csv")
    df["Daily_Consumption"] = df["Quantity_Consumed"] / df["Days"]
    return df


def get_unit_prices_df():
    return pd.read_csv(
        "./datasets/kitchen_resource_consumption/Unit_Prices_Database.csv"
    )


def get_item_df():
    return pd.read_csv("./datasets/kitchen_resource_consumption/item_database.csv")


def get_item_init_state_df():
    return pd.read_csv("./datasets/kitchen_resource_consumption/item_init_state.csv")


def get_cat_stay_timeline_2024_df():
    df = pd.read_csv("./datasets/kitchen_resource_consumption/cats_timeline_2024.csv")
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], format="%d/%m/%Y")
    df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], format="%d/%m/%Y")
    return df


# ureg = UnitRegistry()

# for index, item in item_df.iterrows():
#     try:
#         print(
#             ureg.parse_units(item["Unit"]),
#             ureg.parse_units(item["Smallest_Unit"]),
#         )

#     except errors.UndefinedUnitError as E:
#         print(E)
#         units = ureg.define("{} = 1".format(item["Unit"]))
#         print(
#             ureg.parse_units(item["Unit"]),
#             ureg.parse_units(item["Smallest_Unit"]),
#         )
