#!/usr/bin/env python3

from datasets import get_cat_stay_timeline_2024_df
from collections import Counter, defaultdict
import pandas as pd

CAT_COUNTER = defaultdict(int)


def init_catcounter():
    global CAT_COUNTER

    CAT_COUNTER = defaultdict(int)

    df = get_cat_stay_timeline_2024_df()

    for index, cat_stay in df.iterrows():
        CAT_COUNTER[cat_stay["Arrival_Date"]] += 1

        if pd.isnull(cat_stay["Departure_Date"]):
            continue

        CAT_COUNTER[cat_stay["Departure_Date"] + pd.DateOffset()] -= 1
        # Assuming that the cat is truly absent from the next day

    no_of_cats = 0
    for date in pd.date_range("2024-01-01", "2024-12-31"):
        CAT_COUNTER[date] += no_of_cats
        no_of_cats = CAT_COUNTER[date]


def get_no_of_cats(date):
    assert isinstance(
        date, pd._libs.tslibs.timestamps.Timestamp
    ), "`date` should be a pd._libs.tslibs.timestamps.Timestamp"
    if not CAT_COUNTER:
        init_catcounter()

    return CAT_COUNTER[date]
