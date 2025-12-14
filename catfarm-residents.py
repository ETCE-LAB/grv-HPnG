#!/usr/bin/env python3

from datetime import datetime
import json
import os
import pandas as pd

filename = os.path.join("catfarm-telegram-log.json")


def get_chat_logs() -> dict:
    """
    Return Chat logs as python dict object
    """
    with open(filename, "r") as jsonfile:
        return json.load(jsonfile)


def get_events(chat_messages: list) -> iter:
    """
    Get a list of chat events of a user joining/leaving the room
    """
    for message in chat_messages:
        if message["type"] != "service":
            continue

        if message["action"] in [
            "join_group_by_link",
            "remove_members",
            "invite_members",
        ]:
            yield message


def get_group_count_history(counter_start=1) -> iter:
    """
    Get the group count history of the catfarm groupchat.
    """

    chat_logs = get_chat_logs()
    start_date = datetime.fromtimestamp(int(chat_logs["messages"][0]["date_unixtime"]))

    counter = counter_start

    yield ({"timestamp": start_date, "count": counter_start})

    for event in get_events(chat_logs["messages"]):
        count_delta = 0
        if event["action"] == "join_group_by_link":
            count_delta = 1
        elif event["action"] == "remove_members":
            count_delta = -len(event["members"])
        elif event["action"] == "invite_members":
            count_delta = len(event["members"])

        counter += count_delta
        yield (
            {
                "timestamp": datetime.fromtimestamp(int(event["date_unixtime"])),
                "count": counter,
            }
        )


def get_cat_count_df(counter_start=1) -> pd.DataFrame:
    """
    Return a `pandas.Dataframe` with the log of number of cats at catfarm
    """

    pd.DataFrame.from_records(get_group_count_history(counter_start=counter_start))
