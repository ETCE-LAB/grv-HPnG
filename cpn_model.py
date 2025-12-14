import os
from pprint import pprint

# Import and activate the plugin (and the rest of SNAKES)
import snakes.plugins
import snakes_spn.plugin as spn_plugin
from snakes.nets import *
from snakes.typing import tAll
from collections.abc import Iterable
from collections import defaultdict
from scipy.stats import expon
import pandas as pd

snakes.plugins.load([spn_plugin, "gv"], "snakes.nets", "snk")

import matplotlib.pyplot as plt
import pint
from snk import (
    PetriNet,
    Transition,
    tInteger,
    tFloat,
    Expression,
    Variable,
    Instance,
)
from spn_tools.run_simulation import run_simulation
from datasets import (
    get_dataset_df,
    get_item_df,
    get_unit_prices_df,
    get_item_init_state_df,
)

from rate_function import get_rate_function, get_expenditure_distribution
from no_of_cats import get_no_of_cats
from gaussian_algebra import tGaussianDistribution, GaussianDistribution


GRAPH_FILENAME = os.path.join("/tmp/catfarm_scpn.pdf")


class GDPlace(Place):
    def get_num_tokens(self):
        if len(self.tokens) == 0:
            return 0
        elif len(self.tokens) > 1:
            raise (
                Exception("Token Exception, Place {} has more than 1 token").format(
                    self.name
                )
            )
        else:
            return list(self.tokens)[0]


class pPlace(Place):
    def get_num_tokens(self):
        return len(self.tokens)


def setup_scpn(
    start_date=None,
    order_day=10,
    order_min_delay=5,
    order_exponential_average_delay=5,
    replenishment=defaultdict(int),
):
    if not start_date:
        start_date = pd.to_datetime("2024-03-01")
    else:
        start_date = pd.to_datetime(start_date)

    spn = PetriNet("Catfarm_GD_SPN")

    item_df = get_item_df()
    item_init_state_df = get_item_init_state_df()
    ureg = pint.UnitRegistry()
    resource_places = []

    RATE_FUNCTIONS = {}
    spn.globals["RATE_FUNCTIONS"] = RATE_FUNCTIONS
    spn.globals["REPLENISHMENT"] = replenishment
    transition_label = "Consume Resources"

    consumption_transition = Transition(
        transition_label,
        rate_function=Expression("1"),
    )

    Order_Place = pPlace("Pending Order", [1], tAll)
    spn.add_place(Order_Place)

    order_transition = Transition(
        "Replenish_Resource",
        Expression("time_elapsed>={}".format(order_day + order_min_delay)),
        rate_function=Expression(str(1.0 / order_exponential_average_delay)),
    )

    spn.add_transition(consumption_transition)
    spn.add_transition(order_transition)
    spn.add_input(
        Order_Place.name,
        order_transition.name,
        Variable("order"),
    )

    for index, item in item_df.iterrows():
        # if item["Item_ID"] not in ["Rice", "Raisins"]:
        #     continue
        # Get the initial stock status
        init_stock = item_init_state_df.query("Item_ID=='{}'".format(item["Item_ID"]))[
            "Stock"
        ]
        init_stock = init_stock.iloc[0]
        # Import the smallest unit of the stock item
        try:
            stock_unit = ureg.parse_units(item["Unit"])
            stock_smallest_unit = ureg.parse_units(item["Smallest_Unit"])
        except pint.errors.UndefinedUnitError as E:
            ureg.define("{} = 1".format(item["Unit"]))
            ureg.define("{} = 1".format(item["Smallest_Unit"]))
            stock_unit = ureg.parse_units(item["Unit"])
            stock_smallest_unit = ureg.parse_units(item["Smallest_Unit"])

        # Initialize stock as a pint Object
        init_stock = init_stock * stock_unit

        # Convert stock state to integer into the smallest unit
        init_stock_int = int((init_stock.to(stock_smallest_unit).magnitude))

        place_label = "{} Stock ({})".format(item["Item_ID"], stock_smallest_unit)
        resource_variable = "{}".format(item["Item_ID"].replace(" ", "_"))

        resource_place = GDPlace(
            place_label,
            init_stock_int,
            tAll,
        )

        spn.add_place(resource_place)
        resource_places.append(place_label)

        def item_expenditure_distribution(
            time_elapsed, item_id=item["Item_ID"]
        ) -> GaussianDistribution:
            # Make sure to convert the predicted rate to the smallest unit for the cpn model

            if "delay" not in spn.globals:
                return 0

            gillepsie_delay = spn.globals["delay"]

            no_of_cats = get_no_of_cats(
                start_date + pd.DateOffset(int(time_elapsed + gillepsie_delay))
            )

            if no_of_cats == 0:
                return 0
            item_expenditure = (
                get_expenditure_distribution(item_id, stock_unit)(no_of_cats)
                * gillepsie_delay
            )

            item_expenditure = item_expenditure.to_unit(stock_smallest_unit)

            return item_expenditure

        RATE_FUNCTIONS[item["Item_ID"]] = item_expenditure_distribution

        spn.add_input(
            place_label,
            transition_label,
            Variable(resource_variable),
        )

        spn.add_input(
            place_label,
            order_transition.name,
            Variable(resource_variable),
        )

        rate_function_call = "RATE_FUNCTIONS['{}'](time_elapsed)".format(
            item["Item_ID"]
        )
        rate_function_expression = Expression(
            "{}-{}".format(resource_variable, rate_function_call)
        )

        #
        # It's OK if the resource amount goes negative. It is good to know that
        # we have a resource deficit
        #
        rate_function_expression.globals["RATE_FUNCTIONS"] = RATE_FUNCTIONS

        spn.add_output(
            place_label,
            transition_label,
            rate_function_expression,
        )

        spn.add_output(
            place_label,
            order_transition.name,
            Expression(
                "{} + max(0, {})".format(resource_variable, replenishment[place_label])
            ),
        )
        break

    return spn, resource_places


def draw_graph(spn):
    """
    Print the SPN used for testing graphically to a PDF file.
    """
    spn.draw(GRAPH_FILENAME)


def do_rough_sim():

    spn, resource_places = setup_scpn(
        replenishment=defaultdict(int, {"Raisins Stock (gram)": 0})
    )

    draw_graph(spn)

    sim = run_simulation(spn, max_time=50)

    # plt.set_xlabel("Time (in days)")
    # plt.set_ylabel("Quantity of Resource")

    for resource_place_label in resource_places:
        means = [
            gd.mu if isinstance(gd, GaussianDistribution) else gd
            for gd in sim[resource_place_label]
        ]

        interval_uppers = [
            gd.icdf(0.95) if isinstance(gd, GaussianDistribution) else gd
            for gd in sim[resource_place_label]
        ]

        interval_lowers = [
            gd.icdf(0.05) if isinstance(gd, GaussianDistribution) else gd
            for gd in sim[resource_place_label]
        ]

        no_of_cats = [
            get_no_of_cats(
                pd.to_datetime("2024-03-01") + pd.DateOffset(int(time_elapsed))
            )
            for time_elapsed in sim["time"]
        ]

        plt.plot(sim["time"], means, label=resource_place_label)
        plt.plot(sim["time"], interval_lowers, label=resource_place_label)
        plt.plot(sim["time"], interval_uppers, label=resource_place_label)

        plt.plot(sim["time"], no_of_cats)

        plt.ylim(0, 8000)

        plt.legend()

        plt.show()

    return sim, spn, resource_places
