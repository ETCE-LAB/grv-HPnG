#!/usr/bin/env python3

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import collections
import math

from cpn_model import setup_scpn
from spn_tools.run_simulation import (
    run_repeated_experiment,
    run_repeated_experiment_and_converge,
    aggregate_dataset_in_timeboxes,
    plot_results,
    run_simulation,
)

from no_of_cats import get_no_of_cats
import pandas as pd
from gaussian_algebra import ExperimentMixture, GaussianDistribution


# def get_resource_requirements_ncats(
#     start_date=None,
#     no_of_days=40,
#     no_of_cats=23,
#     num_reps=5,
#     probability=0.95,
#     plot=False,
# ):

#     if start_date is None:
#         start_date = "2024-03-01"

#     catfarm_scpn, resource_places = setup_scpn(start_date=start_date)

#     experiment = run_repeated_experiment(
#         num_reps=num_reps, spn=catfarm_scpn, max_time=no_of_days, verbose=True
#     )

#     aggregate_dataset = (y_data, timebox_duration) = aggregate_dataset_in_timeboxes(
#         experiment, "time", y_vars=resource_places, num_timeboxes=no_of_days
#     )

#     resource_supply_levels = {}
#     print("Computing {}% Confidence Intervals".format(int(confidence * 100)))

#     for resource_name in resource_places:
#         means = np.mean(y_data[resource_name], axis=0)
#         stes = st.sem(y_data[resource_name], axis=0)
#         confidence_intervals = (
#             st.t.interval(confidence=confidence, df=num_reps - 1, loc=means, scale=stes)
#             if num_reps < 30
#             else st.norm.interval(confidence=confidence, loc=means, scale=stes)
#         )

#         resource_supply_levels[resource_name] = {
#             "mean": means,
#             "ste": stes,
#             "confidence_interval": np.array(confidence_intervals).T,
#         }

#     if plot is True:
#         fig, ax = plt.subplots(nrows=1, ncols=1)
#         plot_results(
#             experiment,
#             "time",
#             resource_places,
#             no_of_days,
#             interval_type="confidence",
#             ax=ax,
#         )
#         ax.axline((0, 0), (10, 0), color="r", linestyle="dotted")
#         fig.show()

#     return experiment, aggregate_dataset, resource_supply_levels


def cart(mix1, mix2):

    # (x1, y1), (x2, y2) = (mix1.icdf(0.025), mix1.icdf(1 - 0.025)), (
    #     mix2.icdf(0.025),
    #     mix2.icdf(1 - 0.025),
    # )

    # return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    return abs(mix1.mean() - mix2.mean())


def met(conv1, conv2):
    return sum(cart(conv1[key], conv2[key]) for key in conv1.keys()) / len(conv1)


def plot_run(run, resource_places):
    for resource_place_label in resource_places:
        means = [
            gd.mean() if isinstance(gd, GaussianDistribution) else gd
            for gd in run[resource_place_label]
        ]
        interval_uppers = [
            gd.icdf(0.95) if isinstance(gd, GaussianDistribution) else gd
            for gd in run[resource_place_label]
        ]

        interval_lowers = [
            gd.icdf(0.05) if isinstance(gd, GaussianDistribution) else gd
            for gd in run[resource_place_label]
        ]
        plt.plot(run["time"], means, label=resource_place_label)
        plt.plot(run["time"], interval_lowers, label=resource_place_label)
        plt.plot(run["time"], interval_uppers, label=resource_place_label)

    plt.legend()

    plt.show()


def experiement_convergence(conv, sim):
    if conv is None:
        conv = {
            key: st.Mixture([sim[key][-1]])
            for key in sim.keys()
            if key not in ["time", "Pending Order"]
        }
        return conv

    new_conv = {
        key: st.Mixture((conv[key].components + [sim[key][-1]]))
        for key in sim.keys()
        if key not in ["time", "Pending Order"]
    }
    return new_conv


def get_resource_requirements(
    start_date=None,
    order_day=10,
    no_of_days=50,
    num_reps=10,
    probability_margin=0.95,
    plot=True,
    scpn=None,
    resource_places=None,
    ax2_label="No of residents",
):

    if start_date is None:
        start_date = "2024-03-01"

    if scpn is None:
        catfarm_scpn, resource_places = setup_scpn(
            start_date=start_date, order_day=order_day
        )
    else:
        catfarm_scpn = scpn

    experiment = run_repeated_experiment_and_converge(
        num_reps=num_reps,
        spn=catfarm_scpn,
        max_time=no_of_days,
        verbose=True,
        converge=experiement_convergence,
    )

    aggregate_dataset = (y_data, timebox_duration) = aggregate_dataset_in_timeboxes(
        experiment, "time", y_vars=resource_places, num_timeboxes=no_of_days
    )

    resource_mixture_distributions = {}
    print("Computing Mixture Distributions")
    # print("Computing {}% Probability Intervals".format(int(probability_margin * 100)))

    for resource_name in resource_places:

        resource_mixture_distributions[resource_name] = [
            ExperimentMixture([z for z in x if isinstance(z, GaussianDistribution)])
            for x in zip(*(y_data[resource_name]))
        ]

    fig, ax = None, None

    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        timebox_endtimes = [i * timebox_duration for i in range(1, 1 + no_of_days)]
        for resource_name in resource_places:
            means = []
            interval_uppers = []
            interval_lowers = []

            for exp in resource_mixture_distributions[resource_name]:
                means.append(exp.mixture.mean())
                interval_uppers.append(
                    exp.mixture.icdf(1 - (1 - probability_margin) / 2)
                )
                interval_lowers.append(exp.mixture.icdf((1 - probability_margin) / 2))

            ax.plot(timebox_endtimes, means, label="mean R^j Stock")
            ax.set_ylabel("Resource quantity")
            ax.set_xlabel("Days")
            ax.fill_between(
                timebox_endtimes,
                interval_lowers,
                interval_uppers,
                alpha=0.2,
                label="α% region\naround mean",
            )
        # ax.plot([0, no_of_days + 10], [0, 0])
        ax2 = ax.twinx()
        ax2.set_ylim((0, 30))
        ax.set_ylim((-10, 3000))
        ax2.set_ylabel(ax2_label)
        ax2.plot(
            np.arange(50),
            list(
                get_no_of_cats(pd.to_datetime(start_date) + pd.DateOffset(i))
                for i in range(50)
            ),
            color="grey",
            label="No of\nresidents",
            alpha=0.7,
        )

        fig.show()

    return experiment, aggregate_dataset, resource_mixture_distributions, (fig, ax)


def get_order_arrival_day(run):
    # plot_run(run, resource_names)
    experiment_length = len(run["time"])
    for i in range(experiment_length):
        if run["Pending Order"][i] == 0:
            return int(run["time"][i])

    return np.inf


def get_runout_day(run, resource_names, probability_margin=0.95):
    # plot_run(run, resource_names)
    experiment_length = len(run["time"])
    for i in range(experiment_length):

        if any(
            (
                (0 if resource_supply_level > 0 else 1)
                if isinstance(resource_supply_level, int)
                else resource_supply_level.cdf(0)
            )
            >= (1 - probability_margin)  # (1-X) probability to runout
            for resource_supply_level in [
                run[resource][i] for resource in resource_names
            ]
        ):
            return int(run["time"][i])

    return np.inf


def predict_runout_and_replenishment(
    experiment,
    resource_mixture_distributions,
    probability_margin=0.95,
    order_day=10,
    fig=None,
    ax=None,
    plot=False,
):
    """Return the probability interval of the first day when *any* resource runs
    out completely

    i.e. Say with X% probability that resources will last between n1 and n2
    days. n1, n2 <=no_of_days_to_simulate

    """

    resource_replenishment = {
        resource_name: math.floor(
            -(exp_mixtures[-1].mixture.icdf(1 - probability_margin))
        )
        for resource_name, exp_mixtures in resource_mixture_distributions.items()
    }

    resource_names = list(experiment[0].keys())
    resource_names.remove("time")
    resource_names.remove("Pending Order")

    no_of_runs = len(experiment)
    runout_days = np.array(
        [
            get_runout_day(run, resource_names, probability_margin=probability_margin)
            for run in experiment.values()
        ]
    )  # The number of days it took to be in a state of resource
    # scarcity (runout) in each experiment run
    runout_frequencies = collections.Counter(runout_days)
    runout_probabilities = np.array(list(runout_frequencies.values())) / len(
        runout_days
    )
    runout_distribution = st.rv_discrete(
        values=(list(runout_frequencies.keys()), runout_probabilities)
    )
    runout_day = runout_distribution.ppf(1 - probability_margin)

    order_arrival_days = np.array(
        [get_order_arrival_day(run) for run in experiment.values()]
    )
    order_arrival_day_frequencies = collections.Counter(order_arrival_days)
    order_arrival_day_probabilities = np.array(
        list(order_arrival_day_frequencies.values())
    ) / len(order_arrival_days)
    order_arrival_day_distribution = st.rv_discrete(
        values=(
            list(order_arrival_day_frequencies.keys()),
            order_arrival_day_probabilities,
        )
    )
    order_arrival_day = order_arrival_day_distribution.ppf(probability_margin)

    if plot is True:
        # ax.axline(
        #     (runout_day, 0),
        #     (runout_day, 100),
        #     color="tab:red",
        #     linestyle="dotted",
        #     label="Runout day",
        # )

        for rd, freq in runout_frequencies.items():
            ax.plot(
                [rd, rd],
                [10, freq * 100],
                color="tab:red",
                linewidth=2,
                alpha=0.4,
            )
            # ax.annotate(
            #     freq,
            #     xy=(rd, freq * 100),
            #     xytext=(rd, freq * 100 + 50),
            #     ha="center",
            # )

        ax.plot(
            [-10],
            [100],
            color="tab:red",
            linewidth=4,
            alpha=0.4,
            label="Runout frequency",
        )

    return (
        (order_arrival_day, order_arrival_days, order_arrival_day_distribution),
        (runout_day, runout_days, runout_distribution),
        resource_replenishment,
    )


def run_replenishment_simulation(probability_margin=0.95, num_reps=50):

    initial_order_day = 5

    (experiment, aggregate_dataset, resource_mixture_distributions, (fig, ax)) = (
        get_resource_requirements(
            order_day=initial_order_day,
            no_of_days=50,
            num_reps=num_reps,
            probability_margin=probability_margin,
            plot=True,
            ax2_label="No of Residents/\nRunout frequency",
        )
    )
    fig.show()
    for resource_name, exp_mixtures in resource_mixture_distributions.items():
        fig_mm, ax_mm = plt.subplots(
            nrows=1,
            ncols=1,
        )
        resource_mixture_distributions[resource_name][-1].do_plot_and_show(
            ax=ax_mm,
            fig=fig_mm,
            probability_margin=probability_margin,
            invert=False,
        )

    (
        (order_arrival_day, order_arrival_days, order_arrival_day_distribution),
        (runout_day, runout_days, runout_distribution),
        resource_replenishment,
    ) = predict_runout_and_replenishment(
        experiment,
        resource_mixture_distributions=resource_mixture_distributions,
        probability_margin=probability_margin,
        order_day=initial_order_day,
        fig=fig,
        ax=ax,
        plot=True,
    )

    fig.legend()
    fig.show()

    order_delay = order_arrival_day - initial_order_day

    hindsighted_order_day = runout_day - order_delay

    new_spn, resource_places = setup_scpn(
        replenishment=resource_replenishment,
        order_day=hindsighted_order_day,
    )

    (
        new_experiment,
        new_aggregate_dataset,
        new_resource_mixture_distributions,
        (fig, ax),
    ) = get_resource_requirements(
        order_day=initial_order_day,
        no_of_days=50,
        num_reps=num_reps,
        probability_margin=probability_margin,
        plot=True,
        ax2_label="No of residents/\nOrder arrival frequency",
        scpn=new_spn,
        resource_places=resource_places,
    )
    new_order_arrival_days = np.array(
        [get_order_arrival_day(run) for run in new_experiment.values()]
    )
    new_order_arrival_day_frequencies = collections.Counter(new_order_arrival_days)

    for od, freq in new_order_arrival_day_frequencies.items():
        ax.plot(
            [od, od],
            [10, freq * 100],
            color="tab:green",
            linewidth=2,
            alpha=0.4,
        )
        # ax.annotate(
        #     freq,
        #     xy=(od, freq * 100),
        #     xytext=(od, freq * 100 + 50),
        #     ha="center",
        # )

    ax.plot(
        [-10],
        [100],
        color="tab:green",
        linewidth=4,
        alpha=0.4,
        label="Order arrival\nfrequency",
    )
    fig.legend()
    fig.show()

    for resource_name, exp_mixtures in new_resource_mixture_distributions.items():
        ax_mm, fig_mm = exp_mixtures[-1].do_plot(
            invert=False,
            cdf=False,
            probability_margin=probability_margin,
            critical_line=False,
        )
        ax_mm, fig_mm = resource_mixture_distributions[resource_name][
            -1
        ].do_plot_and_show(
            ax=ax_mm, fig=fig_mm, invert=True, probability_margin=probability_margin
        )
        # exp_mixtures[int(runout_day)].do_plot_and_show(
        #     ax=ax_mm,
        #     fig=fig_mm,
        #     invert_label="pdf(Mix)(x) on runout day",
        #     invert_color="tab:grey",
        #     invert=True,
        #     probability_margin=probability_margin,
        # )

    return (
        (order_arrival_day, order_arrival_days, order_arrival_day_distribution),
        (runout_day, runout_days, runout_distribution),
        resource_replenishment,
    )


run_replenishment_simulation(num_reps=50, probability_margin=0.95)
