#!/usr/bin/env python3

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    RationalQuadratic,
)

from matplotlib import pyplot as plt
import numpy as np

from datasets import (
    get_dataset_df,
    get_item_df,
    get_item_init_state_df,
    get_unit_prices_df,
)
from pint import Unit
from gaussian_algebra import GaussianDistribution

GP_MODEL = {}


def get_unit_dataset_df(item_id):
    dataset_df = get_dataset_df()
    return dataset_df[dataset_df["Item_ID"] == item_id]


def train_gp_model_for_unit(item_id, kernel=None):
    """
    Train a gaussian process model for `item_id`
    """

    unit_dataset_df = get_unit_dataset_df(item_id)

    Y_train = unit_dataset_df["Daily_Consumption"].to_numpy().reshape(-1, 1)
    X_train = unit_dataset_df["No_of_Cats"].to_numpy().reshape(-1, 1)

    if not kernel:
        linear_kernel = ConstantKernel() * DotProduct()
        kernel = linear_kernel + RationalQuadratic(length_scale=1.0, alpha=0.1)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
    gaussian_process.fit(X_train, Y_train)

    return gaussian_process, X_train, Y_train


def predict_using_gp(gaussian_process, X_test):
    mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)
    return mean_prediction, std_prediction


def plot_gp_model_for_unit(item_id, kernel=None):
    gaussian_process, X, Y = train_gp_model_for_unit(item_id, kernel=kernel)
    X_test = np.arange(1, 40).reshape(-1, 1)
    mean_prediction, std_prediction = predict_using_gp(gaussian_process, X_test)

    plt.plot(X, Y, label="Daily Consumption of {}".format(item_id), linestyle="dotted")
    plt.scatter(X, Y, label="Observations")
    plt.plot(X_test, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X_test.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label="95% confidence interval",
    )

    plt.legend()
    plt.xlabel("No of Cats")
    plt.xlabel("Daily Consumption of {}".format(item_id))
    _ = plt.title(
        "Gaussian Process regression for {}; {}".format(
            item_id, str(gaussian_process.kernel_)
        )
    )
    plt.show()


def plot_multiple_gp_models_for_unit(item_id):

    linear_kernel = ConstantKernel(0.1, (1.0, 10.0)) * DotProduct(
        sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)
    )
    kernel1 = linear_kernel + RationalQuadratic(length_scale=10.0, alpha=0.1)
    kernel2 = linear_kernel + RationalQuadratic(length_scale=1.0, alpha=0.1)

    gp1, X, Y = train_gp_model_for_unit(item_id, kernel=kernel1)
    gp2, _, _ = train_gp_model_for_unit(item_id, kernel=kernel2)
    X_test = np.arange(1, 40).reshape(-1, 1)
    mean_1, std_1 = predict_using_gp(gp1, X_test)
    mean_2, std_2 = predict_using_gp(gp2, X_test)

    plt.plot(X, Y, label="Daily Consumption of {}".format(item_id), linestyle="dotted")
    plt.scatter(X, Y, label="Observations")
    plt.plot(X_test, mean_1, label="Mean prediction 1")
    plt.fill_between(
        X_test.ravel(),
        mean_1 - 1.96 * std_1,
        mean_1 + 1.96 * std_1,
        alpha=0.5,
        label="95% confidence interval, mean_1",
    )

    plt.plot(X_test, mean_2, label="Mean prediction 2")
    plt.fill_between(
        X_test.ravel(),
        mean_2 - 1.96 * std_2,
        mean_2 + 1.96 * std_2,
        alpha=0.5,
        label="95% confidence interval, mean_2",
    )

    plt.legend()
    plt.xlabel("No of Cats")
    plt.xlabel("Daily Consumption of {}".format(item_id))
    _ = plt.title(
        "Gaussian Process regression for {}; {} and {}".format(
            item_id, str(gp1.kernel_), str(gp2.kernel_)
        )
    )
    plt.show()


def get_rate_function(item_id):
    global GP_MODEL
    gp_model, _, _ = (
        (GP_MODEL[item_id], None, None)
        if item_id in GP_MODEL
        else train_gp_model_for_unit(item_id)
    )

    GP_MODEL[item_id] = gp_model

    def rate_function(no_of_cats):
        (mean,), (std,) = predict_using_gp(
            gp_model, np.array([no_of_cats]).reshape(-1, 1)
        )
        return np.random.normal(loc=mean, scale=std)

    return rate_function


def get_expenditure_distribution(item_id, unit: Unit):
    global GP_MODEL
    gp_model, _, _ = (
        (GP_MODEL[item_id], None, None)
        if item_id in GP_MODEL
        else train_gp_model_for_unit(item_id)
    )

    GP_MODEL[item_id] = gp_model

    def expenditure_distribution(no_of_cats):
        (mean,), (std,) = predict_using_gp(
            gp_model, np.array([no_of_cats]).reshape(-1, 1)
        )
        return GaussianDistribution(unit=unit, mu=mean, sigma=std)

    return expenditure_distribution
