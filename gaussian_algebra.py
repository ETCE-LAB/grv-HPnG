#!/usr/bin/env python3

from scipy.stats import Normal, Mixture, expon
import matplotlib.pyplot as plt
import numpy as np
from snakes.typing import Instance
from pint import Unit
from numbers import Number


class ExperimentMixture:
    def __init__(self, components, weights=None):
        self.mixture = Mixture(components, weights=weights)

    def do_plot(
        self,
        ax=None,
        fig=None,
        probability_margin=0.95,
        cdf=True,
        invert_label="pdf(Mix)(x) without\noptimal policy",
        invert_color="tab:blue",
        critical_line=True,
        invert=False,
    ):
        x = np.linspace(-3000, 5000, 300)
        x2 = np.linspace(self.mixture.icdf(1 - probability_margin), 5000, 3000)
        (fig, ax) = (fig, ax) if ax is not None else plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel("Resource quantity")
        ax.set_ylabel("Relative Likelihood", color="tab:blue")
        if invert:
            ax.plot(
                x,
                self.mixture.pdf(x),
                color=invert_color,
                linestyle="dotted",
                label=invert_label,
            )
            ax.set_ylabel("Relative Likelihood", color="black")
            return ax, fig
        if cdf is True:
            ax2 = ax.twinx()
            ax2.set_ylabel("Probability", color="tab:orange")
        ax.plot(x, self.mixture.pdf(x), color="tab:blue", label="pdf(Mix)(x)")
        if cdf is True:
            ax2.plot(x, self.mixture.cdf(x), color="tab:orange", label="cdf(Mix)(x)")
        # ax2.annotate(
        #     "({},{})".format(
        #         np.format_float_positional(
        #             self.mixture.icdf(1 - probability_margin), 2
        #         ),
        #         np.format_float_positional(1 - probability_margin, 2),
        #     ),
        #     xy=(self.mixture.icdf(1 - probability_margin), 1 - probability_margin),
        #     xytext=(
        #         self.mixture.icdf(1 - probability_margin)
        #         + abs(self.mixture.icdf(1 - probability_margin)) * 0.07,
        #         1 - probability_margin,
        #     ),
        # )
        ax.fill_between(
            x2,
            # np.zeros_like(x2),
            self.mixture.pdf(x2),
            alpha=0.2,
            label=" α% region",
        )
        if cdf is True:
            ax2.plot(
                [self.mixture.icdf(1 - probability_margin)],
                [1 - probability_margin],
                "o",
                color="tab:orange",
            )
        if critical_line:
            ax2.plot(
                [
                    self.mixture.icdf(1 - probability_margin),
                    self.mixture.icdf(1 - probability_margin),
                ],
                [0, 1 - probability_margin],
                linestyle="dotted",
                color="tab:orange",
            )
            ax.plot(
                [
                    0,
                    self.mixture.icdf(1 - probability_margin),
                ],
                [0, 0],
                linestyle="-",
                marker="|",
                color="black",
                label="icdf(1-α)",
            )
            ax.annotate(
                np.format_float_positional(
                    self.mixture.icdf(1 - probability_margin), 2
                ),
                xy=(self.mixture.icdf(1 - probability_margin) / 2, 0),
                xytext=(self.mixture.icdf(1 - probability_margin) / 2, 0.00005),
                ha="center",
            )
        else:
            ax.annotate(
                np.format_float_positional(
                    self.mixture.icdf(1 - probability_margin), 2
                ),
                xy=(self.mixture.icdf(1 - probability_margin), 0),
                xytext=(500, 0.00005),
                arrowprops=dict(facecolor="black", arrowstyle="->", alpha=0.5),
            )
        return ax, fig

    def do_plot_and_show(
        self,
        ax=None,
        fig=None,
        probability_margin=0.95,
        cdf=True,
        invert_label="pdf(Mix)(x) without\noptimal policy",
        invert_color="tab:blue",
        invert=False,
        critical_line=True,
    ):
        (ax, fig) = self.do_plot(
            ax=ax,
            fig=fig,
            probability_margin=probability_margin,
            cdf=cdf,
            invert_label=invert_label,
            invert_color=invert_color,
            invert=invert,
            critical_line=critical_line,
        )
        if fig:
            fig.legend()
            fig.show()
        return ax, fig


class GaussianDistribution(Normal):

    def __init__(self, *args, unit: Unit, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self.unit = unit

    def do_plot(self, ax=None):
        ax = self.plot() if ax is None else ax
        sample = self.sample(10000)
        ax.hist(sample, density=True, bins=50, alpha=0.5)
        return ax

    def do_plot_and_show(self, ax=None):
        ax = self.do_plot(ax=ax)
        plt.show()
        return ax

    def __add__(self, something) -> "GaussianDistribution":
        if isinstance(something, Normal):
            return GaussianDistribution(
                unit=self.unit,
                mu=(self.mu + something.mu),
                sigma=((self.sigma**2 + something.sigma**2) ** 0.5),
            )
        elif isinstance(something, Number):
            return GaussianDistribution(
                unit=self.unit, mu=(self.mu + something), sigma=self.sigma
            )

    def __sub__(self, something) -> "GaussianDistribution":
        if isinstance(something, Normal):
            return GaussianDistribution(
                unit=self.unit,
                mu=(self.mu - something.mu),
                sigma=((self.sigma**2 + something.sigma**2) ** 0.5),
            )
        elif isinstance(something, Number):
            return GaussianDistribution(
                unit=self.unit, mu=(self.mu - something), sigma=self.sigma
            )

    def __mul__(self, something) -> "GaussianDistribution":
        if isinstance(something, Number):
            return GaussianDistribution(
                unit=self.unit,
                mu=(self.mu * something),
                sigma=((self.sigma**2) * (something**2)) ** 0.5,
            )

    def __truediv__(self, something) -> "GaussianDistribution":
        if isinstance(something, Number):
            something = 1 / something
            return GaussianDistribution(
                unit=self.unit,
                mu=(self.mu * something),
                sigma=((self.sigma**2) * (something**2)) ** 0.5,
            )

    def to_unit(self, unit: Unit) -> "GaussianDistribution":

        mu = (self.mu * self.unit).to(unit).magnitude
        sigma = (self.sigma * self.unit).to(unit).magnitude

        return GaussianDistribution(unit=unit, mu=mu, sigma=sigma)


tGaussianDistribution = Instance(GaussianDistribution)


def get_test_mixture():
    import pint

    ureg = pint.UnitRegistry()
    return ExperimentMixture(
        [
            GaussianDistribution(mu=-80, sigma=200, unit=ureg.kilogram),
            GaussianDistribution(mu=-1300, sigma=200, unit=ureg.kilogram),
            GaussianDistribution(mu=1000, sigma=200, unit=ureg.kilogram),
        ]
    )
