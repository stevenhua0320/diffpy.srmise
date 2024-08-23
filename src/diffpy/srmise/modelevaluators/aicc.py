#!/usr/bin/env python
##############################################################################
#
# SrMise            by Luke Granlund
#                   (c) 2014 trustees of the Michigan State University
#                   (c) 2024 trustees of Columbia University in the City of New York
#                   All rights reserved.
#
# File coded by:    Luke Granlund
#
# See LICENSE.txt for license information.
#
##############################################################################

import logging

import numpy as np

from diffpy.srmise.modelevaluators.base import ModelEvaluator
from diffpy.srmise.srmiseerrors import SrMiseModelEvaluatorError

logger = logging.getLogger("diffpy.srmise")


class AICc(ModelEvaluator):
    """Evaluate and compare models with the AICc statistic.

    Akaike's Information Criterion w/ 2nd order correction for small sample
    sizes (AICc) is a method for comparing statistical models which balances
    raw goodness-of-fit with model parsimony.  Assuming the uncertainties are
    independent normal random variables the AICc has the special form
    implemented by this class:
    AICc = chi^2 + 2*k + 2*k*(k+1)/(n-k-1)
    where chi^2 is the chi-squared statistic, k is the number of free
    parameters in the model, and n is the number of data points.

    Lower values of the AICc imply a better model, but note that the value of
    the statistic has no absolute interpretation, and only differences between
    two models with the same observed values (and uncertainties) have meaning.

    For further details see:
    Burnham, K. P. and Anderson, D. R. "Model selection and Multimodel
    Inference: A Practical Information Theoretic Approach." Springer-Verlag,
    2002.
    """

    def __init__(self):
        """ """
        ModelEvaluator.__init__(self, "AICc", False)
        return

    def evaluate(self, fit, count_fixed=False, kshift=0):
        """Return quality of fit for given ModelCluster using AICc (Akaike's Information Criterion
        with 2nd order correction for small sample size).

        Parameters
        fit: A ModelCluster
            The ModelCluster to evaluate.
        count_fixed : bool
            Whether fixed parameters are considered. Default is False.
        kshift : int
            Treat the model has having this many additional
            parameters.  Negative values also allowed. Default is 0.

        Returns
        -------
        float
            Quality of AICc"""
        # Number of parameters.  By default, fixed parameters are ignored.
        k = fit.model.npars(count_fixed=count_fixed) + kshift
        if k < 0:
            emsg = "AICc not defined for negative number of parameters."
            raise SrMiseModelEvaluatorError(emsg)

        # Number of data points included in the fit
        n = fit.size

        if n < self.minpoints(k):
            logger.warning("AICc.evaluate(): too few data to evaluate quality reliably.")
            n = self.minpoints(k)

        if self.chisq is None:
            self.chisq = self.chi_squared(fit.value(), fit.y_cluster, fit.error_cluster)

        self.stat = self.chisq + self.parpenalty(k, n)

        return self.stat

    def minpoints(self, npars):
        """Calculates the minimum number of points required to make an estimate of a model's quality.

        Parameters
        ----------
        npars : int
            The number of points required to make an estimate of a model's quality.

        Returns
        -------
        int
            The minimum number of points required to make an estimate of a model's quality.
        """

        # From the denominator of AICc, it is clear that the first positive finite contribution to
        # parameter cost is at n>=k+2
        return npars + 2

    def parpenalty(self, k, n):
        """Returns the cost for adding k parameters to the current model cluster.

        Parameters
        ----------
        k : int
            The number of parameters to add.

        n : int
            The number of data points.

        Returns
        -------
        float
            The cost for adding k parameters to the current model cluster.
        """

        # Weight the penalty for additional parameters.
        # If this isn't 1 there had better be a good reason.
        fudgefactor = 1.0

        return (2 * k + float(2 * k * (k + 1)) / (n - k - 1)) * fudgefactor

    def growth_justified(self, fit, k_prime):
        """Is adding k_prime parameters to ModelCluster justified given the current quality of the fit.

        The assumption is that adding k_prime parameters will result in "effectively 0" chiSquared cost,
        and so adding it is justified if the cost of adding these parameters is less than the current
        chiSquared cost.  The validity of this assumption (which depends on an unknown chiSquared value)
        and the impact of the errors used should be examined more thoroughly in the future.

        Parameters
        ----------
        fit : ModelCluster
            The ModelCluster to evaluate.
        k_prime : int
            The prime number of parameters to add.

        Returns
        -------
        bool
            Whether the current model cluster is justified or not.
        """

        if self.chisq is None:
            self.chisq = self.chi_squared(fit.value(), fit.y_cluster, fit.error_cluster)

        k_actual = fit.model.npars(count_fixed=False)  # parameters in current fit
        k_test = k_actual + k_prime  # parameters in prospective fit
        n = fit.size  # the number of data points included in the fit

        # If there are too few points to calculate AICc with the requested number of parameter
        # then clearly that increase in parameters is not justified.
        if n < self.minpoints(k_test):
            return False

        # assert n >= self.minPoints(kActual) #check that AICc is defined for the actual fit
        if n < self.minpoints(k_actual):
            logger.warning("AICc.growth_justified(): too few data to evaluate quality reliably.")
            n = self.minpoints(k_actual)

        penalty = self.parpenalty(k_test, n) - self.parpenalty(k_actual, n)

        return penalty < self.chisq

    @staticmethod
    def akaikeweights(aics):
        """Return sequence of Akaike weights for sequence of AICs

        Parameters
        ----------
        aics : array-like
            The squence of AIC instances

        Returns
        -------
        array-like
            The sequence of Akaike weights
        """

        aic_stats = np.array([aic.stat for aic in aics])
        aic_min = min(aic_stats)
        return np.exp(-(aic_stats - aic_min) / 2.0)

    @staticmethod
    def akaikeprobs(aics):
        """Return sequence of Akaike probabilities for sequence of AICs

        Parameters
        ----------
        aics : array-like
            The squence of AIC instances

        Returns
        -------
        array-like
            The sequence of Akaike probabilities"""
        aic_weights = AICc.akaikeweights(aics)
        return aic_weights / np.sum(aic_weights)


# end of class AICc


# simple test code
if __name__ == "__main__":

    m1 = AICc()
    m2 = AICc()

    m1.stat = 20
    m2.stat = 30

    print(m2 > m1)
