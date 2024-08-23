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


class AIC(ModelEvaluator):
    """Evaluate and compare models with the AIC statistic.

    Akaike's Information Criterion (AIC) is a method for comparing statistical
    models which balances raw goodness-of-fit with model parsimony.  Assuming
    the uncertainties are independent normal random variables the AIC has the
    special form implemented by this class:
    AIC = chi^2 + 2*k
    where chi^2 is the chi-squared statistic, and k is the number of free
    parameters in the model.  This is an asymptotic result for number of
    independent samples n -> infinity.  This is a good approximation for
    n/k <~ 40.

    Lower values of the AIC imply a better model, but note that the value of
    the statistic has no absolute interpretation, and only differences between
    two models with the same observed values (and uncertainties) have meaning.

    For further details see:
    Burnham, K. P. and Anderson, D. R. "Model selection and Multimodel
    Inference: A Practical Information Theoretic Approach." Springer-Verlag,
    2002.
    """

    def __init__(self):
        """ """
        ModelEvaluator.__init__(self, "AIC", False)
        return

    def evaluate(self, fit, count_fixed=False, kshift=0):
        """Return quality of fit for given ModelCluster using AIC (Akaike's Information Criterion).

        Parameters
        ----------
        fit : ModelCluster instance
            The ModelCluster instance to evaluate.
        count_fixed : bool
            Whether fixed parameters are considered. Default is False.
        kshift : int
            Treat the model has having this many additional
            parameters. Negative values also allowed. Default is 0.

        Returns
        -------
        quality : float
            The quality of fit for given ModelCluster."""
        # Number of parameters.  By default, fixed parameters are ignored.
        k = fit.model.npars(count_fixed=count_fixed) + kshift
        if k < 0:
            emsg = "AIC not defined for negative number of parameters."
            raise SrMiseModelEvaluatorError(emsg)

        # Number of data points included in the fit
        n = fit.size

        if n < self.minpoints(k):
            logger.warning("AIC.evaluate(): too few data to evaluate quality reliably.")
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
            The number of parameters in the model.

        Returns
        -------
        int
            The minimum number of points required to make an estimate of a model's quality.
        """

        return 1

    def parpenalty(self, k):
        """Returns the cost for adding k parameters to the current model cluster.

        Parameters
        ----------
        k : int
            The number of added parameters in the model.

        Returns
        -------
        float
            The penalty cost for adding k parameters to the current model cluster.
        """

        # Weight the penalty for additional parameters.
        # If this isn't 1 there had better be a good reason.
        fudgefactor = 1.0

        return (2 * k) * fudgefactor

    def growth_justified(self, fit, k_prime):
        """Returns whether adding k_prime parameters to the given model (ModelCluster) is justified
        given the current quality of the fit.

        The assumption is that adding k_prime parameters will
        result in "effectively 0" chiSquared cost, and so adding it is justified if the cost of adding
        these parameters is less than the current chiSquared cost.
        The validity of this assumption (which depends on an unknown chiSquared value)
        and the impact of the errors used should be examined more thoroughly in the future.

        Parameters
        ----------
        fit : ModelCluster instance
            The ModelCluster instance to evaluate.

        k_prime : int
            The prime number of added parameters in the model.

        Returns
        -------
        bool
            Whether adding k_prime parameters to the given model is justified.
        """

        if self.chisq is None:
            self.chisq = self.chi_squared(fit.value(), fit.y_cluster, fit.error_cluster)

        k_actual = fit.model.npars(count_fixed=False)  # parameters in current fit
        k_test = k_actual + k_prime  # parameters in prospective fit
        n = fit.size  # the number of data points included in the fit

        # If there are too few points to calculate AIC with the requested number of parameter
        # then clearly that increase in parameters is not justified.
        if n < self.minpoints(k_test):
            return False

        # assert n >= self.minPoints(kActual) #check that AIC is defined for the actual fit
        if n < self.minpoints(k_actual):
            logger.warning("AIC.growth_justified(): too few data to evaluate quality reliably.")
            n = self.minpoints(k_actual)

        penalty = self.parpenalty(k_test, n) - self.parpenalty(k_actual, n)

        return penalty < self.chisq

    @staticmethod
    def akaikeweights(aics):
        """Return sequence of Akaike weights for sequence of AICs

        Parameters
        ----------
        aics : array-like
            The sequence of AIC instance.

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
            The sequence of AIC instance.

        Returns
        -------
        array-like
            The sequence of Akaike probabilities"""
        aic_weights = AIC.akaikeweights(aics)
        return aic_weights / np.sum(aic_weights)


# end of class AIC


# simple test code
if __name__ == "__main__":

    m1 = AIC()
    m2 = AIC()

    m1.stat = 20
    m2.stat = 30

    print(m2 > m1)
