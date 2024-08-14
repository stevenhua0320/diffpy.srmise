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

from diffpy.srmise.baselines.base import BaselineFunction
from diffpy.srmise.baselines.polynomial import Polynomial
from diffpy.srmise.srmiseerrors import SrMiseEstimationError

logger = logging.getLogger("diffpy.srmise")


class Arbitrary(BaselineFunction):
    """Methods for evaluating a baseline from an arbitrary function.

    Supports baseline calculations with arbitrary functions.  These functions,
    if implemented, must have the following signatures and return values:
    valuef(pars, x) ==> numpy.array of length x if x is a sequence
                  ==> number if x is a number
    jacobianf(pars, x, free) ==> list, each element a numpy.array of length x if
                                x is a sequence or None if value of free for
                                that parameter is False.
                            ==> list, each element a number if x is a number
                                or None if value of free for that parameter is
                                False
    estimatef(x, y) ==> numpy.array of length npars
    """

    def __init__(self, npars, valuef, jacobianf=None, estimatef=None, Cache=None):
        """Initialize an arbitrary baseline.

        Parameters
        ----------
        npars : int
            The number of parameters which define the function
        valuef : array-like or int
            The function which calculates the value of the baseline at x.
        jacobianf : array-like or None
            The function which calculates the Jacobian of the
                  baseline function with respect to free pars.
        estimatef : array-like or None
            The function which estimates function parameters given the data x and y.
        Cache :  None or callable
            The class (not instance) which implements caching of BaseFunction evaluations.
        """
        # Guarantee valid number of parameters
        try:
            testnpars = int(str(npars))
        except ValueError:
            emsg = "Argument degree must be a non-negative integer."
            raise ValueError(emsg)
        if testnpars < 0:
            emsg = "Argument degree must be a non-negative integer."
            raise ValueError(emsg)
        # Define parameterdict
        # e.g. {"a_0":0, "a_1":1, "a_2":2, "a_3":3} if npars is 4.
        parameterdict = {}
        for d in range(testnpars + 1):
            parameterdict["a_" + str(d)] = d
        formats = ["internal"]
        default_formats = {"default_input": "internal", "default_output": "internal"}

        # Check that the provided functions are at least callable
        if valuef is None or callable(valuef):
            self.valuef = valuef
        else:
            emsg = "Specified value function is not callable."
            raise ValueError(emsg)
        if jacobianf is None or callable(jacobianf):
            self.jacobianf = jacobianf
        else:
            emsg = "Specified jacobian function is not callable."
            raise ValueError(emsg)
        if estimatef is None or callable(estimatef):
            self.estimatef = estimatef
        else:
            emsg = "Specified estimate function is not callable."
            raise ValueError(emsg)

        # TODO: figure out how the metadict can be used to save the functions
        # and use them again when a file is loaded...
        metadict = {
            "npars": (npars, repr),
            "valuef": (valuef, repr),
            "jacobianf": (jacobianf, repr),
            "estimatef": (estimatef, repr),
        }
        BaselineFunction.__init__(self, parameterdict, formats, default_formats, metadict, None, Cache)

    # Methods required by BaselineFunction ####

    def estimate_parameters(self, r, y):
        """Estimate parameters for data baseline.

        Parameters
        ----------
        r : array-like
            The data along r from which to estimate
        y : array-like
            The data along y from which to estimate

        Returns
        -------
        The numpy array of parameters in the default internal format.

        we raise NotImplementedError if no estimation routine is defined, and
        SrMiseEstimationError if parameters cannot be estimated for any other."""
        if self.estimatef is None:
            emsg = "No estimation routine provided to Arbitrary."
            raise NotImplementedError(emsg)

        # TODO: check that estimatef returns something proper?
        try:
            return self.estimatef(r, y)
        except Exception as e:
            emsg = "Error within estimation routine provided to Arbitrary:\n" + str(e)
            raise SrMiseEstimationError(emsg)

    def _jacobianraw(self, pars, r, free):
        """Return the Jacobian of a polynomial.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters
                pars[0] = a_0
                pars[1] = a_1
                ...
        r : array-like or int
            The sequence or scalar over which pars is evaluated
        free : array-like of bools
            The sequence of booleans which determines which derivatives are needed.
            True for evaluation, False for no evaluation.

        Returns
        -------
        numpy.ndarray
            The Jacobian of polynomial with respect to free pars.
        """
        nfree = None
        if self.jacobianf is None:
            nfree = (pars is True).sum()
            if nfree != 0:
                emsg = "No jacobian routine provided to Arbitrary."
                raise NotImplementedError(emsg)
        if len(pars) != self.npars:
            emsg = "Argument pars must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        if len(free) != self.npars:
            emsg = "Argument free must have " + str(self.npars) + " elements."
            raise ValueError(emsg)

        # Allow an arbitrary function without a Jacobian provided act as
        # though it has one if no actual Jacobian would be calculated
        # anyway.  This is a minor convenience for the user, but one with
        # large performance implications if all other functions used while
        # fitting a function define a Jacobian.
        if nfree == 0:
            return [None for p in range(len(pars))]

        # TODO: check that jacobianf returns something proper?
        return self.jacobianf(pars, r, free)

    def _transform_parametersraw(self, pars, in_format, out_format):
        """Convert parameter values from in_format to out_format.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters
        in_format : internal
            The format defined for this class
        out_format: internal
            The format defined for this class

        Defined Format
        --------------
        internal: [a_0, a_1, ...]

        Returns
        -------
        numpy.ndarray
            The standard output of transformed parameters
        """
        temp = np.array(pars)

        # Convert to intermediate format "internal"
        if in_format == "internal":
            pass
        else:
            raise ValueError("Argument 'in_format' must be one of %s." % self.parformats)

        # Convert to specified output format from "internal" format.
        if out_format == "internal":
            pass
        else:
            raise ValueError("Argument 'out_format' must be one of %s." % self.parformats)
        return temp

    def _valueraw(self, pars, r):
        """Compute the value of the polynomial given a set of parameters and evaluation points.

        This method ensures that the input parameters conform to the expected count
        and then delegates the computation to an internal method `valuef`.

        Parameters
        ----------
        pars : array_like
            The sequence of coefficients for the polynomial where each element corresponds to:
            - pars[0] = a_0, the constant term
            - pars[1] = a_1, the coefficient of the first degree term, and so on.
            The length of `pars` must match the expected number of parameters defined in the class.

        r : array_like or float
            The sequence of points or a single point at which the polynomial is to be evaluated.
            If a scalar is provided, it will be treated as a single point for evaluation.

        Returns
        -------
        ndarray or float
            The computed values of the polynomial for each point in `r`.
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have " + str(self.npars) + " elements."
            raise ValueError(emsg)

        # TODO: check that valuef returns something proper?
        return self.valuef(pars, r)

    def getmodule(self):
        return __name__


# end of class Polynomial

# simple test code
if __name__ == "__main__":

    f = Polynomial(degree=3)
    r = np.arange(5)
    pars = np.array([3, 0, 1, 2])
    free = np.array([True, False, True, True])
    print(f._valueraw(pars, r))
    print(f._jacobianraw(pars, r, free))

    f = Polynomial(degree=-1)
    r = np.arange(5)
    pars = np.array([])
    free = np.array([])
    print(f._valueraw(pars, r))
    print(f._jacobianraw(pars, r, free))
