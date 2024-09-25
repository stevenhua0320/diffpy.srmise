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
from diffpy.srmise.srmiseerrors import SrMiseEstimationError

logger = logging.getLogger("diffpy.srmise")


class Polynomial(BaselineFunction):
    """Methods for evaluation and parameter estimation of a polynomial baseline."""

    def __init__(self, degree, Cache=None):
        """Initialize a polynomial function of degree d.

        Parameters
        ----------
        degree: int
            The degree of the polynomial.  Any negative value is interpreted
            as the polynomial of negative infinite degree.
        Cache: class
            The class (not instance) which implements caching of BaseFunction
            evaluations.
        """
        # Guarantee valid degree
        try:
            self.degree = int(str(degree))
        except ValueError:
            emsg = "Argument degree must be an integer."
            raise ValueError(emsg)
        if self.degree < 0:
            self.degree = -1  # interpreted as negative infinity
        # Define parameterdict
        # e.g. {"a_0":3, "a_1":2, "a_2":1, "a_3":0} if degree is 3.
        parameterdict = {}
        for d in range(self.degree + 1):
            parameterdict["a_" + str(d)] = self.degree - d
        formats = ["internal"]
        default_formats = {"default_input": "internal", "default_output": "internal"}
        metadict = {"degree": (degree, repr)}
        BaselineFunction.__init__(self, parameterdict, formats, default_formats, metadict, None, Cache)

    # Methods required by BaselineFunction ####

    def estimate_parameters(self, r, y):
        """Estimate parameters for polynomial baseline.

        Estimation is currently implemented only for degree < 2.  This
        very rudimentary method assumes the baseline crosses the origin, and
        y=baseline+signal, where signal is primarily positive.

        Parameters
        ----------
        r : array-like
            The data along r from which to estimate
        y : array-like
            The data along y from which to estimate

        Returns
        -------
        array-like
            The Numpy array of parameters in the default internal format.
            Raises NotImplementedError if estimation is not implemented for this
            degree, or SrMiseEstimationError if parameters cannot be estimated for
            any other reason.
        """
        if self.degree > 1:
            emsg = "Polynomial implements estimation for baselines of degree <= 1 only."
            raise NotImplementedError(emsg)
        if len(r) != len(y):
            emsg = "Arrays r, y must have equal length."
            raise ValueError(emsg)

        if self.degree == -1:
            return np.array([])

        if self.degree == 0:
            return np.array([0.0])

        if self.degree == 1:
            # Estimate degree=1 baseline.
            # Find best slope for y=slope*r using only the least 10% of all
            # points, assuming the non-baseline component of the data largely
            # lies above the baseline.
            # TODO: Make this more sophisticated.
            try:
                cut = np.max([len(y) / 10, 1])
                cut_idx = y.argsort()[: int(cut)]

                import numpy.linalg as la

                a = np.array([r[cut_idx]]).T
                slope = la.lstsq(a, y[cut_idx], rcond=-1)[0][0]
                return np.array([slope, 0.0])
            except Exception as e:
                emsg = "Error during estimation -- " + str(e)
                raise SrMiseEstimationError(emsg)

    def _jacobianraw(self, pars, r, free):
        """Return the Jacobian of a polynomial.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters for a polynomial of degree d
            pars[0] = a_degree
            pars[1] = a_(degree-1)
            ...
            pars[d] = a_0
        r : array-like
            The sequence or scalar over which pars is evaluated
        free : bool
            The sequence of booleans which determines which derivatives are
              needed. True for evaluation, False for no evaluation.

        Returns
        -------
        jacobian: array-like
            The Jacobian of polynomial with degree d
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        if len(free) != self.npars:
            emsg = "Argument free must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        jacobian = [None for p in range(self.npars)]
        if np.sum(np.logical_not(free)) == self.npars:
            return jacobian

        # The partial derivative with respect to the nth coefficient of a
        # polynomial is just x^nth.
        for idx in range(self.npars):
            if free[idx]:
                jacobian[idx] = np.power(r, idx)
        return jacobian

    def _transform_parametersraw(self, pars, in_format, out_format):
        """Convert parameter values from in_format to out_format.

        Parameters
        pars : array-like
            The sequence of parameters
        in_format : str
            The format defined for this class
        out_format : str
            The format defined for this class

        Defined Formats
        ---------------
        internal: [a_degree, a_(degree-1), ..., a_0]

        Returns
        -------
        array-like
            The transformed parameters in out_format
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
        """Return value of polynomial for the given parameters and r values.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters for a polynomial of degree d
            pars[0] = a_degree
            pars[1] = a_(degree-1)
            ...
            pars[d] = a_0
            If degree is negative infinity, pars is an empty sequence.
        r : array-like
            The sequence or scalar over which pars is evaluated

        Returns
        -------
        float
            The value of polynomial for the given parameters and r values.
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        return np.polyval(pars, r)

    def getmodule(self):
        return __name__


# end of class Polynomial

# simple test code
if __name__ == "__main__":

    # Test polynomial of degree 3
    print("Testing degree 3 polynomial")
    print("---------------------------")
    f = Polynomial(degree=3)
    r = np.arange(5)
    pars = np.array([3, 0, 1, 2])
    free = np.array([True, False, True, True])
    val = f._valueraw(pars, r)
    jac = f._jacobianraw(pars, r, free)
    print("Value:\n", val)
    print("Jacobian: ")
    for j in jac:
        print(" %s" % j)

    # Test polynomial of degree -oo
    print("\nTesting degree -oo polynomial (== 0)")
    print("------------------------------------")
    f = Polynomial(degree=-1)
    r = np.arange(5)
    pars = np.array([])
    free = np.array([])
    val = f._valueraw(pars, r)
    jac = f._jacobianraw(pars, r, free)
    print("Value:\n", val)
    print("Jacobian: ")
    for j in jac:
        print(" %s" % j)

    # Test linear estimation
    print("\nTesting linear baseline estimation")
    print("------------------------------------")
    f = Polynomial(degree=1)
    pars = np.array([1, 0])
    r = np.arange(0, 10, 0.1)
    y = -r + 10 * np.exp(-((r - 5) ** 2)) + np.random.rand(len(r))
    est = f.estimate_parameters(r, y)
    print("Actual baseline: ", np.array([-1, 0.0]))
    # TODO: Make test est baseline in ways of tolerance function
    print("Estimated baseline: ", est)
