#!/usr/bin/env python
##############################################################################
#
# SrMise            by Luke Granlund
#                   (c) 2014 trustees of the Michigan State University.
#                   All rights reserved.
#
# File coded by:    Luke Granlund
#
# See LICENSE.txt for license information.
#
##############################################################################

import numpy as np
from diffpy.srmise.baselines.base import BaselineFunction
from diffpy.srmise.srmiseerrors import SrMiseEstimationError

import matplotlib.pyplot as plt

import logging, diffpy.srmise.srmiselog
logger = logging.getLogger("diffpy.srmise")

class Arbitrary (BaselineFunction):
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
        npars: Number of parameters which define the function
        valuef: Function which calculates the value of the baseline
               at x.
        jacobianf: (None) Function which calculates the Jacobian of the
                  baseline function with respect to free pars.
        estimatef: (None) Function which estimates function parameters given the
                  data x and y.
        Cache: (None) A class (not instance) which implements caching of
               BaseFunction evaluations.
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
        for d in range(self.testnpars+1):
            parameterdict["a_"+str(d)] = d
        formats = ['internal']
        default_formats = {'default_input':'internal', 'default_output':'internal'}

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
        metadict = {}
        metadict["npars"] = (npars, repr)
        metadict["valuef"] = (valuef, repr)
        metadict["jacobianf"] = (jacobianf, repr)
        metadict["estimatef"] = (estimatef, repr)
        BaselineFunction.__init__(self, parameterdict, formats, default_formats, metadict, None, Cache)

    #### Methods required by BaselineFunction ####

    def estimate_parameters(self, r, y):
        """Estimate parameters for data baseline.

        Parameters
        r: (Numpy array) Data along r from which to estimate
        y: (Numpy array) Data along y from which to estimate

        Returns Numpy array of parameters in the default internal format.
        Raises NotImplementedError if no estimation routine is defined, and
        SrMiseEstimationError if parameters cannot be estimated for any other."""
        if self.estimatef is None:
            emsg = "No estimation routine provided to Arbitrary."
            raise NotImplementedError(emsg)

        # TODO: check that estimatef returns something proper?
        try:
            return self.estimatef(r, y)
        except Exception, e:
            emsg = "Error within estimation routine provided to Arbitrary:\n"+\
                   str(e)
            raise SrMiseEstimationError(emsg)

    def _jacobianraw(self, pars, r, free):
        """Return the Jacobian of a polynomial.

        Parameters
        pars: Sequence of parameters
             pars[0] = a_0
             pars[1] = a_1
             ...
        r: sequence or scalar over which pars is evaluated
        free: sequence of booleans which determines which derivatives are
              needed.  True for evaluation, False for no evaluation."""
        nfree = None
        if self.jacobianf is None:
            nfree = (pars == True).sum()
            if nfree != 0:
                emsg = "No jacobian routine provided to Arbitrary."
                raise NotImplementedError(emsg)
        if len(pars) != self.npars:
            emsg = "Argument pars must have "+str(self.npars)+" elements."
            raise ValueError(emsg)
        if len(free) != self.npars:
            emsg = "Argument free must have "+str(self.npars)+" elements."
            raise ValueError(emsg)

        # Allow an arbitrary function without a Jacobian provided act as
        # though it has one if no actual Jacobian would be calculated
        # anyway.  This is a minor convenience for the user, but one with
        # large performance implications if all other functions used while
        # fitting a function define a Jacobian.
        if nfree == 0:
            return [None for p in range(len(par))]

        # TODO: check that jacobianf returns something proper?
        return self.jacobianf(pars, r, free)

    def _transform_parametersraw(self, pars, in_format, out_format):
        """Convert parameter values from in_format to out_format.

        Parameters
        pars: Sequence of parameters
        in_format: A format defined for this class
        out_format: A format defined for this class

        Defined Formats
        internal: [a_0, a_1, ...]"""
        temp = np.array(pars)

        # Convert to intermediate format "internal"
        if in_format == "internal":
            pass
        else:
            raise ValueError("Argument 'in_format' must be one of %s." \
                              % self.parformats)

        # Convert to specified output format from "internal" format.
        if out_format == "internal":
            pass
        else:
            raise ValueError("Argument 'out_format' must be one of %s." \
                              % self.parformats)
        return temp

    def _valueraw(self, pars, r):
        """Return value of polynomial for the given parameters and r values.

        Parameters
        Parameters
        pars: Sequence of parameters
            pars[0] = a_0
            pars[1] = a_1
            ...
        r: sequence or scalar over which pars is evaluated"""
        if len(pars) != self.npars:
            emsg = "Argument pars must have "+str(self.npars)+" elements."
            raise ValueError(emsg)

        # TODO: check that valuef returns something proper?
        return self.valuef(pars, r)

    def getmodule(self):
        return __name__

#end of class Polynomial

# simple test code
if __name__ == '__main__':

    f = Polynomial(degree = 3)
    r = np.arange(5)
    pars = np.array([3, 0, 1, 2])
    free = np.array([True, False, True, True])
    print f._valueraw(pars, r)
    print f._jacobianraw(pars, r, free)

    f = Polynomial(degree = -1)
    r = np.arange(5)
    pars = np.array([])
    free = np.array([])
    print f._valueraw(pars, r)
    print f._jacobianraw(pars, r, free)
