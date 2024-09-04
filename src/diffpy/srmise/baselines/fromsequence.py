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
import scipy.interpolate as spi

from diffpy.srmise.baselines.base import BaselineFunction

logger = logging.getLogger("diffpy.srmise")


class FromSequence(BaselineFunction):
    """Methods for evaluation of a baseline from discrete data via interpolation.

    FromSequence uses cubic spline interpolation (no smoothing) on discrete
    points to approximate the baseline at arbitrary points within the
    interpolation domain.  This baseline function permits no free parameters."""

    def __init__(self, *args, **kwds):
        """Initialize a baseline object based on input sequences `x` and `y`.

        This class provides two ways to initialize: by directly providing the sequences or by
        specifying a file that contains the sequences.

        Parameters
        ----------
        *args : tuple
            The variable length argument list. Can be used to pass `x` and `y` sequences directly.

        **kwds : dict
            The arbitrary keyword arguments. Can be used to specify `x`, `y` sequences or a `file` name.

        x : array_like, optional
            The sequence of x-values defining the baseline. Can be passed as a positional argument or via keyword.

        y : array_like, optional
            The sequence of y-values defining the baseline. Must be provided alongside `x`.

        file : str, optional
            The name of the file containing two columns: one for x-values and one for y-values.

        Usage
        -----
        1. Directly with sequences:
            - `FromSequence(xlist, ylist)`
            - `FromSequence(x=xlist, y=ylist)`

        2. From a file:
            - `FromSequence("filename")`
            - `FromSequence(file="filename")`
        """
        if len(args) == 1 and len(kwds) == 0:
            # load from file
            x, y = self.readxy(args[0])
        elif len(args) == 0 and ("file" in kwds and "x" not in kwds and "y" not in kwds):
            # load file
            x, y = self.readxy(kwds["file"])
        elif len(args) == 2 and len(kwds) == 0:
            # Load x, y directly from arguments
            x = args[0]
            y = args[1]
        elif len(args) == 0 and ("x" in kwds and "y" in kwds and "file" not in kwds):
            # Load x, y from keywords
            x = kwds["x"]
            y = kwds["y"]
        else:
            emsg = "Call to FromSequence does not match any allowed signature."
            raise TypeError(emsg)

        # Guarantee valid lengths
        if len(x) != len(y):
            emsg = "Sequences x and y must have the same length."
            raise ValueError(emsg)
        parameterdict = {}
        formats = ["internal"]
        default_formats = {"default_input": "internal", "default_output": "internal"}
        self.spline = spi.InterpolatedUnivariateSpline(x, y)
        self.minx = x[0]
        self.maxx = x[-1]
        metadict = {}
        metadict["x"] = (x, self.xyrepr)
        metadict["y"] = (y, self.xyrepr)
        BaselineFunction.__init__(self, parameterdict, formats, default_formats, metadict, None, Cache=None)

    # Methods required by BaselineFunction ####

    def estimate_parameters(self, r, y):
        """Return empty numpy array.

        A FromSequence object has no free parameters, so there is nothing
        to estimate.

        Parameters
        ----------
        r : array-like
            The data along r from which to estimate, Ignored
        y : array-like
            The data along y from which to estimate, Ignored

        Returns
        -------
        array-like
            The empty numpy array
        """
        return np.array([])

    def _jacobianraw(self, pars, r, free):
        """Return [].

        A FromSequence baseline has no parameters.

        Parameters
        ----------
        pars :array-like
            The empty sequence
        r : array-like
            The sequence or scalar over which pars is evaluated
        free : array-like
            The empty sequence.

        Returns
        -------
        array-like
            The empty numpy array
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        if len(free) != self.npars:
            emsg = "Argument free must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        return []

    def _transform_parametersraw(self, pars, in_format, out_format):
        """Convert parameter values from in_format to out_format.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters
        in_format : str
            The format defined for this class
        out_format : str
            The format defined for this class

        Defined Formats
        ---------------
        n/a, FromSequence has no parameters

        Returns
        -------
        array-like
            The sequence of parameters converted to out_format
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
            The empty sequence
        r : array-like
            The sequence or scalar over which pars is evaluated

        Returns
        -------
        float
            The value of the polynomial for the given parameters
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        try:
            if r[0] < self.minx or r[-1] > self.maxx:
                logger.warning(
                    "Warning: Evaluating interpolating function over %s, outside safe range of %s.",
                    [r[0], r[-1]],
                    [self.minx, self.maxx],
                )
        except (IndexError, TypeError):
            if r < self.minx or r > self.maxx:
                logger.warning(
                    "Warning: Evaluating interpolating function at %s, outside safe range of %s.",
                    r,
                    [self.minx, self.maxx],
                )
        return self.spline(r)

    def getmodule(self):
        return __name__

    def xyrepr(self, var):
        """Safe string output of x and y, compatible with eval()

        Parameters
        ----------
        var : array-like
            The sequence or scalar over which to evaluate

        Returns
        -------
        str
            The x and y values of the given variable
        """
        return "[%s]" % ", ".join([repr(v) for v in var])

    def readxy(self, filename):
        """ """
        from diffpy.srmise.srmiseerrors import SrMiseDataFormatError

        # TODO: Make this safer
        try:
            datastring = open(filename, "rb").read()
        except Exception as err:
            raise err

        import re

        res = re.search(rb"^[^#]", datastring, re.M)
        if res:
            datastring = datastring[res.end() :].strip()

        x = []
        y = []

        try:
            for line in datastring.split(b"\n"):
                v = line.split()
                x.append(float(v[0]))
                y.append(float(v[1]))
        except (ValueError, IndexError) as err:
            raise SrMiseDataFormatError(str(err))

        return (np.array(x), np.array(y))


# end of class FromSequence

# simple test code
if __name__ == "__main__":

    r = np.arange(0, 9.42413, 0.2)
    b = -(np.tanh(0.5 * r) + np.sin(0.5 * r))
    f = FromSequence(r, b)
    pars = np.array([])
    free = np.array([])

    r2 = np.arange(0, 9.42413, 0.5)
    b2 = f._valueraw(pars, r2)
