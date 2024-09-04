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

logger = logging.getLogger("diffpy.srmise")


class NanoSpherical(BaselineFunction):
    """Methods for evaluation of baseline of spherical nanoparticle of uniform density.

    Allowed formats are
    internal: [scale, radius]

    Given nanoparticle radius R, the baseline is -scale*r*(1-(3r)/(4R)+(r^3)/(16*R^3)) in the
    interval (0, abs(R)), and 0 elsewhere.  Internally, both scale and radius are unconstrained,
    but negative values are mapped to their physically meaningful positive equivalents.

    The expression in parentheses is gamma_0(r) for a sphere.  For a well normalized PDF the
    scale factor is 4*pi*rho_0, where rho_r is the nanoparticle density.

    gamma_0(r) Reference:
    Guinier et al. (1955). Small-angle Scattering from X-rays. New York: John Wiley & Sons, Inc.
    """

    def __init__(self, Cache=None):
        """Initialize a spherical nanoparticle baseline.

        Parameters
        ----------
        Cache : class
            THe class (not instance) which implements caching of BaseFunction
            evaluations.
        """
        # Define parameterdict
        parameterdict = {"scale": 0, "radius": 1}
        formats = ["internal"]
        default_formats = {"default_input": "internal", "default_output": "internal"}
        metadict = {}
        BaselineFunction.__init__(self, parameterdict, formats, default_formats, metadict, None, Cache)

    # Methods required by BaselineFunction ####

    #    def estimate_parameters(self, r, y):
    #        """Estimate parameters for spherical baseline. (Not implemented!)
    #
    #        Parameters
    #        r - array along r from which to estimate
    #        y - array along y from which to estimate
    #
    #        Returns Numpy array of parameters in the default internal format.
    #        Raises NotImplementedError if estimation is not implemented for this
    #        degree, or SrMiseEstimationError if parameters cannot be estimated for
    #        any other reason.
    #        """
    #        if len(r) != len(y):
    #            emsg = "Arrays r, y must have equal length."
    #            raise ValueError(emsg)

    def _jacobianraw(self, pars, r, free):
        """Return the Jacobian of the spherical baseline.

        Parameters
        ----------
        pars : array-like
            The Sequence of parameters for a spherical baseline
            pars[0] = scale
            pars[1] = radius
        r : array-like
            The sequence or scalar over which pars is evaluated.
        free : bool
            The sequence of booleans which determines which derivatives are
            needed. True for evaluation, False for no evaluation.

        Returns
        -------
        array-like
            The Jacobian of the nanospherical baseline.
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

        if np.isscalar(r):
            if r <= 0.0 or r >= 2.0 * pars[1]:
                if free[0]:
                    jacobian[0] = 0.0
                if free[1]:
                    jacobian[1] = 0.0
            else:
                if free[0]:
                    jacobian[0] = self._jacobianrawscale(pars, r)
                if free[1]:
                    jacobian[1] = self._jacobianrawradius(pars, r)
        else:
            s = self._getdomain(pars, r)
            if free[0]:
                jacobian[0] = np.zeros(len(r))
                jacobian[0][s] = self._jacobianrawscale(pars, r[s])
            if free[1]:
                jacobian[1] = np.zeros(len(r))
                jacobian[1][s] = self._jacobianrawradius(pars, r[s])
        return jacobian

    def _jacobianrawscale(self, pars, r):
        """Return partial Jacobian wrt scale without bounds checking.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters for a spherical baseline
            pars[0] = scale
            pars[1] = radius
        r : array-like
            The sequence or scalar over which pars is evaluated.

        Returns
        -------
        array-like
            The partial Jacobian of the nanoparticle baseline wrt scale without bounds checking.
        """
        np.abs(pars[0])
        R = np.abs(pars[1])
        rdivR = r / R
        # From abs'(s) in derivative, which is equivalent to sign(s) except at 0 where it
        # is undefined. Since s=0 is equivalent to the absence of a nanoparticle, sign will
        # be fine.
        sign = np.sign(pars[1])
        return -sign * r * (1 - (3.0 / 4.0) * rdivR + (1.0 / 16.0) * rdivR**3)

    def _jacobianrawradius(self, pars, r):
        """Return partial Jacobian wrt radius without bounds checking.

        Parameters
        ----------
        pars : array-like
            The Sequence of parameters for a spherical baseline
            pars[0] = scale
            pars[1] = radius
        r : array-like
            The sequence or scalar over which pars is evaluated.

        Returns
        -------
        array-like
            The partial Jacobian of the nanoparticle baseline wrt radius without bounds checking.
        """
        s = np.abs(pars[0])
        R = np.abs(pars[1])
        # From abs'(R) in derivative, which is equivalent to sign(R) except at 0 where it
        # is undefined.  Since R=0 is a singularity anyway, sign will be fine.
        sign = np.sign(pars[1])
        return sign * s * (3 * r**2 * (r**2 - 4 * R**2)) / (16 * R**4)

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
        internal - [scale, radius]

        Returns
        -------
        array-like
            The transformed parameter values with out_format.
        """
        temp = np.array(pars)

        # Convert to intermediate format "internal"
        if in_format == "internal":
            # Map both scale and radius to their positive equivalents
            temp[0] = np.abs(temp[0])
            temp[1] = np.abs(temp[1])
        else:
            raise ValueError("Argument 'in_format' must be one of %s." % self.parformats)

        # Convert to specified output format from "internal" format.
        if out_format == "internal":
            pass
        else:
            raise ValueError("Argument 'out_format' must be one of %s." % self.parformats)
        return temp

    def _valueraw(self, pars, r):
        """Return value of spherical baseline for the given parameters and r values.

        Outside the interval [0, radius] the baseline is 0.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters for a spherical baseline
            pars[0] = scale
            pars[1] = radius
        r : array-like
            The sequence or scalar over which pars is evaluated.

        Returns
        -------
        float
            The value of the spherical baseline.
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have " + str(self.npars) + " elements."
            raise ValueError(emsg)
        if np.isscalar(r):
            if r <= 0.0 or r >= 2.0 * pars[1]:
                return 0.0
            else:
                return self._valueraw2(pars, r)
        else:
            out = np.zeros(len(r))
            s = self._getdomain(pars, r)
            out[s] = self._valueraw2(pars, r[s])
            return out

    def _valueraw2(self, pars, r):
        """Return value of spherical baseline without bounds checking for given parameters and r values.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters for a spherical baseline
            pars[0] = scale
            pars[1] = radius
        r : array-like
            The sequence or scalar over which pars is evaluated.

        Returns
        -------
        float
            The value of spherical baseline without bounds checking for given parameters and r values
        """
        s = np.abs(pars[0])
        R = np.abs(pars[1])
        rdivR = r / R
        return -s * r * (1 - (3.0 / 4.0) * rdivR + (1.0 / 16.0) * rdivR**3)

    def _getdomain(self, pars, r):
        """Return slice object for which r > 0 and r < twice the radius

        Parameters
        ----------
        pars : array-like
            The sequence of parameters for a spherical baseline
        r : array-like
            The sequence or scalar over which pars is evaluated.

        Returns
        -------
        slice object
            The slice object for which r > 0 and r < twice the radius
        """
        low = r.searchsorted(0.0, side="right")
        high = r.searchsorted(2.0 * pars[1], side="left")
        return slice(low, high)

    def getmodule(self):
        return __name__


# end of class NanoSpherical

# simple test code
if __name__ == "__main__":

    f = NanoSpherical()
    r = np.arange(-5, 10)
    pars = np.array([-1.0, 7.0])
    free = np.array([False, True])
    print("Testing nanoparticle spherical baseline")
    print("Scale: %f,  Radius: %f" % (pars[0], pars[1]))
    print("-----------------------------------------")
    val = f._valueraw(pars, r)
    jac = f._jacobianraw(pars, r, free)
    outjac = [j if j is not None else [None] * len(r) for j in jac]
    print(
        "r".center(10),
        "value".center(10),
        "jac(scale)".center(10),
        "jac(radius)".center(10),
    )
    for tup in zip(r, val, *outjac):
        for t in tup:
            if t is None:
                print(f"{None}".ljust(10))
            else:
                print(f"{t:.3g}".ljust(10))
