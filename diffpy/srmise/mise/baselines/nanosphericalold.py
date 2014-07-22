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
from diffpy.srmise.mise.baselines.base import BaselineFunction
from diffpy.srmise.mise.miseerrors import MiseEstimationError
import matplotlib.pyplot as plt

import logging, diffpy.srmise.mise.miselog
logger = logging.getLogger("mise.peakextraction")

class NanoSpherical (BaselineFunction):
    """Methods for evaluation and parameter estimation of the baseline for a spherical nanoparticle.

    Allowed formats are
    internal: [scale, diameter]

    The function is -scale*12(r/d)^2*(2-3*r/d+(r/d)^3)/(r*d) in the interval
    (0, abs(diameter)), and 0 elsewhere.  Internally, both scale and diameter
    are unconstrained, but negative values are mapped to their physically
    meaningful positive equivalents.
    """

    def __init__(self, Cache=None):
        """Initialize a spherical nanoparticle baseline.

        Parameters
        Cache - A class (not instance) which implements caching of BaseFunction
               evaluations.
        """
        # Define parameterdict
        parameterdict = {'scale':0, 'diameter':1}
        formats = ['internal']
        default_formats = {'default_input':'internal', 'default_output':'internal'}
        metadict = {}
        BaselineFunction.__init__(self, parameterdict, formats, default_formats, metadict, None, Cache)

    #### Methods required by BaselineFunction ####

#    def estimate_parameters(self, r, y):
#        """Estimate parameters for spherical baseline. (Not implemented!)
#
#        Parameters
#        r - array along r from which to estimate
#        y - array along y from which to estimate
#
#        Returns Numpy array of parameters in the default internal format.
#        Raises NotImplementedError if estimation is not implemented for this
#        degree, or MiseEstimationError if parameters cannot be estimated for
#        any other reason.
#        """
#        if len(r) != len(y):
#            emsg = "Arrays r, y must have equal length."
#            raise ValueError(emsg)

    def _jacobianraw(self, pars, r, free):
        """Return the Jacobian of the spherical baseline.

        Parameters
        pars - Sequence of parameters for a spherical baseline
               pars[0] = scale
               pars[1] = diameter
        r - sequence or scalar over which pars is evaluated.
        free - sequence of booleans which determines which derivatives are
               needed.  True for evaluation, False for no evaluation.
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have "+str(self.npars)+" elements."
            raise ValueError(emsg)
        if len(free) != self.npars:
            emsg = "Argument free must have "+str(self.npars)+" elements."
            raise ValueError(emsg)
        jacobian = [None for p in range(self.npars)]
        if (free == False).sum() == self.npars:
            return jacobian

        if np.isscalar(r):
            if r <= 0. or r >= pars[1]:
                if free[0]: jacobian[0] = 0.
                if free[1]: jacobian[1] = 0.
            else:
                if free[0]: jacobian[0] = self._jacobianrawscale(pars, r)
                if free[1]: jacobian[1] = self._jacobianrawdiameter(pars, r)
        else:
            s = self._getdomain(pars, r)
            if free[0]:
                jacobian[0] = np.zeros(len(r))
                jacobian[0][s] = self._jacobianrawscale(pars, r[s])
            if free[1]:
                jacobian[1] = np.zeros(len(r))
                jacobian[1][s] = self._jacobianrawdiameter(pars, r[s])
        return jacobian

    def _jacobianrawscale(self, pars, r):
        """Return partial Jacobian wrt scale without bounds checking.

        Parameters
        pars - Sequence of parameters for a spherical baseline
               pars[0] = scale
               pars[1] = diameter
        r - sequence or scalar over which pars is evaluated.
        """
        s = np.abs(pars[0])
        d = np.abs(pars[1])
        rdivd = r/d
        # From abs'(s) in derivative, which is equivalent to sign(d) except at 0 where it
        # is undefined.  I arbitrarily choose to use the positive sign.
        if pars[1] >= 0:
            sign = 1
        else:
            sign = -1
        return -sign*12*rdivd**2*(2-3*rdivd+rdivd**3)/(r*d)

    def _jacobianrawdiameter(self, pars, r):
        """Return partial Jacobian wrt diameter without bounds checking.

        Parameters
        pars - Sequence of parameters for a spherical baseline
               pars[0] = scale
               pars[1] = diameter
        r - sequence or scalar over which pars is evaluated.
        """
        s = np.abs(pars[0])
        d = np.abs(pars[1])
        # From abs'(d) in derivative, which is equivalent to sign(d) except at 0 where it
        # is undefined.  Since d=0 is a singularity anyway, sign will be fine.
        sign = np.sign(pars[1])
        return sign*s*72*(d**3-2*d**2*r+r**3)/d**7

    def _transform_parametersraw(self, pars, in_format, out_format):
        """Convert parameter values from in_format to out_format.

        Parameters
        pars - Sequence of parameters
        in_format - A format defined for this class
        out_format - A format defined for this class

        Defined Formats
        internal - [scale, diameter]
        """
        temp = np.array(pars)

        # Convert to intermediate format "internal"
        if in_format == "internal":
            # Map both scale and diameter to their positive equivalents
            temp[0] = np.abs(temp[0])
            temp[1] = np.abs(temp[1])
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
        """Return value of spherical baseline for the given parameters and r values.

        Outside the interval [0, diameter] the baseline is 0.

        Parameters
        pars - Sequence of parameters for a spherical baseline
               pars[0] = scale
               pars[1] = diameter
        r - sequence or scalar over which pars is evaluated.
        """
        if len(pars) != self.npars:
            emsg = "Argument pars must have "+str(self.npars)+" elements."
            raise ValueError(emsg)
        if np.isscalar(r):
            if r <= 0. or r >= pars[1]:
                return 0.
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
        pars - Sequence of parameters for a spherical baseline
               pars[0] = scale
               pars[1] = diameter
        r - sequence or scalar over which pars is evaluated.
        """
        s = np.abs(pars[0])
        d = np.abs(pars[1])
        rdivd = r/d
        return -s*12*rdivd**2*(2-3*rdivd+rdivd**3)/(r*d)

    def _getdomain(self, pars, r):
        """Return slice object for which r > 0 and r < diameter"""
        low = r.searchsorted(0., side='right')
        high = r.searchsorted(pars[1], side='left')
        return slice(low, high)

    def getmodule(self):
        return __name__

#end of class NanoSpherical

# simple test code
if __name__ == '__main__':

    f = NanoSpherical()
    r = np.arange(-5, 10)
    pars = np.array([-1., 7.])
    free = np.array([False, True])
    print "Testing nanoparticle spherical background"
    print "Scale: %f,  Diameter: %f" %(pars[0], pars[1])
    print "-----------------------------------------"
    val = f._valueraw(pars, r)
    jac =  f._jacobianraw(pars, r, free)
    outjac = [j if j is not None else [None]*len(r) for j in jac]
    print "r".center(10), "value".center(10), "jac(scale)".center(10), "jac(diameter)".center(10)
    for tup in zip(r, val, *outjac):
        for t in tup:
            if t is None:
                print ("%s" %None).ljust(10),
            else:
                print ("% .3g" %t).ljust(10),
        print
