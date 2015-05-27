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
from diffpy.srmise.peaks.base import PeakFunction
from diffpy.srmise.srmiseerrors import SrMiseEstimationError, SrMiseScalingError, SrMiseTransformationError
import matplotlib.pyplot as plt

import logging, diffpy.srmise.srmiselog
logger = logging.getLogger("diffpy.srmise")

class Gaussian (PeakFunction):
    """Methods for evaluation and parameter estimation of width-limited Gaussian.

       Allowed formats are
       internal: [position, parameterized width-squared, area]
       pwa: [position, full width at half maximum, area]
       mu_sigma_area: [mu, sigma, area]

       The internal parameterization is unconstrained, but are interpreted
       so that the width is between 0 and a user-provided maximum full width
       at half maximum, and the area is positive.

       Note that all full width at half maximum values are for the
       corresponding Gaussian.
    """

    # Possibly implement cutoff later, but low priority.
    # cutoff=3/np.sqrt(2*np.log(2))
    # cutoff defines a distance = maxwidth*cutoff from the maximum beyond
    # which the function is considered 0. By default this distance is
    # equivalent to 3 standard deviations.
    def __init__(self, maxwidth, Cache=None):
        """maxwidth defined as full width at half maximum for the
        corresponding Gaussian, which is physically relevant."""
        parameterdict={'position':0,'width':1,'area':2}
        formats=['internal','pwa','mu_sigma_area']
        default_formats={'default_input':'internal', 'default_output':'pwa'}
        metadict = {}
        metadict["maxwidth"] = (maxwidth, repr)
        PeakFunction.__init__(self, parameterdict, formats, default_formats, metadict, None, Cache)

        if maxwidth <= 0:
            emsg = "'maxwidth' must be greater than 0."
            raise ValueError(emsg)
        self.maxwidth = maxwidth

        ### Useful constants ###
        #c1 and c2 help with function values
        self.c1 = self.maxwidth*np.sqrt(np.pi/(8*np.log(2)))
        self.c2 = self.maxwidth**2/(8*np.log(2))

        #c3 and c4 help with parameter estimation
        self.c3 = .5*np.sqrt(np.pi/np.log(2))
        self.c4 = np.pi/(self.maxwidth*2)

        #convert sigma to fwhm: fwhm = 2 sqrt(2 log 2) sigma
        self.sigma2fwhm = 2*np.sqrt(2*np.log(2))

        return

    #### Methods required by PeakFunction ####

    def estimate_parameters(self, r, y):
        """Estimate parameters for single peak from data provided.

        Parameters
        r: (Numpy array) Data along r from which to estimate
        y: (Numpy array) Data along y from which to estimate

        Returns Numpy array of parameters in the default internal format.
        Raises SrMiseEstimationError if parameters cannot be estimated for any
        reason."""
        if len(r) != len(y):
            emsg = "Arrays r, y must have equal length."
            raise SrMiseEstimationError(emsg)

        logger.debug("Estimate peak using %s point(s)", len(r))

        minpoints_required = 3

        # filter out negative points
        usable_idx = [i for i in range(len(y)) if y[i] > 0]
        use_r = r[usable_idx]
        use_y = y[usable_idx]

        if len(usable_idx) < minpoints_required:
            emsg = "Not enough data for successful estimation."
            raise SrMiseEstimationError(emsg)

        #### Estimation ####
        guesspars = np.array([0., 0., 0.], dtype=float)
        min_y = use_y.min()
        max_y = use_y.max()
        center = use_r[use_y.argmax()]

        if min_y != max_y:
            weights = (use_y-min_y)**2
            guesspars[0] = np.sum(use_r*weights)/sum(weights)
            # guesspars[0] = center
            if use_y[0] < max_y:
                sigma_left = np.sqrt(-.5*(use_r[0]-guesspars[0])**2/np.log(use_y[0]/max_y))
            else:
                sigma_left = np.sqrt(-.5*np.mean(np.abs(np.array([use_r[0]-guesspars[0], use_r[-1]-guesspars[0]])))**2/np.log(min_y/max_y))
            if use_y[-1]<max_y:
                sigma_right = np.sqrt(-.5*(use_r[-1]-guesspars[0])**2/np.log(use_y[-1]/max_y))
            else:
                sigma_right = np.sqrt(-.5*np.mean(np.abs(np.array([use_r[0]-guesspars[0],use_r[-1]-guesspars[0]])))**2/np.log(min_y/max_y))
            guesspars[1] = .5*(sigma_right+sigma_left)*self.sigma2fwhm
        else:
            # Procede cautiously if min_y == max_y.  Without other information
            # we choose the center of the cluster as the peak center, and make
            # sure the peak has died down by the time it reaches the edge of
            # the data.
            guesspars[0] = (use_r[0]+use_r[-1])/2
            guesspars[1] = (use_r[-1]-use_r[0])*2/(2*np.log(2)) # cluster width/2=2*sigma

        if guesspars[1] > self.maxwidth:
            #account for width-limit
            guesspars[2] = self.c3*max_y*self.maxwidth
            guesspars[1] = np.pi/2 #parameterized in terms of sin
        else:
            guesspars[2] = self.c3*max_y*guesspars[1]
            guesspars[1] = np.arcsin(2*guesspars[1]**2/self.maxwidth**2-1.) #parameterized in terms of sin

        return guesspars

    def scale_at(self, pars, x, scale):
        """Change parameters so value(x)->scale*value(x).

        Does not change position or height of peak's maxima.  Raises
        SrMiseScalingError if the parameters cannot be scaled.

        Parameters
        pars: (Array) Parameters corresponding to a single peak
        x: (float) Position of the border
        scale: (float > 0) Size of scaling at x."""
        if scale <= 0:
            emsg = ''.join(["Cannot scale by ", str(scale), "."])
            raise SrMiseScalingError(emsg)

        if scale == 1:
            return pars
        else:
            ratio = 1/scale # Ugly: Equations orig. solved in terms of ratio

        tpars = self.transform_parameters(pars, in_format="internal", out_format="mu_sigma_area")

        #solves 1. f(rmax;mu1,sigma1,area1)=f(rmax;mu2,sigma2,area2)
        #       2. f(x;mu1,sigma1,area1)=ratio*f(x;mu1,sigma2,area2)
        #       3. mu1=mu2=rmax (the maximum of a Gaussian occurs at r=mu)
        # for mu2, sigma2, area2 (with appropriate unit conversions to fwhm at the end).
        # The expression for rmax is the appropriate solution to df/dr=0
        mu1, sigma1, area1 = tpars

        # the semi-nasty algebra reduces to something nice
        mu2 = mu1
        area2 = np.sqrt(area1**2/(2*np.log(ratio)*sigma1**2/(x-mu1)**2+1))
        sigma2 = sigma1*area2/area1

        tpars[0] = mu2
        tpars[1] = sigma2
        tpars[2] = area2
        try:
            tpars = self.transform_parameters(tpars, in_format="mu_sigma_area", out_format="internal")
        except SrMiseTransformationError, err:
            raise SrMiseScalingError(str(err))
        return tpars

    def _jacobianraw(self, pars, r, free):
        """Return Jacobian of width-limited Gaussian.

           pars: Sequence of parameters for a single width-limited Gaussian
           pars[0]=peak position
           pars[1]=effective width, up to fwhm=maxwidth as par[1] -> inf.
                 =tan(pi/2*fwhm/maxwidth)
           pars[2]=multiplicative constant a, equivalent to peak area
           r: sequence or scalar over which pars is evaluated
           free: sequence of booleans which determines which derivatives are
                 needed.  True for evaluation, False for no evaluation.
        """
        jacobian=[None, None, None]
        if (free == False).sum() == self.npars:
            return jacobian

        #Optimization
        sin_p = np.sin(pars[1]) + 1.
        p0minusr = pars[0]-r
        exp_p = np.exp(-(p0minusr)**2/(self.c2*sin_p))/(self.c1*np.sqrt(sin_p))

        if free[0]:
            #derivative with respect to peak position
            jacobian[0] = -2.*exp_p*p0minusr*np.abs(pars[2])/(self.c2*sin_p)
        if free[1]:
            #derivative with respect to reparameterized peak width
            jacobian[1] = -exp_p*np.abs(pars[2])*np.cos(pars[1])*(self.c2*sin_p-2*p0minusr**2)/(2.*self.c2*sin_p**2)
        if free[2]:
            #derivative with respect to peak area
            #abs'(x)=sign(x) for real x except at 0 where it is undetermined.  Since any real peak necessarily has
            #non-zero area and the function is paramaterized such that values of either sign represent equivalent
            #curves I arbitrarily choose positive sign for pars[2]==0 in order to push the system back into a realistic
            #parameter space should this improbable scenario occur.
            #   jacobian[2] = sign(pars[2])*exp_p
            if pars[2] >= 0:
                jacobian[2] = exp_p
            else:
                jacobian[2] = -exp_p
        return jacobian

    def _transform_parametersraw(self, pars, in_format, out_format):
        """Convert parameter values from in_format to out_format.

           Also restores parameters to a preferred range if it permits multiple
           values that correspond to the same physical result.

           Parameters
           pars: Sequence of parameters
           in_format: A format defined for this class
           out_format: A format defined for this class

           Defined Formats
           internal: [position, parameterized width-squared, area]
           pwa: [position, full width at half maximum, area]
           mu_sigma_area: [mu, sigma, area]
        """
        temp = np.array(pars)

        # Do I need to change anything?  The internal parameters may need to be
        # placed into the preferred range, even though their interpretation does
        # not change.
        if in_format == out_format and in_format != "internal":
            return pars

        # Convert to intermediate format "internal"
        if in_format == "internal":
            # put the parameter for width in the "physical" quadrant [-pi/2,pi/2],
            # where .5*(sin(p)+1) covers fwhm = [0, maxwidth]
            n = np.floor((temp[1]+np.pi/2)/np.pi)
            if np.mod(n, 2) == 0:
                temp[1] = temp[1] - np.pi*n
            else:
                temp[1] = np.pi*n - temp[1]
            temp[2] = np.abs(temp[2]) # map negative area to equivalent positive one
        elif in_format == "pwa":
            if temp[1] > self.maxwidth:
                emsg = "Width %s (FWHM) greater than maximum allowed width %s" %(temp[1], self.maxwidth)
                raise SrMiseTransformationError(emsg)
            temp[1] = np.arcsin(2.*temp[1]**2/self.maxwidth**2-1.)
        elif in_format == "mu_sigma_area":
            fwhm = temp[1]*self.sigma2fwhm
            if fwhm > self.maxwidth:
                emsg = "Width %s (FWHM) greater than maximum allowed width %s" %(fwhm, self.maxwidth)
                raise SrMiseTransformationError(emsg)
            temp[1] = np.arcsin(2.*fwhm**2/self.maxwidth**2-1.)
        else:
            raise ValueError("Argument 'in_format' must be one of %s." \
                              % self.parformats)

        # Convert to specified output format from "internal" format.
        if out_format == "internal":
            pass
        elif out_format == "pwa":
            temp[1] = np.sqrt(.5*(np.sin(temp[1])+1.)*self.maxwidth**2)
        elif out_format == "mu_sigma_area":
            temp[1] = np.sqrt(.5*(np.sin(temp[1])+1.)*self.maxwidth**2)/self.sigma2fwhm
        else:
            raise ValueError("Argument 'out_format' must be one of %s." \
                              % self.parformats)

    def _valueraw(self, pars, r):
        """Return value of width-limited Gaussian for the given parameters and r values.

           pars: Sequence of parameters for a single width-limited Gaussian
           pars[0]=peak position
           pars[1]=effective width, up to fwhm=maxwidth as par[1] -> inf.
                 =tan(pi/2*fwhm/maxwidth)
           pars[2]=multiplicative constant a, equivalent to peak area
           r: sequence or scalar over which pars is evaluated
        """
        return np.abs(pars[2])/(self.c1*np.sqrt(np.sin(pars[1])+1.))* \
            np.exp(-(r-pars[0])**2/(self.c2*(np.sin(pars[1])+1.)))

    def getmodule(self):
        return __name__

    #### Other methods ####

    def max(self, pars):
        """Return position and height of the peak maximum."""
        # TODO: Reconsider this behavior
        if len(pars) == 0:
            return None

        # Transform parameters for convenience.
        tpars = self.transform_parameters(pars, in_format="internal", out_format="mu_sigma_area")

        rmax = tpars[0]
        ymax = self._valueraw(pars, rmax)
        return np.array([rmax, ymax])

#end of class Gaussian

# simple test code
if __name__ == '__main__':

    from numpy.random import randn
    import matplotlib.pyplot as plt
    from diffpy.srmise.modelevaluators import AICc
    from diffpy.srmise.modelcluster import ModelCluster
    from diffpy.srmise.peaks import Peaks

    res = .01
    r = np.arange(2,4,res)
    err = np.ones(len(r)) # default unknown errors
    pf = Gaussian(.7)
    evaluator = AICc()

    pars = [[3, .2, 10], [3.5, .2, 10]]
    ideal_peaks = Peaks([pf.createpeak(p, "pwa") for p in pars])
    y = ideal_peaks.value(r) + .1*randn(len(r))

    guesspars = [[2.7, .15, 5], [3.7, .3, 5]]
    guess_peaks = Peaks([pf.createpeak(p, "pwa") for p in guesspars])
    cluster = ModelCluster(guess_peaks, r, y, err, None, AICc, [pf])

    qual1 = cluster.quality()
    print qual1.stat
    cluster.fit()
    yfit = cluster.calc()
    qual2 = cluster.quality()
    print qual2.stat

    plt.figure(1)
    plt.plot(r, y, r, yfit)
    plt.show()
