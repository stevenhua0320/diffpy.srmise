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
import scipy.fftpack as fp

from diffpy.srmise.peaks.base import PeakFunction

logger = logging.getLogger("diffpy.srmise")


class TerminationRipples(PeakFunction):
    """Methods for evaluation and parameter estimation of a peak function with termination ripples."""

    def __init__(self, base, qmax, extension=4.0, supersample=5.0, Cache=None):
        """Peak function constructor which adds termination ripples to existing function.

        Unlike other peak functions, TerminationRipples can only be evaluated
        over a uniform grid, or at a single value using an ad hoc uniform grid
        defined by qmax, extension, and supersample.

        Parameters
        ----------
        base : PeakFunction instance
            The PeakFunction instance subclass.
        qmax : float
            The cut-off frequency in reciprocal space.
        extension : float
            How many multiples of 2pi/qmax to extend calculations in
            order to avoid edge effects. Default is 4.0.
        supersample : float
            Number intervals over 2pi/qmax when a natural interval
            cannot be determined while extending calculations. Default is 5.0.
        Cache : class
            The class (not instance) which implements caching of PeakFunction
               evaluations."""
        parameterdict = base.parameterdict
        formats = base.parformats
        default_formats = base.default_formats
        self.base = base
        self.qmax = qmax
        self.extension = extension
        self.supersample = supersample
        metadict = {}
        metadict["qmax"] = (qmax, repr)
        metadict["extension"] = (extension, repr)
        metadict["supersample"] = (supersample, repr)
        PeakFunction.__init__(self, parameterdict, formats, default_formats, metadict, base, Cache)
        return

    # Methods required by PeakFunction ####

    # TODO: A smart way to convert from the basefunctions estimate to an
    # appropriate one when ripples are considered.  This may not be necessary,
    # though.
    def estimate_parameters(self, r, y):
        """Estimate parameters for single peak from data provided.

        Uses estimation routine provided by base peak function.

        Parameters
        ----------
        r : array-like
            Data along r from which to estimate
        y : array-like
            Data along y from which to estimate

        Returns
        -------
        array-like
            Numpy array of parameters in the default internal format.
            Raises SrMiseEstimationError if parameters cannot be estimated for any
            reason.
        """
        return self.base.estimate_parameters(r, y)

    # TODO: Can this be implemented sanely for termination ripples?
    def scale_at(self, pars, x, scale):
        """Change parameters so value(x)->scale*value(x) for the base function.

        Does not change position or height of peak's maxima.  Raises
        SrMiseScalingError if the parameters cannot be scaled.

        Parameters
        ----------
         pars : array-like
            The parameters corresponding to a single peak
        x : float
            The position of the border
        scale : float
            The size of scaling at x. Must be positive.

        Returns
        -------
        array-like
            The numpy array of scaled parameters.
        """
        return self.base.scale_at(pars, x, scale)

    def _jacobianraw(self, pars, r, free):
        """Return Jacobian of base function with termination ripples.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters for a single peak
        r : array-like
            The sequence or scalar over which pars is evaluated
        free : array-like
            The sequence of booleans which determines which derivatives are
            needed.  True for evaluation, False for no evaluation.

        Returns
        -------
        array-like
            The Jacobian matrix of base function with termination ripples.
        """
        return self.base._jacobianraw(pars, r, free)

    def _transform_derivativesraw(self, pars, in_format, out_format):
        """Return gradient matrix for the pars converted from in_format to out_format.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters
        in_format : str
            The format defined for base peak function
        out_format : str
            The format defined for base peak function

        Returns
        -------
        ndarray
            The Jacobian matrix of base function with termination ripples with out_format.
        """
        return self.base._transform_derivativesraw(pars, in_format, out_format)

    def _transform_parametersraw(self, pars, in_format, out_format):
        """Convert parameter values from in_format to out_format.

        Parameters
        ----------
        pars : array-like
            The sequence of parameters
        in_format : str
            The format defined for base peak function
        out_format : str
            The format defined for base peak function

        Returns
        -------
        array-like
            The sequence of parameter values with out_format.
        """
        return self.base._transform_parametersraw(pars, in_format, out_format)

    def _valueraw(self, pars, r):
        """Return value of base peak function for the given parameters and r values.

        pars : array-like
            The sequence of parameters for a single peak
        r : array-like or float
            The sequence or scalar over which pars is evaluated

        Returns
        -------
        float
            The value of base peak function for the given parameters and r."""
        return self.base._valueraw(pars, r)

    # Overridden PeakFunction functions ####
    # jacobian() and value() are not normally overridden by PeakFunction
    # subclasses, but are here to minimize the effect of edge-effects while
    # introducing termination ripples.

    def jacobian(self, peak, r, rng=None):
        """Calculate (rippled) jacobian, possibly restricted by range.

        Parameters
        ----------
        peak : PeakFunction instance
            The Peak to be evaluated
        r : array-like
            The sequence or scalar over which peak is evaluated
        rng : slice object
            Optional slice object restricts which r-values are evaluated.
            The output has same length as r, but unevaluated objects have
            a default value of 0.  If caching is enabled these may be
            previously calculated values instead. Default is None

        Returns
        -------
        jac : array-like
            The Jacobian of base function with termination ripples."""
        if self is not peak._owner:
            raise ValueError(
                "Argument 'peak' must be evaluated by the "
                "PeakFunction subclass instance with which "
                "it is associated."
            )

        # normally r will be a sequence, but also allow single numeric values
        try:
            if len(r) > 1:
                dr = (r[-1] - r[0]) / (len(r) - 1)
            else:
                # dr is ad hoc if r is a single point
                dr = 2 * np.pi / (self.supersample * self.qmax)

            if rng is None:
                rng = slice(0, len(r))
            rpart = r[rng]
            (ext_r, ext_slice) = self.extend_grid(rpart, dr)
            jac = self._jacobianraw(peak.pars, ext_r, peak.free)
            output = [None for j in jac]
            for idx in range(len(output)):
                if jac[idx] is not None:
                    jac[idx] = self.cut_freq(jac[idx], dr)
                    output[idx] = r * 0.0
                    output[idx][rng] = jac[idx][ext_slice]
            return output
        except TypeError:
            # dr is ad hoc if r is a single point.
            dr = 2 * np.pi / (self.supersample * self.qmax)
            (ext_r, ext_slice) = self.extend_grid(np.array([r]), dr)
            jac = self._jacobianraw(peak.pars, ext_r, peak.free)
            for idx in range(len(output)):
                if jac[idx] is not None:
                    jac[idx] = self.cut_freq(jac[idx], dr)[ext_slice][0]
            return jac

    def value(self, peak, r, rng=None):
        """Calculate (rippled) value of peak, possibly restricted by range.

        This function overrides its counterpart in PeakFunction in order
        to minimize the impact of edge-effects from introducing termination
        ripples into an existing peak function.

        Parameters
        ----------
        peak : Peak instance
            The Peak to be evaluated
        r : array-like
            The sequence or scalar over which peak is evaluated
        rng : slice object
            Optional slice object restricts which r-values are evaluated.
            The output has same length as r, but unevaluated objects have
            a default value of 0.  If caching is enabled these may be
            previously calculated values instead. Default is None.

        Returns
        -------
        output : array-like
            The (rippled) value of peak, possibly restricted by range.
        """
        if self is not peak._owner:
            raise ValueError(
                "Argument 'peak' must be evaluated by the "
                "PeakFunction subclass instance with which "
                "it is associated."
            )

        # normally r will be a sequence, but also allow single numeric values

        dr_super = 2 * np.pi / (self.supersample * self.qmax)
        if np.isscalar(r):
            # dr is ad hoc if r is a single point.
            (ext_r, ext_slice) = self.extend_grid(np.array([r]), dr_super)
            value = self._valueraw(peak.pars, ext_r)
            value = self.cut_freq(value, dr_super)
            return value[ext_slice][0]
        else:
            if rng is None:
                rng = slice(0, len(r))

            output = r * 0.0

            # Make sure the actual dr used for finding termination ripples
            # is at least as fine as dr_super, while still calculating the
            # function at precisely the requested points.
            # When the underlying function is sampled too coarsely it can
            # miss critical high frequency components and return a very
            # poor approximation to the continuous case.  The actual fineness
            # of sampling needed to avoid the worst of these discretization
            # issues is difficult to determine without detailed knowledge
            # of the underlying function.
            dr = (r[-1] - r[0]) / (len(r) - 1)
            segments = np.ceil(dr / dr_super)
            dr_segmented = dr / segments

            rpart = r[rng]
            if segments > 1:
                rpart = np.arange(rpart[0], rpart[-1] + dr_segmented / 2, dr_segmented)

            (ext_r, ext_slice) = self.extend_grid(rpart, dr_segmented)
            value = self._valueraw(peak.pars, ext_r)
            value = self.cut_freq(value, dr_segmented)
            output[rng] = value[ext_slice][::segments]

            return output

    def getmodule(self):
        return __name__

    # Other methods ####

    def cut_freq(self, sequence, delta):
        """Remove high-frequency components from sequence.

        This is equivalent to the discrete convolution of a signal with a sinc
        function sin(2*pi*r/qmax)/r.

        Parameters
        ----------
        sequence : array-like
            The sequence to alter.
        delta : int
            The spacing between elements in sequence.

        Returns
        -------
        array-like
            The sequence with high-frequency components removed.
        """
        padlen = int(2 ** np.ceil(np.log2(len(sequence))))
        padseq = fp.fft(sequence, padlen)
        dq = 2 * np.pi / ((padlen - 1) * delta)
        lowidx = int(np.ceil(self.qmax / dq))
        hiidx = padlen + 1 - lowidx

        # Remove hi-frequency components
        padseq[lowidx:hiidx] = 0

        padseq = fp.ifft(padseq)
        return np.real(padseq[0 : len(sequence)])

    def extend_grid(self, r, dr):
        """Return (extended r, slice giving original range).

        Parameters
        ----------
        r : array-like or float
            The sequence or scalar over which peak is evaluated
        dr : array-like or float
            The uncertainties over which peak is evaluated

        Returns
        -------
        tuple
            The extended r, slice giving original range."""
        ext = self.extension * 2 * np.pi / self.qmax
        left_ext = np.arange(r[0] - dr, max(0.0, r[0] - ext - dr), -dr)[::-1]
        right_ext = np.arange(r[-1] + dr, r[-1] + ext + dr, dr)
        ext_r = np.concatenate((left_ext, r, right_ext))
        ext_slice = slice(len(left_ext), len(ext_r) - len(right_ext))
        return (ext_r, ext_slice)


# end of class TerminationRipples

# simple test code
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from numpy.random import randn

    from diffpy.srmise.modelcluster import ModelCluster
    from diffpy.srmise.modelevaluators.aicc import AICc
    from diffpy.srmise.peaks.base import Peaks
    from diffpy.srmise.peaks.gaussianoverr import GaussianOverR

    res = 0.01
    r = np.arange(2, 4, res)
    err = np.ones(len(r))  # default unknown errors
    pf1 = GaussianOverR(0.7)
    pf2 = TerminationRipples(pf1, 20.0)
    evaluator = AICc()

    pars = [[3, 0.2, 10], [3.5, 0.2, 10]]
    ideal_peaks = Peaks([pf1.actualize(p, "pwa") for p in pars])
    ripple_peaks = Peaks([pf2.actualize(p, "pwa") for p in pars])
    y_ideal = ideal_peaks.value(r)
    y_ripple = ripple_peaks.value(r) + 0.1 * randn(len(r))

    guesspars = [[2.7, 0.15, 5], [3.7, 0.3, 5]]
    guess_peaks = Peaks([pf2.actualize(p, "pwa") for p in guesspars])
    cluster = ModelCluster(guess_peaks, r, y_ripple, err, None, AICc, [pf2])

    qual1 = cluster.quality()
    print(qual1.stat)
    cluster.fit()
    yfit = cluster.calc()
    qual2 = cluster.quality()
    print(qual2.stat)

    plt.figure(1)
    plt.plot(r, y_ideal, r, y_ripple, r, yfit)
    plt.show()
