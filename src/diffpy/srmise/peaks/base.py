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

from diffpy.srmise.basefunction import BaseFunction
from diffpy.srmise.modelparts import ModelPart, ModelParts
from diffpy.srmise.srmiseerrors import SrMiseDataFormatError, SrMiseScalingError

logger = logging.getLogger("diffpy.srmise")


class PeakFunction(BaseFunction):
    """Base class for functions which represent peaks.

    Class members
    -------------
    parameterdict: A dictionary mapping string keys to their index in the
                   sequence of parameters.  The "position" key is required,
                   while all others are arbitrary.  These keys apply only to
                   the default "internal" format.
    parformats: A sequence of strings defining what formats are recognized
                by a peak function.
    default_formats: A dictionary which maps the strings "default_input" and
                     "default_output" to strings also appearing in parformats.
                     "default_input"-> format used internally within the class
                     "default_output"-> Default format to use when converting
                                        parameters for outside use.

    Class methods (implemented by inheriting classes)
    -------------------------------------------------
    estimate_parameters()
    scale_at()
    _jacobianraw() (optional, but strongly recommended)
    _transform_derivativesraw() (optional, supports propagation of uncertainty for different paramaterizations)
    _transform_parametersraw()
    _valueraw()

    Class methods
    -------------
    actualize()

    Inherited methods
    -----------------
    jacobian()
    value()
    transform_derivatives()
    transform_parameters()
    """

    def __init__(
        self,
        parameterdict,
        parformats,
        default_formats,
        metadict,
        base=None,
        Cache=None,
    ):
        """Set parameterdict defined by subclass

        parameterdict: A dictionary mapping string keys to their index in a
                       sequence of parameters for this PeakFunction subclass.
                       The key "position" is required.
        parformats: A sequence strings containing all allowed input/output
                    formats defined for the peak function's parameters.
        default_formats: A dictionary mapping the string keys "internal" and
                        "default_output" to formats from parformats.
        metadict: Dictionary mapping string keys to tuple (v, m) where v is an
                  additional argument required by function, and m is a method
                  whose string output recreates v when passed to eval().
        base: A basefunction subclass instance which this one decorates with
              additional functionality.
        Cache: A class (not instance) which implements caching of BaseFunction
               evaluations."""
        if "position" not in parameterdict:
            emsg = "Argument parameterdict missing required key 'position'."
            raise ValueError(emsg)
        BaseFunction.__init__(self, parameterdict, parformats, default_formats, metadict, base, Cache)

    # # "Virtual" class methods ####

    def scale_at(self, peak, x, scale):
        emsg = "scale_at must be implemented in a PeakFunction subclass."
        raise NotImplementedError(emsg)

    # # Methods required by BaseFunction ####

    def actualize(
        self,
        pars,
        in_format="default_input",
        free=None,
        removable=True,
        static_owner=False,
    ):
        converted = self.transform_parameters(pars, in_format, out_format="internal")
        return Peak(self, converted, free, removable, static_owner)

    def getmodule(self):
        return __name__


# end of class PeakFunction


class Peaks(ModelParts):
    """A collection for Peak objects."""

    def __init__(self, *args, **kwds):
        # Check that args[0] (if it exists) is an instance of Peaks?
        ModelParts.__init__(self, *args, **kwds)

    def argsort(self, key="position"):
        """Return sequence of indices which sort peaks in order specified by key."""
        keypars = np.array([p[key] for p in self])
        # In normal use the peaks will already be sorted, so check for it.
        sorted = True
        for i in range(len(keypars) - 1):
            if keypars[i] > keypars[i + 1]:
                sorted = False
                break
        if not sorted:
            return keypars.argsort().tolist()
        else:
            return range(len(keypars))

    def match_at(self, x, y):
        """Alter peaks so their sum at x is y, preserving each peak's maximum.

        Each peak is scaled equally.  Peaks with fixed parameters, a maximum
        very close to x, or other issues may prevent optimal results.  If the
        peaks cannot be scaled at all they are left unchanged.

        Parameters:
        x: (float) Position at which to match.
        y: (float) Height to match.

        Returns True if one or more peaks was scaled, False otherwise.
        """
        height = self.value(x)
        if height == 0:
            return False

        orig = self.copy()

        try:
            scale = y / height

            # First attempt at scaling peaks.  Record which peaks, if any,
            # were not scaled in case a second attempt is required.
            scaled = []
            all_scaled = True
            any_scaled = False
            fixed_height = 0.0
            for peak in self:
                scaled.append(peak.scale_at(x, scale))
                all_scaled = all_scaled and scaled[-1]
                any_scaled = any_scaled or scaled[-1]
                if not scaled[-1]:
                    fixed_height += peak.value(x)

            # Second attempt at scaling peaks.
            if not all_scaled and fixed_height < y and fixed_height < height:
                self[:] = orig[:]
                any_scaled = False
                scale = (y - fixed_height) / (height - fixed_height)
                for peak, s in (self, scaled):
                    if s:
                        # "or" is short-circuited, so scale_at() must be first
                        # to guarantee it is called.
                        any_scaled = peak.scale_at(x, scale) or any_scaled
        except Exception as e:
            logger.debug("An exception prevented matching -- %s", e)
            self[:] = orig[:]
            return False
        return any_scaled

    def sort(self, reverse=False, key="position"):
        """Sort peaks in order specified by key."""
        keypars = np.array([p[key] for p in self])
        order = keypars.argsort()
        self[:] = [self[idx] for idx in order]
        return


# End of class Peaks


class Peak(ModelPart):
    """Represents a single peak associated with a PeakFunction subclass."""

    def __init__(self, owner, pars, free=None, removable=True, static_owner=False):
        """Set instance members.

        owner: an instance of a PeakFunction subclass
        pars: Sequence of parameters which define a single peak
        free: Sequence of Boolean variables.  If False, the corresponding
              parameter will not be changed.
        removable: Boolean determines whether this peak can be removed.
        static_owner: (False) Whether or not the owner can be changed with
                      changeowner()

        Note that free and removable are not mutually exclusive.  If any
        values are not free but removable=True then the entire peak may be
        removed during peak extraction, but the held parameters for this
        peak will remain unchanged until that point.
        """
        ModelPart.__init__(self, owner, pars, free, removable, static_owner)

    def scale_at(self, x, scale):
        """Change parameters so value(x)->scale*value(x).

        Does not change position or height of peak's maxima.  If parameters
        that are not free would be changed, or violates other constraints,
        the peak is not adjusted.

        Parameters
        x: (float) Position of the border
        scale: (float > 0) Amount by which to scale.

        Returns True if parameters were scaled, False otherwise.
        """
        # Reminder: Bitwise operators "&" and "~" work element-wise with
        # numpy arrays.

        # Check for no free parameters.
        if np.all(~self.free):
            return False

        try:
            adj_pars = self._owner.scale_at(self.pars, x, scale)
        except SrMiseScalingError as err:
            logger.debug("Cannot scale peak:", err)
            return False

        # Check if a fixed parameter was changed.
        if np.any((self.pars != adj_pars) & ~self.free):
            logger.debug("Cannot scale peak: a fixed parameter was changed")
            return False
        self.pars = adj_pars
        return True

    @staticmethod
    def factory(peakstr, ownerlist):
        """Instantiate a Peak from a string.

        Parameters:
        peakstr: string representing peak
        ownerlist: List of BaseFunctions that owner is in
        """

        data = peakstr.strip().splitlines()

        # dictionary of parameters
        pdict = {}
        for d in data:
            parse_value = d.split("=", 1)
            if len(parse_value) == 2:
                try:
                    pdict[parse_value[0]] = eval(parse_value[1])
                except Exception:
                    emsg = "Invalid parameter: %s" % d
                    raise SrMiseDataFormatError(emsg)
            else:
                emsg = "Invalid parameter: %s" % d
                raise SrMiseDataFormatError(emsg)

        # Correctly initialize the base function, if one exists.
        idx = pdict["owner"]
        if idx > len(ownerlist):
            emsg = "Dependent base function not in ownerlist."
            raise ValueError(emsg)
        pdict["owner"] = ownerlist[idx]

        return Peak(**pdict)


# End of class Peak

# simple test code
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from numpy.random import randn

    from diffpy.srmise.modelcluster import ModelCluster
    from diffpy.srmise.modelevaluators.aicc import AICc
    from diffpy.srmise.peaks.gaussianoverr import GaussianOverR

    res = 0.01
    r = np.arange(2, 4, res)
    err = np.ones(len(r))  # default unknown errors
    pf = GaussianOverR(0.7)
    evaluator = AICc()

    pars = [[3, 0.2, 10], [3.5, 0.2, 10]]
    ideal_peaks = Peaks([pf.actualize(p, "pwa") for p in pars])
    y = ideal_peaks.value(r) + 0.1 * randn(len(r))

    guesspars = [[2.7, 0.15, 5], [3.7, 0.3, 5]]
    guess_peaks = Peaks([pf.actualize(p, "pwa") for p in guesspars])
    cluster = ModelCluster(guess_peaks, r, y, err, None, AICc, [pf])

    qual1 = cluster.quality()
    print(qual1.stat)
    cluster.fit()
    yfit = cluster.calc()
    qual2 = cluster.quality()
    print(qual2.stat)

    plt.figure(1)
    plt.plot(r, y, r, yfit)
    plt.show()
