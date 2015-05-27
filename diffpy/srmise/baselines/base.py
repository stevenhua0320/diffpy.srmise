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
from diffpy.srmise.srmiseerrors import *
from diffpy.srmise.basefunction import BaseFunction
from diffpy.srmise.modelparts import ModelPart

import logging, diffpy.srmise.srmiselog
logger = logging.getLogger("diffpy.srmise")

class BaselineFunction(BaseFunction):
    """Base class for functions which represent some data's baseline term.

    Class members
    -------------
    parameterdict: A dictionary mapping string keys to their index in the
                   sequence of parameters.  These keys apply only to
                   the default "internal" format.
    parformats: A sequence of strings defining what formats are recognized
                by a baseline function.
    default_formats: A dictionary which maps the strings "default_input" and
                     "default_output" to strings also appearing in parformats.
                     "default_input"-> format used internally within the class
                     "default_output"-> Default format to use when converting
                                        parameters for outside use.

    Class methods (implemented by inheriting classes)
    -------------------------------------------------
    estimate_parameters() (optional)
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

    def __init__(self, parameterdict, parformats, default_formats, metadict, base=None, Cache=None):
        """Set parameterdict defined by subclass

        parameterdict: A dictionary mapping string keys to their index in a
                       sequence of parameters for this BaselineFunction subclass.
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
        BaseFunction.__init__(self, parameterdict, parformats, default_formats, metadict, base, Cache)


    #### "Virtual" class methods ####


    #### Methods required by BaseFunction ####

    def actualize(self, pars, in_format="default_input", free=None, removable=False, static_owner=False):
        converted = self.transform_parameters(pars, in_format, out_format="internal")
        return Baseline(self, converted, free, removable, static_owner)

    def getmodule(self):
        return __name__

#end of class BaselineFunction

class Baseline(ModelPart):
    """Represents a baseline associated with a BaselineFunction subclass."""

    def __init__(self, owner, pars, free=None, removable=False, static_owner=False):
        """Set instance members.

        owner: an instance of a BaselineFunction subclass
        pars: Sequence of parameters which define the baseline
        free: Sequence of Boolean variables.  If False, the corresponding
              parameter will not be changed.
        removable: (False) Boolean determines whether the baseline can be removed.
        static_owner: (False) Whether or not the owner can be changed with
                      changeowner()

        Note that free and removable are not mutually exclusive.  If any
        values are not free but removable=True then the entire baseline may be
        may be removed during peak extraction, but the held parameters for the
        baseline will remain unchanged until that point.
        """
        ModelPart.__init__(self, owner, pars, free, removable, static_owner)

    @staticmethod
    def factory(baselinestr, ownerlist):
        """Instantiate a Peak from a string.

        Parameters:
        baselinestr: string representing Baseline
        ownerlist: List of BaseFunctions that owner is in
        """
        from numpy import array

        data = baselinestr.strip().splitlines()

        # dictionary of parameters
        pdict = {}
        for d in data:
            l = d.split("=", 1)
            if len(l) == 2:
                try:
                    pdict[l[0]] = eval(l[1])
                except Exception:
                    emsg = ("Invalid parameter: %s" %d)
                    raise SrMiseDataFormatError(emsg)
            else:
                emsg = ("Invalid parameter: %s" %d)
                raise SrMiseDataFormatError(emsg)

        # Correctly initialize the base function, if one exists.
        idx = pdict["owner"]
        if idx > len(ownerlist):
            emsg = "Dependent base function not in ownerlist."
            raise ValueError(emsg)
        pdict["owner"] = ownerlist[idx]

        return Baseline(**pdict)

# End of class Baseline

# simple test code
if __name__ == '__main__':

    from numpy.random import randn
    import matplotlib.pyplot as plt
    from diffpy.srmise.modelevaluators import AICc
    from diffpy.srmise.modelcluster import ModelCluster
    from diffpy.srmise.peaks import GaussianOverR

    res = .01
    r = np.arange(2,4,res)
    err = np.ones(len(r)) #default unknown errors
    pf = GaussianOverR(.7)
    evaluator = AICc()

    pars = [[3, .2, 10], [3.5, .2, 10]]
    ideal_peaks = Peaks([pf.createpeak(p, "pwa") for p in pars])
    y = ideal_peaks.value(r) + .1*randn(len(r))
