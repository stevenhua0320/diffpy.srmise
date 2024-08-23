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
from diffpy.srmise.modelparts import ModelPart
from diffpy.srmise.srmiseerrors import SrMiseDataFormatError

logger = logging.getLogger("diffpy.srmise")


class BaselineFunction(BaseFunction):
    """Base class for functions which represent some data's baseline term.

    Class members
    -------------
    parameterdict: dict
        The dictionary mapping string keys to their index in the
        sequence of parameters.  These keys apply only to
        the default "internal" format.
    parformats: array-like
        The sequence of strings defining what formats are recognized
        by a baseline function.
    default_formats: dict
        The dictionary which maps the strings "default_input" and
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

        parameterdict : dict
            The dictionary mapping string keys to their index in a
            sequence of parameters for this BaselineFunction subclass.
        parformats : array-like
            The sequence strings containing all allowed input/output
            formats defined for the peak function's parameters.
        default_formats : dict
            The dictionary mapping the string keys "internal" and
            default_output" to formats from parformats.
        metadict: dict
            The dictionary mapping string keys to tuple (v, m) where v is an
            additional argument required by function, and m is a method
            whose string output recreates v when passed to eval().
        base : The basefunction subclass
            The basefunction subclass instance which this one decorates with
            additional functionality.
        Cache : class
            The class (not instance) which implements caching of BaseFunction
            evaluations."""
        BaseFunction.__init__(self, parameterdict, parformats, default_formats, metadict, base, Cache)

    # "Virtual" class methods ####

    # Methods required by BaseFunction ####

    def actualize(
        self,
        pars,
        in_format="default_input",
        free=None,
        removable=False,
        static_owner=False,
    ):
        converted = self.transform_parameters(pars, in_format, out_format="internal")
        return Baseline(self, converted, free, removable, static_owner)

    def getmodule(self):
        return __name__


# end of class BaselineFunction


class Baseline(ModelPart):
    """Represents a baseline associated with a BaselineFunction subclass."""

    def __init__(self, owner, pars, free=None, removable=False, static_owner=False):
        """Initialize the BaselineComponent instance with specified configurations.

            Parameters
            ----------
            owner : BaselineFunction subclass instance
                The owner object which is an instance of a subclass of BaselineFunction.
            pars : array-like
                The sequence of parameters defining the characteristics of the baseline.
            free : Sequence of bool, optional
                The sequence parallel to `pars` where each boolean value indicates whether
                the corresponding parameter is adjustable. If False, that parameter is fixed.
                Defaults to None, implying all parameters are free by default.
            removable : bool, optional
                A flag indicating whether the baseline can be removed during processing.
                Defaults to False.
            static_owner : bool, optional
                Determines if the owner of the baseline can be altered using the
        `       changeowner()` method. Defaults to False.

            Notes
            -----
            - The `free` and `removable` parameters are independent; a baseline can be marked
            as removable even if some of its parameters are fixed (`free` is False). In such
            cases, the baseline may be removed during peak extraction, but the fixed
            parameters will persist until removal.
        """
        ModelPart.__init__(self, owner, pars, free, removable, static_owner)

    @staticmethod
    def factory(baselinestr, ownerlist):
        """Instantiate a Peak from a string.

        Parameters
        ----------
        baselinestr : str
            The string representing Baseline
        ownerlist : array-like
            The list of BaseFunctions that owner is in
        """

        data = baselinestr.strip().splitlines()

        # dictionary of parameters
        pdict = {}
        for d in data:
            result = d.split("=", 1)
            if len(result) == 2:
                try:
                    pdict[result[0]] = eval(result[1])
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

        return Baseline(**pdict)


# End of class Baseline

# simple test code
if __name__ == "__main__":

    from numpy.random import randn

    from diffpy.srmise.modelevaluators.aicc import AICc
    from diffpy.srmise.peaks.base import Peaks
    from diffpy.srmise.peaks.gaussianoverr import GaussianOverR

    res = 0.01
    r = np.arange(2, 4, res)
    err = np.ones(len(r))  # default unknown errors
    pf = GaussianOverR(0.7)
    evaluator = AICc()

    pars = [[3, 0.2, 10], [3.5, 0.2, 10]]
    ideal_peaks = Peaks([pf.actualize(p, "pwa") for p in pars])
    y = ideal_peaks.value(r) + 0.1 * randn(len(r))
