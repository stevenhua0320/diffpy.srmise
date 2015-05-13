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
#
# Routines for analyzing and comparing the quality of models to the atomic
# pair distribution function.  The total number of intrinsic peaks in the PDF
# is on the order of the number of atoms in the sample squared, and these
# overlap, so developing a model to the entire PDF is a terribly
# underconstrained problem.  There are two primary considerations to make when
# interpreting peaks extracted from the PDF as significantly favored
# interatomic distances (i.e. few or no other interatomic distances appear
# between the maxima of an extracted peak and its neighbors):
# 1. The more ordered the system the more likely this interpretation is to be
#    valid.  In contrast, it is not appropriate to interpret peaks extracted
#    from the PDF of amorphous structures in this way.
# 2. The number of overlapping peaks increases roughly as r^2, so peaks
#    extracted at low r are more likely to correspond with this interpretation
#    than those at high r.
# Several information theoretic methods are provided that penalize
# overfitting.
#
# Errors in the PDF are correlated within a short range, but at present all
# data points are considered independently distributed.  Any provided errors
# on the PDF are also not yet considered.
#
# Model selection criteria:
#    Akaike information criterion (AIC)
#    Akaike information criterion w/ small sample correction (AICc)
#
#
########################################################################

import numpy as np
from diffpy.srmise.srmiseerrors import SrMiseModelEvaluatorError

import logging, diffpy.srmise.srmiselog
logger = logging.getLogger("diffpy.srmise")

class ModelEvaluator:
    """Class for evaluating the quality of a fit.  Comparison between different
       models of the same type is defined so that better models are 'greater than'
       worse models."""

    def __init__(self, method, higher_is_better):
        """method = name of method (string)
           higher_is_better = boolean
        """
        self.method = method
        self.higher_is_better = higher_is_better
        self.stat = None
        self.chisq = None
        return

    def __lt__(self, other):
        """   """

        assert self.method == other.method #Comparison between same types required
        assert self.stat != None and other.stat != None #The statistic must already be calculated

        if self.higher_is_better:
            return self.stat < other.stat
        else:
            return other.stat < self.stat

    def __le__(self, other):
        """   """

        assert self.method == other.method #Comparison between same types required
        assert self.stat != None and other.stat != None #The statistic must already be calculated

        if self.higher_is_better:
            return self.stat <= other.stat
        else:
            return other.stat <= self.stat

    def __eq__(self, other):
        """   """

        assert self.method == other.method #Comparison between same types required
        assert self.stat != None and other.stat != None #The statistic must already be calculated

        return self.stat == other.stat

    def __ne__(self, other):
        """   """

        assert self.method == other.method #Comparison between same types required
        assert self.stat != None and other.stat != None #The statistic must already be calculated

        return self.stat != other.stat

    def __gt__(self, other):
        """   """

        assert self.method == other.method #Comparison between same types required
        assert self.stat != None and other.stat != None #The statistic must already be calculated

        if self.higher_is_better:
            return self.stat > other.stat
        else:
            return other.stat > self.stat

    def __ge__(self, other):
        """   """

        assert self.method == other.method #Comparison between same types required
        assert self.stat != None and other.stat != None #The statistic must already be calculated

        if self.higher_is_better:
            return self.stat >= other.stat
        else:
            return other.stat >= self.stat


    def chi_squared(self, expected, observed, error):
        """Calculates chi-squared statistic."""

        self.chisq = np.sum((expected-observed)**2/error**2)
        return self.chisq

# end of class ModelEvaluator
