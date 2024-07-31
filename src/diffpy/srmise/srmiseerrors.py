#!/usr/bin/env python
##############################################################################
#
# SrMise            by Luke Granlund
#                   (c) 2014 trustees of the Michigan State University
#                   (c) 2024 trustees of Columia University in the City of New York
#                   All rights reserved.
#
# File coded by:    Luke Granlund
#
# See LICENSE.txt for license information.
#
##############################################################################
"""Defines all custom exceptions used by diffpy.srmise.

   Classes
   -------
   SrMiseError: Subclass of Exception, and superclass of all diffpy.srmise exceptions.
   SrMiseDataFormatError: Error in format of diffpy.srmise data.
   SrMiseEstimationError: Parameter estimation error.
   SrMiseFileError: Error while reading/writing files.
   SrMiseFitError: Error while fitting.
   SrMiseLogError: Error while logging.
   SrMiseModelEvaluatorError: Error while computing or comparing model quality.
   SrMisePDFKeyError: Error in key referencing component of PDF dataset.
   SrMiseQmaxError: Error in value of Qmax.
   SrMiseScalingError: Error while scaling a peak function.
   SrMiseStaticOwnerError: Error when changing ModelPart instance owner.
   """


# Superclass class for diffpy.srmise.mise
class SrMiseError(Exception):
    """Superclass of all diffpy.srmise exceptions."""

    def __init__(self, info):
        """initialize

        info: description string"""
        Exception.__init__(self)
        self.info = info

    def __str__(self):
        return self.info


# SrMiseError subclasses ###


class SrMiseDataFormatError(SrMiseError):
    """diffpy.srmise exception class.  Error in formatted data."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseEstimationError(SrMiseError):
    """diffpy.srmise.modelevaluator exception class.  Parameter estimation error."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseFileError(SrMiseError):
    """diffpy.srmise exception class.  Error while reading/writing files."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseFitError(SrMiseError):
    """diffpy.srmise exception class.  Error occurred during fitting."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseLogError(SrMiseError):
    """diffpy.srmise exception class.  Error while handling logging capabilities."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseModelEvaluatorError(SrMiseError):
    """diffpy.srmise.modelevaluator exception class.  Error when comparing models."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseQmaxError(SrMiseError):
    """diffpy.srmise.modelevaluator exception class.  Error when setting qmax."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseScalingError(SrMiseError):
    """diffpy.srmise.peaks exception class.  Error when scaling a peak function."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseStaticOwnerError(SrMiseError):
    """diffpy.srmise exception class.  Attempt to change owner of static model part."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseTransformationError(SrMiseError):
    """diffpy.srmise exception class.  Error transforming model/covariance parameters."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMiseUndefinedCovarianceError(SrMiseError):
    """diffpy.srmise exception class.  Attempted to perform on undefined covariance."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)


class SrMisePDFKeyError(SrMiseError):
    """diffpy.srmise exception class.  Requested PDF key can't be found."""

    def __init__(self, info):
        """initialize

        info -- description string"""
        SrMiseError.__init__(self, info)
