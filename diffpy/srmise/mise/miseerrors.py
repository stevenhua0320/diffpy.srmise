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
"""Defines all custom exceptions used by diffpy.srmise.mise.

   Classes
   -------
   MiseError: Subclass of Exception, and superclass of all diffpy.srmise.mise exceptions.
   MiseDataFormatError: Error in format of diffpy.srmise data.
   MiseEstimationError: Parameter estimation error.
   MiseFileError: Error while reading/writing files.
   MiseFitError: Error while fitting.
   MiseLogError: Error while logging.
   MiseModelEvaluatorError: Error while computing or comparing model quality.
   MiseScalingError: Error while scaling a peak function.
   MiseStaticOwnerError: Error when changing ModelPart instance owner.
   """

### Superclass class for diffpy.srmise.mise
class MiseError(Exception):
    """Superclass of all diffpy.srmise.mise exceptions."""
    def __init__(self, info):
        """initialize

        info: description string"""
        Exception.__init__(self)
        self.info = info

    def __str__(self):
        return self.info


### MiseError subclasses ###

class MiseDataFormatError(MiseError):
    """diffpy.srmise.mise exception class.  Error in formatted data."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)


class MiseEstimationError(MiseError):
    """diffpy.srmise.mise.modelevaluator exception class.  Parameter estimation error."""
    def __init__(self, info):
       """initialize

       info -- description string"""
       MiseError.__init__(self, info)


class MiseFileError(MiseError):
    """diffpy.srmise.mise exception class.  Error while reading/writing files."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)


class MiseFitError(MiseError):
    """diffpy.srmise.mise exception class.  Error occurred during fitting."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)


class MiseLogError(MiseError):
    """diffpy.srmise.mise exception class.  Error while handling logging capabilities."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)


class MiseModelEvaluatorError(MiseError):
    """diffpy.srmise.mise.modelevaluator exception class.  Error when comparing models."""
    def __init__(self, info):
       """initialize

       info -- description string"""
       MiseError.__init__(self, info)


class MiseScalingError(MiseError):
    """diffpy.srmise.mise.peaks exception class.  Error when scaling a peak function."""
    def __init__(self, info):
       """initialize

       info -- description string"""
       MiseError.__init__(self, info)


class MiseStaticOwnerError(MiseError):
    """diffpy.srmise.mise exception class.  Attempt to change owner of static model part."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)

class MiseTransformationError(MiseError):
    """diffpy.srmise.mise exception class.  Error transforming model/covariance parameters."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)

class MiseUndefinedCovarianceError(MiseError):
    """diffpy.srmise.mise exception class.  Attempted to perform on undefined covariance."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)
        
class MisePDFKeyError(MiseError):
    """diffpy.srmise.mise exception class.  Requested PDF key can't be found."""
    def __init__(self, info):
        """initialize

        info -- description string"""
        MiseError.__init__(self, info)
