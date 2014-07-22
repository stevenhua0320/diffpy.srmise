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

"""Definition of __version__ and __date__ for SrMise.
"""

__id__ = "$Id: version.py 44 2014-07-12 21:10:58Z luke $"

# obtain version information
from pkg_resources import get_distribution
__version__ = get_distribution('diffpy.srmise').version

# we assume that tag_date was used and __version__ ends in YYYYMMDD
__date__ = __version__[-8:-4] + '-' + \
           __version__[-4:-2] + '-' + __version__[-2:]

# End of file
