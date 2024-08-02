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
"""Controls the logging and plotting options of diffpy.srmise.

By default messages are logged to stdout, but logging to a file is also supported.
SrMiseLog defines five levels of message importance, and all messages at least
as important as the set level are displayed and/or written to the appropriate file.
Levels are specified as non-negative integers or equivalent strings, and are identical
to those found in the Python logging package.

'debug' -> 10
'info' -> 20 (default)
'warning' -> 30
'error' -> 40
'critical' -> 50

Liveplotting plots the value of a model each time it is fit, showing results for
before and after fitting.  User input is optionally required to procede after
fitting.

Functions
---------
addfilelog: Send logging information to a file.
liveplotting: Set whether to use liveplotting, and whether to wait after each liveplot
setfilelevel: Set logging level for the file logger.
setlevel: Set logging level of default logger.
gettracer: Get a TracePeaks instance for tracing peak extraction.
"""

import logging
import os.path
import re

from diffpy.srmise.srmiseerrors import SrMiseDataFormatError, SrMiseFileError, SrMiseLogError

# Default settings ###
defaultformat = "%(message)s"
defaultlevel = logging.INFO

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Set up logging to stdout ###
logger = logging.getLogger("diffpy.srmise")
logger.setLevel(defaultlevel)
ch = logging.StreamHandler()
ch.setLevel(defaultlevel)

formatter = logging.Formatter(defaultformat)
ch.setFormatter(formatter)

logger.addHandler(ch)

# Optional file logger ###
fh = None

# Make updated plots as fitting progresses. ###
liveplots = False
wait = False


def addfilelog(filename, level=defaultlevel, format=defaultformat):
    """Log output from diffpy.srmise in specified file.

    Parameters
    filename: Name of file to receiving output
    level: The logging level
    format: A string defining format of output messages conforming to logging package.
    """
    global fh
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    formatter = logging.Formatter(format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def setfilelevel(level):
    """Set level of file logger.

    Parameters
    level: The logging level."""
    global fh
    if fh is not None:
        level = LEVELS.get(level, level)
        fh.setLevel(level)
        if level < logger.getEffectiveLevel():
            logger.setLevel(level)
    else:
        emsg = "File handler does not exist, cannot set its level."
        raise SrMiseLogError(emsg)


def setlevel(level):
    """Set level of default (stdout) logger.

    Parameters
    level: The logging level."""
    global ch
    level = LEVELS.get(level, level)
    ch.setLevel(level)
    if level < logger.getEffectiveLevel():
        logger.setLevel(level)


def liveplotting(lp, w=False):
    """Set whether or not to use live plotting.

    When using liveplotting, a plot will be shown and updated
    as extraction progresses.

    Parameters
    lp: Use live plotting (True) or not (False).
    w: (False) Whether to wait for user after plotting."""
    global liveplots
    global wait
    if lp is True or lp is False:
        liveplots = lp
    else:
        emsg = "Parameter lp must be a boolean."
        raise ValueError(emsg)

    if w is True or w is False:
        wait = w
    else:
        emsg = "Parameter w must be a boolean."
        raise ValueError(emsg)


# TracePeaks.  Primary purpose is to enable creating movies. ###


class TracePeaks(object):
    """Output trace information during peak extraction."""

    def __init__(self, **kwds):

        self.__filter = None
        self.filter = kwds.get("filter", ["False"])

        self.filebase = kwds.get("filebase", None)
        self.store = kwds.get("store", False)
        self.trace = []
        self.counter = None
        self.recursion = None
        self.call = None
        self.reset_trace()

        if self.filebase:
            dir = os.path.dirname(self.filebase)
            if dir and not os.path.exists(dir):
                os.makedirs(dir)

        return

    def emit(self, *args, **kwds):
        """Write current trace to file.

        Parameters
        Any number of ModelCluster instances"""
        if not eval(self.filter):
            return
        else:
            trace = self.maketrace(*args, **kwds)
            if self.filebase:
                self.write(trace)
            if self.store:
                self.trace.append(trace)
            self.counter += 1
            return

    def maketrace(self, *args, **kwds):
        """Return dictionary of trace properties.

        Keywords
        model - Use specified model (Peaks instance) instead of those in args.
        """
        mc = args[0].copy()
        mc.slice = None
        mc.change_slice(slice(len(mc.r_data)))
        clusters = []
        for m in args:
            clusters.append(m.slice)
        for m in args[1:]:
            mc.replacepeaks(m.model)
        return {
            "mc": mc,
            "clusters": clusters,
            "recursion": self.recursion,
            "counter": self.counter,
        }

    def writestr(self, trace):
        """Return string representation of current trace."""
        lines = []
        lines.append("### Trace")
        lines.append("counter=%i" % trace["counter"])
        lines.append("recursion=%i" % trace["recursion"])
        lines.append("clusters=%s" % trace["clusters"])
        lines.append("### ModelCluster")

        lines.append(trace["mc"].writestr())
        return "\n".join(lines)

    def write(self, trace):
        """Write current trace to file."""
        filename = "%s_%i" % (self.filebase, trace["counter"])
        f = open(filename, "w")
        bytes = self.writestr(trace)
        f.write(bytes)
        f.close()

    def read(self, filename):
        """Read tracer ModelCluster from file.

        Parameters
        filename - file from which to read

        Returns dictionary with keys
        "clusters" - List of cluster regions [[r0,r1],[r2,r3],...]
        "counter" - The count when object was created
        "mc" - A ModelCluster instance
        "recursion" - The recursion level of mc"""
        try:
            return self.readstr(open(filename, "rb").read())
        except SrMiseDataFormatError as err:
            logger.exception("")
            basename = os.path.basename(filename)
            emsg = ("Could not open '%s' due to unsupported file format " + "or corrupted data. [%s]") % (
                basename,
                err,
            )
            raise SrMiseFileError(emsg)

    def readstr(self, datastring):
        """Read tracer ModelCluster from string.

        Parameters
        datastring - The string representation of a trace

        Returns dictionary with keys
        "clusters" - List of cluster regions [[r0,r1],[r2,r3],...]
        "counter" - The count when object was created
        "mc" - A ModelCluster instance
        "recursion" - The recursion level of mc"""

        # find where the ModelCluster section starts
        res = re.search(r"^#+ ModelCluster\s*(?:#.*\s+)*", datastring, re.M)
        if res:
            header = datastring[: res.start()]
            mc = datastring[res.end() :].strip()
        else:
            emsg = "Required section 'ModelCluster' not found."
            raise SrMiseDataFormatError(emsg)

        # instantiate ModelCluster
        if re.match(r"^None$", mc):
            mc = None
        else:
            from diffpy.srmise.modelcluster import ModelCluster

            mc = ModelCluster.factory(mc)

        res = re.search(r"^clusters=(.*)$", header, re.M)
        if res:
            clusters = eval(res.groups()[0].strip())
        else:
            emsg = "Required field 'clusters' not found."
            raise SrMiseDataFormatError(emsg)

        res = re.search(r"^recursion=(.*)$", header, re.M)
        if res:
            eval(res.groups()[0].strip())
        else:
            emsg = "Required field 'recursion' not found."
            raise SrMiseDataFormatError(emsg)

        res = re.search(r"^counter=(.*)$", header, re.M)
        if res:
            eval(res.groups()[0].strip())
        else:
            emsg = "Required field 'counter' not found."
            raise SrMiseDataFormatError(emsg)

        return {
            "mc": mc,
            "clusters": clusters,
            "recursion": self.recursion,
            "counter": self.counter,
        }

    def pushr(self):
        """Enter a layer of recursion, and return new level."""
        self.recursion += 1
        return self.recursion

    def popr(self):
        """Exit a layer of recursion, and return new level."""
        self.recursion -= 1
        return self.recursion

    def pushc(self):
        """Enter a new tracer-aware function."""
        if self.call == 0:
            self.reset_trace()
        self.call += 1
        return self.call

    def popc(self):
        """Exit a tracer-aware function."""
        self.call -= 1
        return self.call

    def reset_trace(self):
        self.call = 0
        self.counter = 0
        self.recursion = 0
        self.stored = []

    # filter property
    def setfilter(self, filter):
        self.__filter = compile(" and ".join(["(%s)" % f for f in filter]), "<string>", "eval")

    def getfilter(self):
        return self.__filter

    filter = property(getfilter, setfilter)


# End of class TracePeaks


def settracer(**kwds):
    global tracer
    tracer = TracePeaks(**kwds)
    return tracer


# Default tracer never emits
tracer = settracer()
