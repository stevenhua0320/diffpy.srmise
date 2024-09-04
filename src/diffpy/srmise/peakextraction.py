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

import logging
import os.path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from diffpy.srmise import srmiselog
from diffpy.srmise.baselines.base import Baseline
from diffpy.srmise.dataclusters import DataClusters
from diffpy.srmise.modelcluster import ModelCluster, ModelCovariance
from diffpy.srmise.peaks.base import Peak, Peaks
from diffpy.srmise.srmiseerrors import SrMiseDataFormatError, SrMiseEstimationError, SrMiseFileError

logger = logging.getLogger("diffpy.srmise")


class PeakExtraction(object):
    """Class for peak extraction.

    Parameters
    ----------
    x : array-like
        The x coordinates of the data
    y : array-like
        The y coordinates of the data
    dx : array-like
        The uncertainties in the x coordinates (not used)
    dy : array-like
        The uncertainties in the y coordinates
    effective_dy : array-like
        The uncertainties in the y coordinates actually used during extraction
    rng : list
        The [xmin, xmax] Range of x coordinates over which to extract peaks
    pf : array-like
        The sequence of peak functions that can be extracted
    initial_peaks: Peaks object
        The peaks present at start of extraction
    baseline : Baseline object
        The baseline for data
    cres : float
        The resolution of clustering
    error_method : ErrorEvaluator class
        The Evaluation class used to compare models

    Calculated members
    ------------------
    extracted : ModelCluster object
        The ModelCluster object after extraction
    extraction_type : Type of extraction
    """

    def __init__(self, newvars=[]):
        """Initialize PeakExtraction object.

        Parameters
        newvars : array-like
            Sequence of strings that represent additional extraction parameters."""
        self.clear()
        self.extractvars = dict.fromkeys(
            (
                "effective_dy",
                "rng",
                "pf",
                "initial_peaks",
                "baseline",
                "cres",
                "error_method",
            )
        )
        for k in newvars:
            if k not in self.extractvars:
                self.extractvars[k] = None
            else:
                emsg = "Extraction variable %s conflicts with existing variable" % k
                raise ValueError(emsg)
        return

    def clear(self):
        """Clear all members.

        The purpose of the method is to ensure the object is in initialized state."""
        self.x = None
        self.y = None
        self.dx = None
        self.dy = None
        self.effective_dy = None
        self.cres = None
        self.pf = None
        self.baseline = None
        self.error_method = None
        self.initial_peaks = None
        self.rng = None
        self.clearcalc()

    def clearcalc(self):
        """Clear all calculated members."""
        self.extracted = None
        self.extraction_type = None

    def setdata(self, x, y, dx=None, dy=None):
        if len(x) != len(y):
            emsg = "Sequences x and y must have the same length."
            raise ValueError(emsg)
        self.x = np.array(x)
        self.y = np.array(y)
        if dx is None:
            self.dx = np.zeros(len(x))
        else:
            self.dx = np.array(dx)
        if dy is None:
            self.dy = np.zeros(len(x))
        else:
            self.dy = np.array(dy)
        if len(self.x) != len(self.dx) or len(self.x) != len(self.dy):
            emsg = "Sequences dx and dy (if present) must have the same length as x"
            raise ValueError(emsg)
        # self.defaultvars()
        return

    def setvars(self, quiet=False, **kwds):
        """Set one or more extraction variables.

        Parameters
        ----------
        quiet : bool
            The log changes quietly. Default is False.
        cres : float
            The clustering resolution, must be > 0.
        effective_dy : array-like
            The uncertainties actually used during extraction
        pf : list
            The sequence of PeakFunctionBase subclass instances.
        baseline : Baseline instance or BaselineFunction instance
            The Baseline instance or BaselineFunction instance that use built-in estimation
        error_method : ErrorEvaluator subclass instance
            The ErrorEvaluator subclass instance used to compare models. Default is AIC.
        initial_peaks : Peaks instance
            These peaks are present at the start of extraction.
        rng : array-like
            The sequence specifying the least and greatest x-values over which to extract peaks.
        """
        for k, v in kwds.items():
            if k in self.extractvars:
                if quiet:
                    logger.debug("Setting variable %s=%s", k, v)
                else:
                    logger.info("Setting variable %s=%s", k, v)
                setattr(self, k, v)
            else:
                emsg = "Invalid extraction variable: %s=%s" % (k, v)
                raise ValueError(emsg)
        self.defaultvars()

    def defaultvars(self, *args):
        """Set unset(=None) extraction variables to default values.

        Certain variables may be partially set for convenience, and are transformed
        appropriately. See 'Default values assigned' below.

        Parameters
        ----------
        *args : str
                The variable argument list where each string corresponds to an extraction
                variable name.

        Default values assigned:
        - `cres` : 4 times the average spacing between elements in `x`.
        - `effective_dy` : If all elements in `y` are positive, it's set to the data `dy`;
                           otherwise, it's 5% of the range (`max(y)` - `min(y)`). If `effective_dy`
                           is a positive scalar, an array of that value with a length matching `y` is used.
        - `pf` : A list containing a single Gaussian overlap function with the maximum width
                spanning the entire `x` range (`x[-1] - x[0]`).
        - `baseline` : A flat baseline at `y=0`, indicating no background signal.
        - `error_method` : Uses the AIC (Akaike Information Criterion) for evaluating model fits.
        - `initial_peaks` : Assumes no initial peak guesses, implying peaks will be detected from scratch.
        - `rng` : The default range is set to span the entire `x` dataset, i.e., `[x[0], x[-1]]`.
                 If a range is partially defined, e.g., `[None, 100.]`, the `None` value is replaced
                with the respective boundary of the `x` data.

        Note that the default values of very important parameters like the uncertainty
        and clustering resolution are crude guesses at best.
        """
        if self.cres is None or "cres" in args:
            self.cres = 4 * (self.x[-1] - self.x[0]) / len(self.x)

        if self.effective_dy is None or "effective_dy" in args:
            if np.all(self.dy > 0):
                # That is, all points positive uncertainty.
                self.effective_dy = self.dy
            else:
                # A terribly crude guess
                self.effective_dy = 0.05 * (np.max(self.y) - np.min(self.y)) * np.ones(len(self.x))
        elif np.isscalar(self.effective_dy) and self.effective_dy > 0:
            self.effective_dy = self.effective_dy * np.ones(len(self.x))

        if self.pf is None or "pf" in args:
            from diffpy.srmise.peaks.gaussianoverr import GaussianOverR

            # TODO: Make a more useful default.
            self.pf = [GaussianOverR(self.x[-1] - self.x[0])]

        if self.rng is None or "rng" in args:
            self.rng = [self.x[0], self.x[-1]]
        else:
            if self.rng[0] is None:
                self.rng[0] = self.x[0]
            if self.rng[1] is None:
                self.rng[1] = self.x[-1]

        # Set baseline where the type is given, but parameters must be estimated.
        if hasattr(self.baseline, "estimate_parameters"):
            try:
                s = self.getrangeslice()
                epars = self.baseline.estimate_parameters(self.x[s], self.y[s])
                self.baseline = self.baseline.actualize(epars, "internal")
                logger.info("Estimating baseline: %s" % self.baseline)
            except (NotImplementedError, SrMiseEstimationError):
                logger.error("Could not estimate baseline from provided BaselineFunction, trying default values.")
                self.baseline = None

        if self.baseline is None or "baseline" in args:
            from diffpy.srmise.baselines.polynomial import Polynomial

            bl = Polynomial(degree=-1)
            self.baseline = bl.actualize(np.array([]), "internal")

        if self.error_method is None or "error_method" in args:
            from diffpy.srmise.modelevaluators.aic import AIC

            self.error_method = AIC

        if self.initial_peaks is None or "initial_peaks" in args:
            self.initial_peaks = Peaks()

    def __str__(self):
        """Return string summary of PeakExtraction."""
        out = []
        for k in self.extractvars:
            out.append("%s: %s" % (k, getattr(self, k)))
        if self.extracted is not None:
            out.append("Extraction type: %s" % self.extraction_type)
            out.append("--- Extracted ---")
            out.append(str(self.extracted))
        else:
            out.append("No extracted peaks exist.")

        return "\n".join(out) + "\n"

    def plot(self, **kwds):
        """Convenience function to plot data and extracted peaks with matplotlib.

        Uses initial peaks instead if no peaks have been extracted.

        Takes same keywords as ModelCluster.plottable()

        Parameters
        ----------
        **kwds :args
            The keyword arguments to pass to matplotlib.
        """
        plt.clf()
        if self.extracted is not None:
            plt.plot(*self.extracted.plottable(kwds))
        else:

            # Make sure all required extraction variables have some value
            self.defaultvars()

            rangeslice = self.getrangeslice()
            x = self.x[rangeslice]
            y = self.y[rangeslice]
            dy = self.dy[rangeslice]
            mcluster = ModelCluster(
                self.initial_peaks,
                self.baseline,
                x,
                y,
                dy,
                None,
                self.error_method,
                self.pf,
            )
            plt.plot(*mcluster.plottable(kwds))

    def read(self, filename):
        """load PeakExtraction object from file

        Parameters
        ----------
        filename : str
            The file from which to read

        Returns
        -------
        self
        """
        try:
            self.readstr(open(filename, "rb").read())
        except SrMiseDataFormatError as err:
            logger.exception("")
            basename = os.path.basename(filename)
            emsg = ("Could not open '%s' due to unsupported file format " + "or corrupted data. [%s]") % (
                basename,
                err,
            )
            raise SrMiseFileError(emsg)
        return self

    def readstr(self, datastring):
        """Initialize members from string.

        Parameters
        ----------
        datastring : array-like
            The raw data to read
        """
        from diffpy.srmise.basefunction import BaseFunction

        self.clear()

        # The major components are:
        # - Header
        # - BaselineFunctions
        # - PeakFunctions
        # - BaselineObject
        # - InitialPeaks
        # - SrMiseMetaData
        # - MetaData
        # - StartData
        # - Results

        # Lists holding BaseFunctions as they are instantiated
        safepf = []
        safebf = []

        # find where the results section starts
        res = re.search(r"^#+ Results\s*(?:#.*\s+)*", datastring, re.M)
        if res:
            results = datastring[res.end() :].strip()
            header = datastring[: res.start()]

        # find data section, and what information it contains
        res = re.search(r"^#+ start data\s*(?:#.*\s+)*", header, re.M)
        if res:
            start_data = header[res.end() :].strip()
            start_data_info = header[res.start() : res.end()]
            header = header[: res.start()]
        res = re.search(r"^(#+L.*)$", start_data_info, re.M)
        if res:
            start_data_info = start_data_info[res.start() : res.end()].strip()
        hasx = False
        hasy = False
        hasdx = False
        hasdy = False
        hasedy = False
        res = re.search(r"\bx\b", start_data_info)
        if res:
            hasx = True
        res = re.search(r"\by\b", start_data_info)
        if res:
            hasy = True
        res = re.search(r"\bdx\b", start_data_info)
        if res:
            hasdx = True
        res = re.search(r"\bdy\b", start_data_info)
        if res:
            hasdy = True
        res = re.search(r"\edy\b", start_data_info)
        if res:
            hasedy = True

        res = re.search(r"^#+ Metadata\s*(?:#.*\s+)*", header, re.M)
        if res:
            metadata = header[res.end() :].strip()
            header = header[: res.start()]

        res = re.search(r"^#+ SrMiseMetadata\s*(?:#.*\s+)*", header, re.M)
        if res:
            srmisemetadata = header[res.end() :].strip()
            header = header[: res.start()]

        res = re.search(r"^#+ InitialPeaks.*$", header, re.M)
        if res:
            initial_peaks = header[res.end() :].strip()
            header = header[: res.start()]

        res = re.search(r"^#+ BaselineObject\s*(?:#.*\s+)*", header, re.M)
        if res:
            baselineobject = header[res.end() :].strip()
            header = header[: res.start()]

        res = re.search(r"^#+ PeakFunctions.*$", header, re.M)
        if res:
            peakfunctions = header[res.end() :].strip()
            header = header[: res.start()]

        res = re.search(r"^#+ BaselineFunctions.*$", header, re.M)
        if res:
            baselinefunctions = header[res.end() :].strip()
            header = header[: res.start()]

        #  Instantiating baseline functions
        res = re.split(r"(?m)^#+ BaselineFunction \d+\s*(?:#.*\s+)*", baselinefunctions)
        for s in res[1:]:
            safebf.append(BaseFunction.factory(s, safebf))

        #  Instantiating peak functions
        res = re.split(r"(?m)^#+ PeakFunction \d+\s*(?:#.*\s+)*", peakfunctions)
        for s in res[1:]:
            safepf.append(BaseFunction.factory(s, safepf))

        #  Instantiating Baseline object
        if re.match(r"^None$", baselineobject):
            self.baseline = None
        elif re.match(r"^\d+$", baselineobject):
            self.baseline = safebf[int(baselineobject)]
        else:
            self.baseline = Baseline.factory(baselineobject, safebf)

        #  Instantiating initial peaks
        if re.match(r"^None$", initial_peaks):
            self.initial_peaks = None
        else:
            self.initial_peaks = Peaks()
            res = re.split(r"(?m)^#+ InitialPeak\s*(?:#.*\s+)*", initial_peaks)
            for s in res[1:]:
                self.initial_peaks.append(Peak.factory(s, safepf))

        #  Instantiating srmise metatdata

        # pf
        res = re.search(r"^pf=(.*)$", srmisemetadata, re.M)
        self.pf = eval(res.groups()[0].strip())
        if self.pf is not None:
            self.pf = [safepf[i] for i in self.pf]

        # cres
        rx = {"f": r"[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?"}
        regexp = r"\bcres *= *(%(f)s)\b" % rx
        res = re.search(regexp, srmisemetadata, re.I)
        self.cres = float(res.groups()[0])

        # error_method
        res = re.search(r"^ModelEvaluator=(.*)$", srmisemetadata, re.M)
        __import__("diffpy.srmise.modelevaluators")
        module = sys.modules["diffpy.srmise.modelevaluators"]
        self.error_method = getattr(module, res.groups()[0].strip())

        # range
        res = re.search(r"^Range=(.*)$", srmisemetadata, re.M)
        self.rng = eval(res.groups()[0].strip())

        #  Instantiating other metadata
        self.readmetadata(metadata)

        #  Instantiating start data
        # read actual data - x, y, dx, dy, plus effective_dy
        arrays = []
        if hasx:
            self.x = []
            arrays.append(self.x)
        else:
            self.x = None
        if hasy:
            self.y = []
            arrays.append(self.y)
        else:
            self.y = None
        if hasdx:
            self.dx = []
            arrays.append(self.dx)
        else:
            self.dx = None
        if hasdy:
            self.dy = []
            arrays.append(self.dy)
        else:
            self.dy = None
        if hasedy:
            self.effective_dy = []
            arrays.append(self.effective_dy)
        else:
            self.effective_dy = None
        # raise SrMiseDataFormatError if something goes wrong
        try:
            for line in start_data.split("\n"):
                split_line = line.split()
                if len(arrays) != len(split_line):
                    emsg = "Number of value fields does not match that given by '%s'" % start_data_info
                for a, v in zip(arrays, line.split()):
                    a.append(float(v))
        except (ValueError, IndexError) as err:
            raise SrMiseDataFormatError(str(err))
        if hasx:
            self.x = np.array(self.x)
        if hasy:
            self.y = np.array(self.y)
        if hasdx:
            self.dx = np.array(self.dx)
        if hasdy:
            self.dy = np.array(self.dy)
        if hasedy:
            self.effective_dy = np.array(self.effective_dy)

        #  Instantiating results
        res = re.search(r"^#+ ModelCluster\s*(?:#.*\s+)*", results, re.M)
        if res:
            mc = results[res.end() :].strip()
            results = results[: res.start()]

        # extraction type
        res = re.search(r"^extraction_type=(.*)$", results, re.M)
        if res:
            self.extraction_type = eval(res.groups()[0].strip())
        else:
            emsg = "Required field 'extraction_type' not found."
            raise SrMiseDataFormatError(emsg)

        # extracted
        if re.match(r"^None$", mc):
            self.extracted = None
        else:
            self.extracted = ModelCluster.factory(mc, pfbaselist=safepf, blfbaselist=safebf)

    def write(self, filename):
        """Write string representation of PeakExtraction instance to file.

        Parameters
        ----------
        filename : str
            The name of the file to write
        """
        bytes = self.writestr()
        f = open(filename, "w")
        f.write(bytes)
        f.close()
        return

    def writestr(self):
        """Return string representation of PeakExtraction object.

        Returns
        -------
        The str representation of PeakExtraction object
        """
        import time
        from getpass import getuser

        from diffpy.srmise import __version__
        from diffpy.srmise.basefunction import BaseFunction

        lines = []

        # Header
        lines.extend(
            [
                "History written: " + time.ctime(),
                "produced by " + getuser(),
                "diffpy.srmise version %s" % __version__,
                "##### PDF Peak Extraction",
            ]
        )

        # Generate list of PeakFunctions and BaselineFunctions
        # so I can refer to them by index when necessary.
        allpf = []
        allbf = []
        if self.pf is not None:
            allpf.extend(self.pf)
        if self.initial_peaks is not None:
            allpf.extend([i.owner() for i in self.initial_peaks])
        if self.baseline is not None:
            if isinstance(self.baseline, BaseFunction):
                allbf.append(self.baseline)
            else:  # should be a ModelPart
                allbf.append(self.baseline.owner())
        if self.extracted is not None:
            allpf.extend(self.extracted.peak_funcs)
            allpf.extend([p.owner() for p in self.extracted.model])
            if self.extracted.baseline is not None:
                allbf.append(self.extracted.baseline.owner())
        allpf = list(set(allpf))
        allbf = list(set(allbf))
        safepf = BaseFunction.safefunctionlist(allpf)
        safebf = BaseFunction.safefunctionlist(allbf)

        # Indexed baseline functions
        lines.append("## BaselineFunctions")
        for i, bf in enumerate(safebf):
            lines.append("# BaselineFunction %s" % i)
            lines.append(bf.writestr(safebf))

        # Indexed peak functions
        lines.append("## PeakFunctions")
        for i, pf in enumerate(safepf):
            lines.append("# PeakFunction %s" % i)
            lines.append(pf.writestr(safepf))

        # Baseline
        lines.append("# BaselineObject")
        if self.baseline is None:
            lines.append("None")
        elif self.baseline in safebf:
            lines.append("%s" % repr(safebf.index(self.baseline)))
        else:
            lines.append(self.baseline.writestr(safebf))

        # Initial peaks
        lines.append("## InitialPeaks")
        if self.initial_peaks is None:
            lines.append("None")
        else:
            for ip in self.initial_peaks:
                lines.append("# InitialPeak")
                lines.append(ip.writestr(safepf))

        lines.append("# SrMiseMetadata")

        # Extractable peak types
        if self.pf is None:
            lines.append("pf=None")
        else:
            lines.append("pf=%s" % repr([safepf.index(p) for p in self.pf]))

        # Clustering resolution
        lines.append("cres=%g" % self.cres)
        # Model evaluator
        if self.error_method is None:
            lines.append("ModelEvaluator=None")
        else:
            lines.append("ModelEvaluator=%s" % self.error_method.__name__)
        # Extraction range
        lines.append("Range=%s" % repr(self.rng))

        # Everything not defined by PeakExtraction
        lines.append("# Metadata")
        lines.append(self.writemetadata())

        # Raw data used in extraction.
        lines.append("##### start data")
        line = ["#L"]
        numlines = 0
        if self.x is not None:
            line.append("x")
            numlines = len(self.x)
        if self.y is not None:
            line.append("y")
            numlines = len(self.y)
        if self.dx is not None:
            line.append("dx")
            numlines = len(self.dx)
        if self.dy is not None:
            line.append("dy")
            numlines = len(self.dy)
        if self.effective_dy is not None:
            line.append("edy")
            numlines = len(self.effective_dy)
        lines.append(" ".join(line))
        for i in range(numlines):
            line = []
            if self.x is not None:
                line.append("%g" % self.x[i])
            if self.y is not None:
                line.append("%g" % self.y[i])
            if self.dx is not None:
                line.append("%g" % self.dx[i])
            if self.dy is not None:
                line.append("%g" % self.dy[i])
            if self.effective_dy is not None:
                line.append("%g" % self.effective_dy[i])
            lines.append(" ".join(line))

        #  Calculated members
        lines.append("##### Results")
        lines.append("extraction_type=%s" % repr(self.extraction_type))

        lines.append("### ModelCluster")
        if self.extracted is None:
            lines.append("None")
        else:
            lines.append(self.extracted.writestr(pfbaselist=safepf, blfbaselist=safebf))

        datastring = "\n".join(lines) + "\n"
        return datastring

    def writemetadata(self):
        """Return string for metadata not defined by srmise class."""
        return

    def readmetadata(self):
        """Return string for metadata not defined by srmise class."""
        return

    def writesummary(self):
        """Return summary of peak extraction results."""
        pass

    def getrangeslice(self):
        """Convert the ranges in terms of x-coordinates to a slice object."""
        low_idx = 0
        while self.x[low_idx] < max(self.x[0], self.rng[0]):
            low_idx += 1
        hi_idx = len(self.x)
        while self.x[hi_idx - 1] > min(self.x[-1], self.rng[1]):
            hi_idx -= 1
        return slice(low_idx, hi_idx)

    def estimate_peak(self, x, add=True):
        """Return new estimated peak near x.

        Peaks already extracted, if any, are taken into account.  If none exist,
        use those specified by initial_peaks instead.

        Parameters
        ----------
        x : array-like
            The oordinate of the point of interest
        add : bool
            Automatically add peak to extracted peaks or initial_peaks. Default is True.

        Returns
        -------
        The Peak object, or None if estimation fails.
        """
        # Make sure all required extraction variables have some value
        self.defaultvars()

        if self.extracted is not None:
            # Determine clusters using existing peaks and baseline in extracted
            x1 = self.extracted.r_cluster
            y1 = self.extracted.y_cluster - self.extracted.value()
            dy = self.extracted.error_cluster
        else:
            # Determine clusters using initial_peaks and pre-defined or estimated baseline
            rangeslice = self.getrangeslice()
            x1 = self.x[rangeslice]
            y1 = self.y[rangeslice] - self.baseline.value(x1) - self.initial_peaks.value(x1)
            dy = self.effective_dy[rangeslice]

        if x < x1[0] or x > x1[-1]:
            emsg = "Argument x=%s outside allowed range (%s, %s)." % (x, x1[0], x1[-1])
            raise ValueError(emsg)

        # Object performing clustering on data. Note that DataClusters
        # provides an iterator that clusters the next point and returns
        # itself. Thus, dclusters and step (below) refer to the same object.

        dclusters = DataClusters(x1, y1, self.cres)  # Cluster with baseline removed
        dclusters.makeclusters()
        cidx = dclusters.find_nearest_cluster2(x)[0]
        cslice = dclusters.cut(cidx)

        x1 = x1[cslice]
        y1 = y1[cslice]
        dy = dy[cslice]

        mcluster = ModelCluster(None, None, x1, y1, dy, None, self.error_method, self.pf)
        mcluster.fit()

        if len(mcluster.model) > 0:
            if add:
                logger.info("Adding peak: %s" % mcluster.model[0])
                self.add_peaks(mcluster.model)
            else:
                logger.info("Found peak: %s" % mcluster.model[0])
            return mcluster.model[0]
        else:
            logger.info("No peaks found.")
            return None

    def add_peaks(self, peaks):
        """Add peaks to extracted peaks, or initial_peaks if no extracted peaks exist.

        Parameters
        ----------
        peaks: Peaks object
            The peaks instance
        """
        if self.extracted is not None:
            self.extracted.replacepeaks(peaks)
        else:
            if self.initial_peaks is None:
                self.setvars("initial_peaks")
            self.initial_peaks.extend(peaks)
            self.initial_peaks.sort(key="position")

    def extract_single(self, recursion_depth=1):
        """Find ModelCluster with peaks extracted from data. Return ModelCovariance instance at top level.

        Every extracted peak is one of the peak functions supplied. All
        comparisons of different peak models are performed with the class
        specified by error_method.

        Parameters
        recursion_depth: (1) Tracks recursion with extract_single."""
        self.clearcalc()
        tracer = srmiselog.tracer
        tracer.pushc()
        tracer.pushr()

        # Make sure all required extraction variables have some value
        self.defaultvars()
        bl = self.baseline

        # Copy initial_peaks
        # While it would be nice to integrate them into extracted model naturally
        # as it progresses, this is fraught with difficulties.  Thus, they will
        # only be added back in before the final prune.
        ip = self.initial_peaks.copy()

        rangeslice = self.getrangeslice()
        x = self.x[rangeslice]
        y = self.y[rangeslice] - bl.value(x) - ip.value(x)
        dy = self.effective_dy[rangeslice]

        # Object performing clustering on data. Note that DataClusters
        # provides an iterator that clusters the next point and returns
        # itself. Thus, dclusters and step (below) refer to the same object.

        dclusters = DataClusters(x, y, self.cres)  # Cluster with baseline removed

        # The data for model clusters includes the baseline
        y = self.y[rangeslice] - ip.value(x)

        # List of ModelClusters containing extracted peaks.
        mclusters = [ModelCluster(None, bl, x, y, dy, dclusters.cut(0), self.error_method, self.pf)]

        # The minimum number of points required to make a valid fit, as
        # determined by the peak functions and error method used.  This is a
        # conservative estimate.
        minpoints = max([self.error_method().minpoints(p.npars) for p in self.pf])

        stepcounter = 0

        # #########################
        #  Main extraction loop ###
        for step in dclusters:

            stepcounter += 1

            msg = "\n\n------ Recursion: %s  Step: %s  Cluster: %s %s ------"
            logger.debug(
                msg,
                recursion_depth,
                stepcounter,
                step.lastcluster_idx,
                step.clusters[step.lastcluster_idx],
            )

            # Update mclusters
            if len(step.clusters) > len(mclusters):
                # Add a new cluster
                mclusters.insert(
                    step.lastcluster_idx,
                    ModelCluster(
                        None,
                        bl,
                        x,
                        y,
                        dy,
                        step.cut(step.lastcluster_idx),
                        self.error_method,
                        self.pf,
                    ),
                )
            else:
                # Update an existing cluster
                mclusters[step.lastcluster_idx].change_slice(step.cut(step.lastcluster_idx))

            # Find newly adjacent clusters
            adjacent = step.find_adjacent_clusters().ravel()

            # Various assertions in case terrible things are afoot.
            # These could save some gray hairs if they are needed.
            # ------
            # dclusters and mclusters should have consistent lengths
            assert len(step.clusters) == len(mclusters)
            # Clusters are always combined after becoming adjacent, so at most
            # three clusters can become adjacent at any given step.
            assert len(adjacent) <= 3

            #  Update cluster fits ###
            # 1. Refit clusters adjacent to at least one other cluster.
            for a in adjacent:
                mclusters[a].fit(justify=True)

            # 2. If necessary, update the fit of the cluster which has just
            #    had one or more points added.  This occurs if the function
            #    has not been fit before but now contains enough data points
            #    to make a good estimate or if the size of the cluster has
            #    increased enough (e.g. doubled in size) since it was last
            #    fit.
            mclusters[step.lastcluster_idx].contingent_fit(minpoints, 2.0)

            # 3. Boundary recursion.  If a cluster fills to the boundary of
            #    the data it should be recursively fit as though it were
            #    combining with an empty cluster at the boundary.  This should
            #    reveal hidden peaks that might otherwise be improperly fit
            #    with just a single peak function.
            #
            #    Note: If I later implement intra-cluster fitting, this
            #    section may become redundant...or the basis for doing it
            #    properly.  Two if statements are required, in case the fit
            #    results in all peaks blowing up and being removed.
            #
            #    Note: The operation here is very similar to combining
            #    clusters and recursing.  Attempt to be be consistent with
            #    that section.  The primary difference is no need to create an
            #    enlarged cluster ("new_cluster") or an intermediate cluster
            #    ("adj_cluster").

            if step.lastpoint_idx == 0 or step.lastpoint_idx == len(step.x) - 1:
                logger.debug("Boundary full: %s", step.lastpoint_idx)
                full_cluster = ModelCluster(mclusters[step.lastcluster_idx])
                full_cluster.fit(True)

                # Estimate coordinate where clusters combine.
                border_x = x[step.lastcluster_idx]
                border_y = y[step.lastcluster_idx]

                # Determine neighborhood appropriate for fitting (no larger than combined clusters)
                if len(full_cluster.model) > 0:
                    peak_pos = np.array([p["position"] for p in full_cluster.model])
                    pivot = peak_pos.searchsorted(border_x)
                else:
                    peak_pos = np.array([])
                    pivot = 0

                # near_peaks: array containing the indices of two nearest peaks on either side of border_x
                # other_peaks: all the other peaks in full_cluster
                # left_data, right_data: indices defining the extent of the "interpeak range" for x, etc.
                near_peaks = np.array([], dtype=np.int)

                # interpeak range goes from peak to peak of next nearest peaks, although their contributions
                # to the data are still removed.
                if pivot == 0:
                    # No peaks left of border_x!
                    left_data = full_cluster.slice.indices(len(x))[0]
                elif pivot == 1:
                    # One peak left
                    left_data = full_cluster.slice.indices(len(x))[0]
                    near_peaks = np.append(near_peaks, pivot - 1)
                else:
                    # left_data -> one more peak to the left
                    left_data = max(0, x.searchsorted(peak_pos[pivot - 2]) - 1)
                    near_peaks = np.append(near_peaks, pivot - 1)

                if pivot == len(peak_pos):
                    # No peaks right of border_x!
                    right_data = full_cluster.slice.indices(len(x))[1] - 1
                elif pivot == len(peak_pos) - 1:
                    # One peak right
                    right_data = full_cluster.slice.indices(len(x))[1] - 1
                    near_peaks = np.append(near_peaks, pivot)
                else:
                    # right_data -> one more peak to the right
                    right_data = min(len(x), x.searchsorted(peak_pos[pivot + 1]) + 1)
                    near_peaks = np.append(near_peaks, pivot)

                other_peaks = np.concatenate([np.arange(0, pivot - 1), np.arange(pivot + 1, len(peak_pos))])

                # Go from indices to lists of peaks.
                near_peaks = Peaks([full_cluster.model[i] for i in near_peaks])
                other_peaks = Peaks([full_cluster.model[i] for i in other_peaks])

                #  Remove contribution of peaks outside neighborhood
                # Define range of fitting/recursion to the interpeak range
                # The adjusted error is passed unchanged.  This may introduce
                # a few more peaks than is justified, but they can be pruned
                # with the correct statistics at the top level of recursion.
                adj_slice = slice(left_data, right_data + 1)
                adj_x = x[adj_slice]
                adj_y = y[adj_slice] - other_peaks.value(adj_x)
                adj_error = dy[adj_slice]

                adj_cluster = ModelCluster(
                    near_peaks,
                    bl,
                    adj_x,
                    adj_y,
                    adj_error,
                    slice(len(adj_x)),
                    self.error_method,
                    self.pf,
                )

                # Recursively cluster/fit the residual
                rec_r = adj_x
                rec_y = adj_y - near_peaks.value(rec_r)
                rec_error = adj_error

                # Quick check to see if there is anything to find
                min_npars = min([p.npars for p in self.pf])
                checkrec = ModelCluster(
                    None,
                    None,
                    rec_r,
                    rec_y,
                    rec_error,
                    None,
                    self.error_method,
                    self.pf,
                )
                recurse = len(near_peaks) > 0 and checkrec.quality().growth_justified(checkrec, min_npars)

                if recurse and recursion_depth < 3:
                    logger.info(
                        "\n*********STARTING RECURSION level %s (full boundary)************"
                        % (recursion_depth + 1)
                    )
                    rec_search = PeakExtraction()
                    rec_search.setdata(rec_r, rec_y, None, rec_error)
                    rec_search.setvars(
                        quiet=True,
                        baseline=bl,
                        cres=self.cres,
                        pf=self.pf,
                        error_method=self.error_method,
                    )
                    rec_search.extract_single(recursion_depth + 1)
                    rec = rec_search.extracted
                    logger.info(
                        "*********ENDING RECURSION level %s (full boundary) ************\n" % (recursion_depth + 1)
                    )

                    # Incorporate best peaks from recursive search.
                    adj_cluster.augment(rec)

                #  Select which model to use
                full_cluster.model = other_peaks
                full_cluster.replacepeaks(adj_cluster.model)
                full_cluster.fit(True)

                msg = [
                    "---Result of full boundary---",
                    "Original cluster:",
                    "%s",
                    "Final cluster:",
                    "%s",
                    "---End of combining clusters---",
                ]
                logger.debug("\n".join(msg), mclusters[step.lastcluster_idx], full_cluster)

                mclusters[step.lastcluster_idx] = full_cluster
            #  End update cluster fits ###

            #  Combine adjacent clusters ###

            # Iterate in reverse order to preserve earlier indices
            for idx in adjacent[-1:0:-1]:

                msg = ["Current model"]
                msg.extend(["%s" for m in mclusters])
                logger.debug("\n".join(msg), *[m.model for m in mclusters])

                cleft = step.clusters[idx - 1]
                cright = step.clusters[idx]

                new_cluster = ModelCluster.join_adjacent(mclusters[idx - 1], mclusters[idx])

                # Estimate coordinate where clusters combine.
                border_x = 0.5 * (x[cleft[1]] + x[cright[0]])
                border_y = 0.5 * (y[cleft[1]] + y[cright[0]])

                # Determine neighborhood appropriate for fitting (no larger than combined clusters)
                if len(new_cluster.model) > 0:
                    peak_pos = np.array([p["position"] for p in new_cluster.model])
                    pivot = peak_pos.searchsorted(border_x)
                else:
                    peak_pos = np.array([])
                    pivot = 0

                # near_peaks: array containing the indices of two nearest peaks on either side of border_x
                # other_peaks: all the other peaks in new_cluster
                # left_data, right_data: indices defining the extent of the "interpeak range" for x, etc.
                near_peaks = np.array([], dtype=np.int)

                # interpeak range goes from peak to peak of next nearest peaks, although their contributions
                # to the data are still removed.
                if pivot == 0:
                    # No peaks left of border_x!
                    left_data = new_cluster.slice.indices(len(x))[0]
                elif pivot == 1:
                    # One peak left
                    left_data = new_cluster.slice.indices(len(x))[0]
                    near_peaks = np.append(near_peaks, pivot - 1)
                else:
                    # left_data -> one more peak to the left
                    left_data = max(0, x.searchsorted(peak_pos[pivot - 2]) - 1)
                    near_peaks = np.append(near_peaks, pivot - 1)

                if pivot == len(peak_pos):
                    # No peaks right of border_x!
                    right_data = new_cluster.slice.indices(len(x))[1] - 1
                elif pivot == len(peak_pos) - 1:
                    # One peak right
                    right_data = new_cluster.slice.indices(len(x))[1] - 1
                    near_peaks = np.append(near_peaks, pivot)
                else:
                    # right_data -> one more peak to the right
                    right_data = min(len(x), x.searchsorted(peak_pos[pivot + 1]) + 1)
                    near_peaks = np.append(near_peaks, pivot)

                other_peaks = np.concatenate([np.arange(0, pivot - 1), np.arange(pivot + 1, len(peak_pos))])

                # Go from indices to lists of peaks.
                near_peaks = Peaks([new_cluster.model[i] for i in near_peaks])
                other_peaks = Peaks([new_cluster.model[i] for i in other_peaks])

                #  Remove contribution of peaks outside neighborhood
                # Define range of fitting/recursion to the interpeak range
                # The adjusted error is passed unchanged.  This may introduce
                # a few more peaks than is justified, but they can be pruned
                # with the correct statistics at the top level of recursion.
                adj_slice = slice(left_data, right_data + 1)
                adj_x = x[adj_slice]
                adj_y = y[adj_slice] - other_peaks.value(adj_x)
                adj_error = dy[adj_slice]

                # # Perform recursion on a version that is scaled at the
                # border, as well as on that is simply fit beforehand.  In
                # many cases these lead to nearly identical results, but
                # occasionally one works much better than the other.
                adj_cluster1 = ModelCluster(
                    near_peaks.copy(),
                    bl,
                    adj_x,
                    adj_y,
                    adj_error,
                    slice(len(adj_x)),
                    self.error_method,
                    self.pf,
                )
                adj_cluster2 = ModelCluster(
                    near_peaks.copy(),
                    bl,
                    adj_x,
                    adj_y,
                    adj_error,
                    slice(len(adj_x)),
                    self.error_method,
                    self.pf,
                )

                # Adjust cluster at border if there is at least one peak on
                # either side.
                if len(near_peaks) == 2:
                    adj_cluster1.reduce_to(border_x, border_y)

                    # Recursively cluster/fit the residual
                    rec_r1 = adj_x
                    # rec_y1 = adj_y - near_peaks.value(rec_r1)
                    rec_y1 = adj_y - adj_cluster1.model.value(rec_r1)
                    rec_error1 = adj_error

                    # Quick check to see if there is anything to find
                    min_npars = min([p.npars for p in self.pf])
                    checkrec = ModelCluster(
                        None,
                        None,
                        rec_r1,
                        rec_y1,
                        rec_error1,
                        None,
                        self.error_method,
                        self.pf,
                    )
                    recurse1 = checkrec.quality().growth_justified(checkrec, min_npars)

                    if recurse1 and recursion_depth < 3:
                        logger.info(
                            "\n*********STARTING RECURSION level %s (reduce at border)************"
                            % (recursion_depth + 1)
                        )
                        rec_search1 = PeakExtraction()
                        rec_search1.setdata(rec_r1, rec_y1, None, rec_error1)
                        rec_search1.setvars(
                            quiet=True,
                            baseline=bl,
                            cres=self.cres,
                            pf=self.pf,
                            error_method=self.error_method,
                        )
                        rec_search1.extract_single(recursion_depth + 1)
                        rec1 = rec_search1.extracted
                        logger.info(
                            "*********ENDING RECURSION level %s (reduce at border) ************\n"
                            % (recursion_depth + 1)
                        )

                        # Incorporate best peaks from recursive search.
                        adj_cluster1.augment(rec1)

                # Fit cluster model
                adj_cluster2.fit(True)

                # Recursively cluster/fit the residual
                rec_r2 = adj_x
                # rec_y2 = adj_y - near_peaks.value(rec_r2)
                rec_y2 = adj_y - adj_cluster2.model.value(rec_r2)
                rec_error2 = adj_error

                # Quick check to see if there is anything to find
                min_npars = min([p.npars for p in self.pf])
                checkrec = ModelCluster(
                    None,
                    None,
                    rec_r2,
                    rec_y2,
                    rec_error2,
                    None,
                    self.error_method,
                    self.pf,
                )
                recurse2 = len(near_peaks) > 0 and checkrec.quality().growth_justified(checkrec, min_npars)

                if recurse2 and recursion_depth < 3:
                    logger.info(
                        "\n*********STARTING RECURSION level %s (prefit)************" % (recursion_depth + 1)
                    )
                    rec_search2 = PeakExtraction()
                    rec_search2.setdata(rec_r2, rec_y2, None, rec_error2)
                    rec_search2.setvars(
                        quiet=True,
                        baseline=bl,
                        cres=self.cres,
                        pf=self.pf,
                        error_method=self.error_method,
                    )
                    rec_search2.extract_single(recursion_depth + 1)
                    rec2 = rec_search2.extracted
                    logger.info(
                        "*********ENDING RECURSION level %s (prefit) ************\n" % (recursion_depth + 1)
                    )

                    # Incorporate best peaks from recursive search.
                    adj_cluster2.augment(rec2)

                #  Select which model to use
                new_cluster.model = other_peaks
                rej_cluster = ModelCluster(new_cluster)
                q1 = adj_cluster1.quality(self.error_method)
                q2 = adj_cluster2.quality(self.error_method)
                if q1 > q2:
                    new_cluster.replacepeaks(adj_cluster1.model)
                    rej_cluster.replacepeaks(adj_cluster2.model)
                else:
                    new_cluster.replacepeaks(adj_cluster2.model)
                    rej_cluster.replacepeaks(adj_cluster1.model)

                new_cluster.fit(True)

                msg = [
                    "---Result of combining clusters---",
                    "First cluster:",
                    "%s",
                    "Second cluster:",
                    "%s",
                    "Resulting cluster:",
                    "%s",
                    "---End of combining clusters---",
                ]

                logger.debug("\n".join(msg), mclusters[idx - 1], mclusters[idx], new_cluster)

                mclusters[idx - 1] = new_cluster
                del mclusters[idx]

            #  End combine adjacent clusters loop ###

            # Finally, combine clusters in dclusters
            if len(adjacent) > 0:
                step.combine_clusters([adjacent])

            tracer.emit(*mclusters)

        #  End main extraction loop ###
        # #############################

        # Put initial peaks back in
        mclusters[0].addexternalpeaks(ip)

        # Remove unnecessary peaks
        mclusters[0].prune()

        # At the top level of recursion the baseline should be fit as well.
        # Other than simply removing the baseline for recursive calls (viable
        # but annoying for display purposes) this is the simplest solution to
        # only fitting the baseline at the very end.
        # Also calculates covariance at this level.
        if recursion_depth == 1:
            cov = ModelCovariance()
            mclusters[0].fit(fitbaseline=True, cov=cov)

        # Update calculated instance variables
        self.extraction_type = "extract_single"
        self.extracted = mclusters[0]

        tracer.popc()
        tracer.popr()

        if recursion_depth == 1:
            return cov

    def fit_single(self):
        """Fit peaks in initial_peaks with baseline. Return ModelCovariance
        instance summarizing results."""

        self.clearcalc()

        # Make sure all required extraction variables have some value
        self.defaultvars()

        # Define grids
        rngslice = self.getrangeslice()
        x = self.x[rngslice]
        y = self.y[rngslice]
        dy = self.effective_dy[rngslice]

        # Set up ModelCluster
        ext = ModelCluster(
            self.initial_peaks,
            self.baseline,
            x,
            y,
            dy,
            None,
            self.error_method,
            self.pf,
        )

        # Fit model with baseline and calculate covariance matrix
        cov = ModelCovariance()
        ext.fit(fitbaseline=True, estimate=False, cov=cov, cov_format="default_output")

        # Update calculated instance variables
        self.extraction_type = "fit_single"
        self.extracted = ext

        return cov


# end PeakExtraction class


# simple test code
if __name__ == "__main__":

    from numpy.random import randn

    from diffpy.srmise.modelevaluators.aicc import AICc
    from diffpy.srmise.peaks.gaussianoverr import GaussianOverR

    srmiselog.setlevel("info")
    srmiselog.liveplotting(False)

    pf = GaussianOverR(0.7)
    res = 0.01

    pars = [[3, 0.2, 10], [3.5, 0.2, 10]]
    ideal_peaks = Peaks([pf.actualize(p, "pwa") for p in pars])

    r = np.arange(2, 4, res)
    y = ideal_peaks.value(r) + randn(len(r))

    err = np.ones(len(r))
    evaluator = AICc()

    te = PeakExtraction()
    te.setdata(r, y, None, err)
    te.setvars(rng=[1.51, 10.0], pf=[pf], cres=0.1, effective_dy=1.5 * err)
    te.extract_single()

    print("--- Actual Peak parameters ---")
    print(ideal_peaks)

    print("\n--- After extraction ---")
    print(te)

    te.plot()
    input()
