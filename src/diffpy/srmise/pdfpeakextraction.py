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

import numpy as np

from diffpy.srmise import srmiselog
from diffpy.srmise.modelcluster import ModelCluster, ModelCovariance

# from diffpy.pdfgui.control.pdfdataset import PDFDataSet
from diffpy.srmise.pdfdataset import PDFDataSet
from diffpy.srmise.peakextraction import PeakExtraction
from diffpy.srmise.srmiseerrors import (
    SrMiseDataFormatError,
    SrMiseError,
    SrMiseQmaxError,
    SrMiseStaticOwnerError,
    SrMiseUndefinedCovarianceError,
)

logger = logging.getLogger("diffpy.srmise")


class PDFPeakExtraction(PeakExtraction):
    """PDFPeakExtraction extends the PeakExtraction class to specialize in extracting
    peaks from PDF (Probability Density Function) data.

    Parameters
    ----------
    filename : str
        The source PDF file path.
    nyquist : bool, optional
        Specifies whether to fit the final model at the Nyquist sampling rate.
        Defaults to False.
    qmax : float, optional
        The maximum q value to use during peak extraction. Use 0 to denote infinity.
        Defaults to 0.
    qmax_reportedbypdf : float
        The qmax value read directly from the PDF file.
    qmax_fromdata : float
        The qmax value determined directly from the PDF data.
    scale : bool, optional
        Determines whether to use increased uncertainties during supersampling.
        This can expedite extraction by minimizing the detection of minuscule peaks,
        albeit risking the overlook of minor features. When `Nyquist` is True,
        uncertainties are scaled to maintain a similar chi-square error pre-
        and post-resampling. Defaults to False if `Nyquist` is True.
    supersample : int, optional
        Ensures the data is supersampled by at least this factor above the
        Nyquist rate before initiating peak extraction. Defaults to 1.
    """

    def __init__(self):
        """Initialize the PDFPeakExtraction class."""
        newvars = ["qmax", "supersample", "nyquist", "scale"]
        PeakExtraction.__init__(self, newvars)
        return

    def loadpdf(self, pdf):
        """Load dataset.

        Parameters
        ----------
        pdf: PDFDataSet instance or str
            The PDFDataSet instance or a PDF file name.

        """
        self.clear()
        if isinstance(pdf, PDFDataSet):
            d = pdf
        else:
            d = PDFDataSet("ForPeakExtraction")
            d.read(pdf)
            self.filename = os.path.abspath(pdf)
        self.setdata(d.robs, d.Gobs, d.drobs, d.dGobs)
        self.qmax_reportedbypdf = d.qmax
        return

    def setdata(self, x, y, dx=None, dy=None):
        """Set data.

        Parameters
        ----------
        x : array-like
            The x-coordinates of the data.
        y : array-like
            The y-coordinates of the data."""
        PeakExtraction.setdata(self, x, y, dx, dy)
        try:
            self.qmax_fromdata = find_qmax(self.x, self.y)[0]
        except Exception:
            logger.info("Could not determine qmax from the data.")

    def clear(self):
        """Clear all members.

        The purpose of the method is to ensure the object is in a clean state."""
        # TODO: Clear additional members
        self.filename = None
        self.nyquist = None
        self.qmax = None
        self.qmax_reportedbypdf = None
        self.qmax_fromdata = None
        self.scale = None
        self.supersample = None
        PeakExtraction.clear(self)

    def setvars(self, quiet=False, **kwds):
        """Set one or more extraction variables.

        Parameters
        ----------
        quiet : bool, optional
            Log changes quietly. Default is False.

        **kwds : keyword arguments
            Additional variables to set. Possible keywords include:

        - cres : float
            The clustering resolution, must be greater than or equal to 0.

        - effective_dy : float
            The uncertainties actually used during extraction. Aliased as 'dg'.

        - pf : list of PeakFunctionBase subclasses instances
            Sequence of peak function base subclass instances.

        - baseline : Baseline or BaselineFunction instance
            Baseline instance or BaselineFunction instance for built-in estimation.

        - error_method : ErrorEvaluator subclass instance
            Error evaluator subclass instance used to compare models. Default is AIC.

        - initial_peaks : Peaks instance
            Peaks instance representing the peaks present at the start of extraction.

        - rng : tuple of (float, float)
            Specifies the least and greatest x-values over which to extract peaks.

        - qmax : float or "automatic"
            The qmax value for the probability density function (pdf).
            If set to "automatic", it will be estimated from the data.

        - nyquist : bool
            Whether to use nyquist sampling or not.

        - supersample : int
            Degree of supersampling above the Nyquist rate to use.

        - scale : bool
            Scale uncertainties on recursion when nyquist is True.
        """
        # Treat "dg" as alias for "effective_dy"
        if "dg" in kwds:
            if "effective_dy" not in kwds:
                kwds["effective_dy"] = kwds.pop("dg")
            else:
                emsg = "Keyword 'dg' is alias for 'effective_dy', cannot specify both."
                raise ValueError(emsg)
        if "qmax" in kwds:
            if kwds["qmax"] == "automatic":
                if self.qmax_fromdata is not None:
                    kwds["qmax"] = self.qmax_fromdata
                else:
                    emsg = "Qmax could not be automatically determined."
                    raise SrMiseQmaxError(emsg)
        if "supersample" in kwds:
            if kwds["supersample"] < 1.0:
                emsg = "Supersample must be a value greater than 1."
                raise ValueError(emsg)
        PeakExtraction.setvars(self, quiet, **kwds)

    def defaultvars(self, *args):
        """Set default values.

        Parameters
        ----------
        *args : argparse.Namespace
            Arguments passed to PeakExtraction.setdata()."""
        nargs = list(args)

        # qmax preference: reported, then fromdata, then 0.
        if self.qmax is None or "qmax" in args:
            if self.qmax_reportedbypdf is not None:
                self.qmax = self.qmax_reportedbypdf
            elif self.qmax_fromdata is not None:
                self.qmax = self.qmax_fromdata
            else:
                self.qmax = 0.0
            if "qmax" in args:
                nargs.remove("qmax")

        # nyquist
        if self.nyquist is None or "nyquist" in args:
            if self.qmax > 0:
                self.nyquist = True
            else:
                self.nyquist = False
            if "nyquist" in args:
                nargs.remove("nyquist")

        # scale
        if self.scale is None or "scale" in args:
            if self.nyquist:
                self.scale = False
            else:
                self.scale = False
            if "scale" in args:
                nargs.remove("scale")

        # supersample
        if self.supersample is None or "supersample" in args:
            self.supersample = 4.0
            if "supersample" in args:
                nargs.remove("supersample")

        # Override defaults from PeakExtraction
        if self.cres is None or "cres" in args:
            if self.qmax > 0:
                self.cres = np.pi / self.qmax
                if "cres" in args:
                    nargs.remove("cres")

        if self.pf is None or "pf" in args:
            from diffpy.srmise.peaks.gaussianoverr import GaussianOverR

            self.pf = [GaussianOverR(0.7)]
            if "pf" in args:
                nargs.remove("pf")

        # Enable "dg" as alias for "effective_dy"
        if "dg" in args and "effective_dy" not in args:
            nargs.add("effective_dy")

        # Set other defaults
        PeakExtraction.defaultvars(self, *nargs)

    def resampledata(self, dr, **kwds):
        """Return (x, y, error in x, effective error in y) resampled by interval dr.

        Uses values of self.x, self.y, self.dx, self.effective_dy.  The range is
        constrained by self.rng.

        The effective error may be scaled if class member scale is True.

        The method for 'resampling' the uncertainties is interpolation, since insufficient
        information exists in a PDFPeakExtraction object to propogate them correctly on the
        new grid.

        Parameters
        ----------
        dr : float
            The sampling interval for resampling the data.

        **kwds : dict, optional
            Additional keyword arguments.

            - eps : float, default=1e-6
                Suppresses the information lost warning when the difference between `dr`
                and the Nyquist interval `dr_nyquist` is less than `eps`.

        Returns
        -------
        tuple of ndarray
            A tuple containing the resampled (x, y, error in x, effective error in y)."""
        self.defaultvars()  # Find correct range if necessary.
        eps = kwds.get("eps", 10**-6)

        if self.qmax == 0:
            logger.warning("Resampling when qmax=0.  Information may be lost!")
        else:
            dr_nyquist = np.pi / self.qmax

            # Not a robust epsilon test, but all physical Nyquist rates in same oom.
            if dr - dr_nyquist > eps:
                logger.warning(
                    "Resampling at %s, below Nyquist rate of %s.  Information will be lost!" % (dr, dr_nyquist)
                )

        r = np.arange(max(self.x[0], self.rng[0]), min(self.x[-1], self.rng[1]), dr)
        y = resample(self.x, self.y, r)

        # TODO: Use a justified way to "resample" the uncertainties.
        # Even semi-justified would improve this ugly hack.
        if self.dx is None:
            r_error = None
        else:
            r_error = np.interp(r, self.x, self.dx)
        y_error = self.errorscale(dr) * np.interp(r, self.x, self.effective_dy)

        return (r, y, r_error, y_error)

    def errorscale(self, dr):
        """Return proper scale of uncertainties.

        Always returns 1 unless qmax > 0, Nyquist sampling
        is enabled, and scale is True.

        Parameters
        ----------
        dr: float
            The sampling interval

        Returns
        -------
        float
            The uncertainties scaled."""
        if self.qmax > 0 and self.nyquist and self.scale:
            dr_nyquist = np.pi / self.qmax
            return np.max([np.sqrt(dr_nyquist / dr), 1.0])
        else:
            return 1.0

    def extract(self, **kwds):
        """Extract peaks from the PDF. Returns ModelCovariance instance summarizing results.

        Parameters
        ----------
        **kwds : dict
            Additional keyword arguments that might influence the extraction process.
            These could include parameters like `qmax`, `supersample`, `nyquist`, etc., which
            affect resampling and model refinement strategies.

        Returns
        -------
        ModelCovariance
            An instance of ModelCovariance summarizing the covariance of the extracted model parameters.
        """
        # TODO: The sanest way forward is to create a PeakExtraction object that does
        # the calculations for resampled data.  All the relevant extraction variables
        # can be carefully controlled this way as well.  Furthermore, it continues to
        # expose extract_single without change. (Could also implement a keyword system
        # here to control certain values.)

        # TODO: Add extraction variables for how I control resampling.  Add these to
        # newvar, clean, defaultvar, etc.  In most cases the default are just fine.

        self.clearcalc()

        tracer = srmiselog.tracer
        tracer.pushc()

        # Make sure all required extraction variables have some value
        self.defaultvars()

        # Determine grid spacing
        dr_raw = (self.x[-1] - self.x[0]) / (len(self.x) - 1)

        logger.info("Extract using qmax=%s", self.qmax)

        if self.qmax > 0:
            dr_nyquist = np.pi / self.qmax
            if dr_raw > dr_nyquist:
                # Technically I should yell for dr_raw >= dr_nyquist, since information
                # loss may occur at equality.
                logger.warning(
                    "The input PDF appears to be missing information: The "
                    "sampling interval of the input PDF (%s) is larger than "
                    "the Nyquist interval (%s) defined by qmax=%s.  This information "
                    "is irretrievable." % (dr_raw, dr_nyquist, self.qmax)
                )
        else:
            # Do I actually need this?
            dr_nyquist = dr_raw  # not actually Nyquist sampling, natch

        # Define grids
        rngslice = self.getrangeslice()
        if self.qmax == 0:
            r1 = self.x[rngslice]
            y1 = self.y[rngslice]
            y_error1 = self.effective_dy[rngslice]
        else:
            if dr_nyquist / dr_raw < self.supersample:
                # supersample PDF for initial extraction
                dr = dr_nyquist / self.supersample
                (r1, y1, r_error1, y_error1) = self.resampledata(dr)
            else:
                # provided PDF already sufficiently oversampled
                r1 = self.x[rngslice]
                y1 = self.y[rngslice]
                y_error1 = self.errorscale(dr_raw) * self.effective_dy[rngslice]

        # Set up initial extraction
        pe = PeakExtraction()
        pe.setdata(r1, y1, None, None)

        msg = [
            "Performing initial peak extraction",
            "----------------------------------",
        ]
        logger.info("\n".join(msg))

        pe.setvars(
            cres=self.cres,
            pf=self.pf,
            effective_dy=y_error1,
            baseline=self.baseline,
            error_method=self.error_method,
            initial_peaks=self.initial_peaks,
        )

        # Initial extraction
        pe.extract_single()
        ext = pe.extracted
        bl = pe.extracted.baseline

        # Prune model with termination ripples
        if self.qmax > 0:

            msg = [
                "Model after initial extraction.",
                "%s",
                "\n",
                "-----------------------------",
                "Adding termination ripples.",
            ]

            logger.info("\n".join(msg), ext)

            from diffpy.srmise.peaks.terminationripples import TerminationRipples

            owners = list(set([p.owner() for p in ext.model]))
            tfuncs = {}
            for o in owners:
                tfuncs[o] = TerminationRipples(o, self.qmax)
            for p in ext.model:
                try:
                    # TODO: Make this check more robust, or incorporate similar
                    #       functionality into the design of peakfunctionbase.
                    #       It is easy to imagine wrapping a peak function
                    #       several times to achieve different effects, but it
                    #       isn't necessarily clear when any given wrapping
                    #       is inadmissible.
                    if not isinstance(p.owner(), TerminationRipples):
                        p.changeowner(tfuncs[p.owner()])
                except SrMiseStaticOwnerError:
                    pass

            # Use Nyquist sampled data if desired
            if self.nyquist:

                logger.info("Applying Nyquist sampling.")

                # Models with more free parameters than data points cannot be fit
                # with chi-squared methods, so sometimes the existing model cannot
                # be pruned if the data are immediately resampled at the Nyquist
                # rate.  In that case, keep resampling/pruning until no more parameters
                # can be removed or until able to resample at the Nyquist rate.
                # This either gives the correct statistics, or uses the least amount
                # of supersampling that can be mustered.  The latter case usually
                # indicates that effective_dy is too small.
                while True:
                    totalfreepars = ext.npars(count_fixed=False)
                    if totalfreepars >= (r1[-1] - r1[0]) / dr_nyquist:
                        dr_resample = 0.9999 * (r1[-1] - r1[0]) / totalfreepars
                    else:
                        dr_resample = dr_nyquist

                    logger.info(
                        "Data resampled at dr=%s. (Nyquist rate=%s)",
                        dr_resample,
                        dr_nyquist,
                    )

                    (r2, y2, r_error2, y_error2) = self.resampledata(dr_resample)

                    ext = ModelCluster(
                        ext.model,
                        bl,
                        r2,
                        y2,
                        y_error2,
                        None,
                        self.error_method,
                        self.pf,
                    )
                    ext.fit()  # Clean up parameters after resampling.

                    ext.prune()

                    if dr_resample == dr_nyquist:
                        logger.info("Nyquist sampling complete.")
                        break
                    elif ext.npars(count_fixed=False) == totalfreepars:
                        msg = [
                            "Warning: Nyquist sampling could not be completed, too many free parameters.",
                            "The data have been oversampled by the least possible amount, but their ",
                            "uncertainties are correlated and the model is probably overfit.",
                            "Data sampled at dr=%s",
                            "Nyquist rate=%s",
                            "Degree of oversampling: %s",
                        ]
                        logger.warning(
                            "\n".join(msg),
                            dr_resample,
                            dr_nyquist,
                            dr_nyquist / dr_resample,
                        )
                        break

            else:
                ext = ModelCluster(ext.model, bl, r1, y1, y_error1, None, self.error_method, self.pf)
                ext.prune()

            logger.info("Model after resampling and termination ripples:\n%s", ext)

        # Fit model with baseline, report covariance matrix
        cov = ModelCovariance()
        ext.fit(fitbaseline=True, cov=cov, cov_format="default_output")
        tracer.emit(ext)

        logger.info("Model after fitting with baseline:")
        try:
            logger.info(str(cov))
            # logger.info("Correlations > .8:\n%s", "\n".join(str(c) for c in cov.correlationwarning(.8)))
        except SrMiseUndefinedCovarianceError:
            logger.warning("Covariance not defined for final model.  Fit may not have converged.")
            logger.info(str(ext))

        # Update calculated instance variables
        self.extraction_type = "extract"
        self.extracted = ext

        tracer.popc()

        return cov

    def fit(self, **kwds):
        """Fit peaks in the PDF. Returns ModelCovariance instance summarizing results.

        Parameters
        ----------
        **kwds : dict
            Keyword arguments passed to ModelCovariance instance. Different keywords could have
            different strategies to fit. See ModelCovariance class for more information.

        Returns
        -------
        ModelCovariance instance
            The fitted ModelCovariance instance.
        """

        self.clearcalc()

        # Make sure all required extraction variables have some value
        self.defaultvars()

        # Determine grid spacing
        dr_raw = (self.x[-1] - self.x[0]) / (len(self.x) - 1)

        logger.info("Fit using qmax=%s", self.qmax)

        if self.qmax > 0:
            dr_nyquist = np.pi / self.qmax
            if dr_raw > dr_nyquist:
                # Technically I should yell for dr_raw >= dr_nyquist, since information
                # loss may occur at equality.
                logger.warning(
                    "The input PDF appears to be missing information: The "
                    "sampling interval of the input PDF (%s) is larger than "
                    "the Nyquist interval (%s) defined by qmax=%s.  This information "
                    "is irretrievable." % (dr_raw, dr_nyquist, self.qmax)
                )
        else:
            # Do I actually need this?
            dr_nyquist = dr_raw  # not actually Nyquist sampling, natch

        # Define grids
        rngslice = self.getrangeslice()
        if self.qmax == 0 or not self.nyquist:
            r1 = self.x[rngslice]
            y1 = self.y[rngslice]
            y_error1 = self.effective_dy[rngslice]
        else:
            (r1, y1, r_error1, y_error1) = self.resampledata(dr_nyquist)

        # Set up for fit_single
        pe = PeakExtraction()
        pe.setdata(r1, y1, None, None)

        msg = ["Performing peak fitting", "-----------------------"]
        logger.info("\n".join(msg))

        pe.setvars(
            cres=self.cres,
            pf=self.pf,
            effective_dy=y_error1,
            baseline=self.baseline,
            error_method=self.error_method,
            initial_peaks=self.initial_peaks,
        )

        # Perform fit
        cov = pe.fit_single()
        ext = pe.extracted

        logger.info("Model after fitting with baseline:")
        logger.info(str(ext))
        try:
            logger.info(str(cov))
        except SrMiseUndefinedCovarianceError:
            logger.warning("Covariance not defined for final model.  Fit may not have converged.")

        # Update calculated instance variables
        self.extraction_type = "fit"
        self.extracted = ext

        return cov

    def writemetadata(self):
        """Return string representation of peak extraction from PDF."""
        lines = []
        lines.append("filename=%s" % repr(self.filename))
        lines.append("nyquist=%s" % repr(self.nyquist))
        lines.append("qmax=%s" % repr(self.qmax))
        lines.append("qmax_reportedbypdf=%s" % repr(self.qmax_reportedbypdf))
        lines.append("qmax_fromdata=%s" % repr(self.qmax_fromdata))
        lines.append("scale=%s" % repr(self.scale))
        lines.append("supersample=%s" % repr(self.supersample))

        datastring = "\n".join(lines) + "\n"
        return datastring

    def readmetadata(self, metastr):
        """Read metadata from string.

        Parameters
        ----------
        metastr : str
            Metadata string to read.

        Returns
        -------
        None
        """

        # filename
        res = re.search(r"^filename=(.*)$", metastr, re.M)
        if res:
            self.filename = eval(res.groups()[0].strip())
        else:
            emsg = "metastr does not match required field 'filename'"
            raise SrMiseDataFormatError(emsg)

        # nyquist
        res = re.search(r"^nyquist=(.*)$", metastr, re.M)
        if res:
            self.nyquist = eval(res.groups()[0].strip())
        else:
            emsg = "metastr does not match required field 'nyquist'"
            raise SrMiseDataFormatError(emsg)

        # qmax
        res = re.search(r"^qmax=(.*)$", metastr, re.M)
        if res:
            self.qmax = eval(res.groups()[0].strip())
        else:
            emsg = "metastr does not match required field 'qmax'"
            raise SrMiseDataFormatError(emsg)

        # qmax_reportedbypdf
        res = re.search(r"^qmax_reportedbypdf=(.*)$", metastr, re.M)
        if res:
            self.qmax_reportedbypdf = eval(res.groups()[0].strip())
        else:
            emsg = "metastr does not match required field 'qmax_reportedbypdf'"
            raise SrMiseDataFormatError(emsg)

        # qmax_fromdata
        res = re.search(r"^qmax_fromdata=(.*)$", metastr, re.M)
        if res:
            self.qmax_fromdata = eval(res.groups()[0].strip())
        else:
            emsg = "metastr does not match required field 'qmax_fromdata'"
            raise SrMiseDataFormatError(emsg)

        # scale
        res = re.search(r"^scale=(.*)$", metastr, re.M)
        if res:
            self.scale = eval(res.groups()[0].strip())
        else:
            emsg = "metastr does not match required field 'scale'"
            raise SrMiseDataFormatError(emsg)

        # supersample
        res = re.search(r"^supersample=(.*)$", metastr, re.M)
        if res:
            self.supersample = eval(res.groups()[0].strip())
        else:
            emsg = "metastr does not match required field 'supersample'"
            raise SrMiseDataFormatError(emsg)

    def writepwa(self, filename, comments="n/a"):
        """Write string summarizing extracted peaks to file.

        Parameters
        ----------
        filename : str
            The name of the file to write

        comments : str
            The comments to write

        Returns
        -------
        None
        """
        bytes = self.writepwastr(comments)
        f = open(filename, "w")
        f.write(bytes)
        f.close()
        return

    def writepwastr(self, comments):
        """Return string of extracted peaks (position, width, area) in PDF.

        There is not enough information to recreate the extracted peaks from
        this file.

        Parameters
        ----------
        comments : str
            The string added to header containing notes about the output.

        Returns
        -------
        None
        """

        if self.extracted is None:
            emsg = "Cannot write summary: Peak Extraction has not been performed."
            raise SrMiseError(emsg)

        import time
        from getpass import getuser

        from diffpy.srmise import __version__
        from diffpy.srmise.basefunction import BaseFunction

        lines = []

        # Header
        lines.extend(
            [
                "Summary written: " + time.ctime(),
                "produced by " + getuser(),
                "diffpy.srmise version %s" % __version__,
                "##### User comments",
            ]
        )

        tcomments = []
        for c in comments.splitlines():
            tcomments.append("# " + c)

        lines.extend(tcomments)

        lines.extend(
            [
                "##### PDF Peak Extraction Summary",
                "# The information below is not sufficient to replicate extraction.",
            ]
        )

        # PDF peak extraction metadata
        lines.append("## PDF metadata")
        lines.append("filename=%s" % self.filename)
        lines.append("nyquist=%s" % repr(self.nyquist))
        lines.append("qmax=%s" % repr(self.qmax))
        lines.append("qmax_reportedbypdf=%s" % repr(self.qmax_reportedbypdf))
        lines.append("qmax_fromdata=%s" % repr(self.qmax_fromdata))
        lines.append("scale=%s" % repr(self.scale))
        lines.append("supersample=%s" % repr(self.supersample))
        lines.append("")

        lines.append("## Peak extraction metadata")

        # Extraction range
        lines.append("Range=%s" % repr(self.rng))

        # Clustering resolution
        lines.append("cres=%g" % self.cres)

        # Average effective dy
        lines.append("effective_dy=%s (mean)" % np.mean(self.effective_dy))
        # Passing the entire thing is something we're avoiding in the summary.

        lines.append("")

        # Initial peaks
        # I'm not sure what I want for this, but probably nothing.
        # lines.append('## Initial Peaks')
        # lines.append('')

        lines.append("## Model Quality")
        # Quality of fit
        lines.append("# Quality reported by ModelEvaluator: %s" % self.extracted.quality().stat)
        lines.append("# Free parameters in extracted peaks: %s" % self.extracted.model.npars(count_fixed=False))
        if self.baseline is not None:
            fblpars = self.baseline.npars(count_fixed=False)
        else:
            fblpars = 0
        lines.append("# Free parameters in baseline: %s" % fblpars)
        lines.append("# Length of data in final fit: %s" % len(self.extracted.r_data))

        # Model evaluator
        if self.error_method is None:
            lines.append("ModelEvaluator=None")
        else:
            lines.append("ModelEvaluator=%s" % self.error_method.__name__)

        lines.append("")

        # Generate list of PeakFunctions and BaselineFunctions
        # so I can refer to them by index.
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

        # Baseline Functions
        lines.append("## Baseline Functions")
        lines.append("# Index Type")
        for i, bf in enumerate(safebf):
            if bf.base is not None:
                base = "(base=%s)" % safebf.index(bf.base)
            else:
                base = ""
            lines.append("%s %s %s" % (i, bf.getmodule(), base))

        lines.append("")

        # Baseline
        lines.append("## Baseline")
        lines.append("# Parameters of baseline, followed by comment which")
        lines.append("# gives the index of corresponding Baseline Function.")
        if self.extracted.baseline is None:
            lines.append("None")
        else:
            # Generate parameter labels from the baseline function's parameterdict
            blf = self.extracted.baseline.owner()
            if blf.npars > 0:
                parlbl = blf.parameterdict.keys()
                paridx = np.array(blf.parameterdict.values()).argsort()
                lines.append("# " + " ".join([str(parlbl[i]) for i in paridx]))
                blpars = " ".join([str(p) for p in self.extracted.baseline.pars])
            else:
                blpars = "(no parameters)"
            blidx = safebf.index(blf)
            lines.append("%s # %s" % (blpars, blidx))

        lines.append("")

        # Peak Functions
        lines.append("## Peak Functions")
        lines.append("# Index Type")
        for i, pf in enumerate(safepf):
            if pf.base is not None:
                base = "(base=%s)" % safepf.index(pf.base)
            else:
                base = ""
            lines.append("%s %s %s" % (i, pf.getmodule(), base))

        lines.append("")

        # PWA
        lines.append("## Extracted Peaks")
        lines.append("# Parameters are given in the natural units of the data,")
        lines.append("# where width is measured as full-width at half maximum.")
        lines.append("# Each line is followd by a comment which gives the index")
        lines.append("# of the corresponding Peak Function.")
        lines.append("#L position fwhm area")
        for p in self.extracted.model:
            pf = p.owner()
            pwa = pf.transform_parameters(p, in_format="internal", out_format="pwa")
            pidx = pf.parameterdict["position"]
            widx = pf.parameterdict["width"]
            aidx = pf.parameterdict["area"]
            pfidx = safepf.index(pf)
            lines.append("%s %s %s # %s" % (pwa[pidx], pwa[widx], pwa[aidx], pfidx))

        datastring = "\n".join(lines) + "\n"
        return datastring


# end PDFPeakExtraction class


def resample(orig_r, orig_y, new_r):
    """Resample sequence with Whittaker-Shannon interpolation formula.

    Parameters
    ----------
    orig_r : array-like
        The r grid of the original sample.
    orig_y : array-like
        The data to resample.
    new_r : array-like
        The resampled r grid.

    Returns
    -------
    new_y : array-like
        The sequence of same type as new_r with the resampled values.
    """
    n = len(orig_r)
    dr = (orig_r[-1] - orig_r[0]) / (n - 1)

    if new_r[0] < orig_r[0]:
        logger.warning("Resampling outside original grid: %s (requested) < %s (original)" % (new_r[0], orig_r[0]))

    if new_r[-1] > orig_r[-1]:
        logger.warning(
            "Resampling outside original grid: %s (requested) > %s (original)" % (new_r[-1], orig_r[-1])
        )

    new_y = new_r * 0.0
    shift = orig_r[0] / dr
    for i, y in enumerate(orig_y):
        new_y += y * np.sinc(new_r / dr - (i + shift))

    return new_y


def find_qmax(r, y, showgraphs=False):
    """Determine approximate qmax from PDF.

    Parameters
    ----------
    r : array-like
        The r values of the PDF.
    y : array-like
        The corresponding y values of the PDF.
    showgraphs : bool
        If True, the graphs are shown.

    Returns
    -------
    tuple
        The qmax of the PDF and its corresponding uncertainties.
    """
    if len(r) != len(y):
        emsg = "Argument arrays must have the same length."
        raise ValueError(emsg)

    dr = (r[-1] - r[0]) / (len(r) - 1)

    # Nyquist-sampled PDFs already have the minimum number of data points, so
    # they must be resampled so sudden changes in reciprocal space amplitudes
    # can be observed.
    new_r = np.linspace(r[0], r[-1], 2 * len(r))
    new_y = resample(r, y, new_r)
    new_dr = (new_r[-1] - r[0]) / (len(new_r) - 1)

    yfft = np.imag(np.fft.fft(new_y))[: len(new_y) / 2]

    d_ratio = stdratio(yfft)

    # Negative of the 2nd (forward) difference of d_ratio.  There should be
    # a noticeable jump in the ratio at qmax.  Before and after qmax the ratio
    # smoothly rises (roughly linearly or as a second-order polynomial).  If
    # there is a large amount of extra data past qmax the ratio will fall
    # smoothly as d_left start to include points past qmax.  Taking the 2nd
    # difference removes this smoothly varying part, leaving relatively sharp
    # peaks near the jump.  The jump in d_ratio (even if obscured) appears as
    # a negative peak in the 2nd derivative, hence multiplication by -1.
    dder = -np.diff(d_ratio, 2)

    # The index of yfft which leads to the jump in d_ratio.  (Search dder in
    # reverse since we probably want the last maximum in it, in the unlikely case
    # it appears more than once, while argmax returns the first.)
    m_idx = len(dder) - np.argmax(dder[::-1]) + 1

    dq = 2 * np.pi / ((len(new_y) - 1) * new_dr)
    qmax = dq * m_idx

    # Calculate dq again for original grid
    dq = 2 * np.pi / ((len(y) - 1) * dr)

    if showgraphs:
        import matplotlib.pyplot as plt

        v1 = np.max([m_idx - int(np.sqrt(m_idx)), 2])
        v2 = np.min([m_idx + int(np.sqrt(len(yfft) - 1 - m_idx)), len(d_ratio)])

        plt.ion()

        # DFT of the PDF
        plt.figure()
        plt.subplot(211)  # near obtained qmax
        plt.plot(dq * np.arange(v1, v2), yfft[v1:v2])
        plt.axvline(x=qmax, color="r")
        plt.suptitle("qmax: %s (dq=%s)" % (qmax, dq))
        plt.subplot(212)  # over whole range
        plt.plot(dq * np.arange(len(yfft)), yfft)
        plt.axvline(x=qmax, color="r")

        plt.show()
        plt.ioff()
        input()

    return (qmax, dq)


def stdratio(data):
    """Calculate ratio of standard deviation for runs of equal length in data.

    Uses a numerically-stable online algorithm for calculating the standard
    deviation.

    Parameters
    ----------
    data : array-like
        The sequence of data values

    Returns
    -------
    array-like
        an array of length floor(len(data)/2)-1.  The ith element is
        equivalent to std(data[:i+2])/std(data[i+2:2i+4])."""

    limit = int(np.floor(len(data) / 2))
    std_left = np.zeros(limit)
    std_right = np.zeros(limit)

    n = 0
    mean = 0.0
    m2 = 0.0

    # Update std_left.  This is the usual online algorithm.
    for i in range(limit):
        n = n + 1
        delta = data[i] - mean
        mean = mean + delta / n
        m2 = m2 + delta * (data[i] - mean)
        std_left[i] = m2

    n = 2
    mean = (data[2] + data[3]) / 2
    m2 = (data[2] - mean) ** 2 + (data[3] - mean) ** 2
    std_right[0] = 0.0
    std_right[1] = m2

    # Update std_right.  Remove one from left side, add two on right.
    for i in range(2, limit):
        # Remove one from left
        n = n - 1
        delta = data[i] - mean
        mean = mean - delta / n
        m2 = m2 - delta * (data[i] - mean)

        # Add two to right
        n = n + 2
        mean_add = (data[2 * i] + data[2 * i + 1]) / 2
        m2_add = (data[2 * i] - mean_add) ** 2 + (data[2 * i + 1] - mean_add) ** 2
        delta = mean_add - mean
        mean = mean + (2 * delta) / n
        m2 = m2 + m2_add + (delta**2) * (2 * n - 4) / n
        std_right[i] = m2

    return np.sqrt(std_left[1:] / std_right[1:])


# Redirect to main extraction script
if __name__ == "__main__":
    from diffpy.srmise.applications import extract

    extract.main()
