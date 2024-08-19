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

import matplotlib.pyplot as plt
import numpy as np

from diffpy.srmise.modelcluster import ModelCluster
from diffpy.srmise.pdfpeakextraction import PDFPeakExtraction


# This is a total hack-job right now, and isn't suitable for
# general use. Limitations include:
# 1) Only works with PDFPeakExtraction, not PeakExtraction
# 2) Only constant uncertainties are supported
# 3) Not using srmiselog
# 4) Really ugly kluge to allow FromSequence to pickle.
class PeakStability:
    """Utility to test robustness of peaks.

    results: [error scalar, model, bl, dr]
    ppe: a PDFPeakExtraction instance"""

    def __init__(self):
        self.results = []
        self.ppe = None
        self.current = None

    def setppe(self, ppe):
        self.ppe = ppe

    def load(self, filename):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        in_s = open(filename, "rb")
        try:
            (self.results, ppestr) = pickle.load(in_s)
            self.ppe = PDFPeakExtraction()
            self.ppe.readstr(ppestr)

            # Ugly kluge for the baseline, since FromSequence
            # can't pickle.
            for r in self.results:
                bl = self.ppe.baseline
                kwds = r[2]
                if r[2] is not None:
                    kwds = r[2]
                    if hasattr(bl, "estimate_parameters"):
                        r[2] = bl.actualize(default_input="internal", **kwds)
                    else:
                        r[2] = bl.owner().actualize(in_format="internal", **kwds)
        finally:
            in_s.close()

        self.setcurrent(0)

    def save(self, filename):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        out_s = open(filename, "wb")
        try:
            # Write to the stream
            outstr = self.ppe.writestr()

            # ugly kluge to let FromSequence pickle
            # (it stores xyrepr() in metadict)
            results2 = []
            for r in self.results:
                if r[2] is None:
                    bldict = None
                else:
                    bldict = {
                        "pars": r[2].pars,
                        "free": r[2].free,
                        "removable": r[2].removable,
                        "static_owner": r[2].static_owner,
                    }
                results2.append([r[0], r[1], bldict, r[3]])
            pickle.dump([results2, outstr], out_s)
        finally:
            out_s.close()

    def plotseries(self, style="o", **kwds):
        plt.figure()
        plt.ioff()
        for e, r, bl, dr in self.results:
            peakpos = [p["position"] for p in r]
            es = [e] * len(peakpos)
            plt.plot(peakpos, es, style, **kwds)
        plt.ion()
        plt.draw()

    def plot(self, **kwds):
        """Plot the current model.  Keywords passed to pyplot.plot()"""
        plt.clf()
        plt.plot(*self.ppe.extracted.plottable(), **kwds)
        q = self.ppe.extracted.quality()
        plt.suptitle(
            "[%i/%i]\n"
            "Uncertainty: %6.3f. Peaks: %i.\n"
            "Quality: %6.3f.  Chi-square: %6.3f"
            % (
                self.current + 1,
                len(self.results),
                self.ppe.effective_dy[0],
                len(self.ppe.extracted.model),
                q.stat,
                q.chisq,
            )
        )

    def setcurrent(self, idx):
        """Make the idxth model the active one.

        Parameters
        ----------
        idx : int
            The index of the model to be tested.

        Returns
        -------
        None
        """
        self.current = idx
        if idx is not None:
            result = self.results[idx]
            self.ppe.setvars(quiet=True, effective_dy=result[0] * np.ones(len(self.ppe.x)))
            (r, y, dr, dy) = self.ppe.resampledata(result[3])
            self.ppe.extracted = ModelCluster(
                result[1], result[2], r, y, dy, None, self.ppe.error_method, self.ppe.pf
            )
        else:
            self.ppe.clearcalc()

    def animate(self, results=None, step=False, **kwds):
        """Show animation of extracted peaks from first to last.

           Keywords passed to pyplot.plot()

        Parameters
        ----------
        step : bool
            Require keypress to show next plot
        results array-like
            The indices of results to show
        """
        if results is None:
            results = range(len(self.results))

        oldcurrent = self.current
        self.setcurrent(0)
        plt.ion()
        plt.plot(*self.ppe.extracted.plottable())
        plt.axis()
        for i in results:
            self.setcurrent(i)
            plt.ioff()
            self.plot(**kwds)
            plt.ion()
            plt.draw()
            if step:
                input()

        self.setcurrent(oldcurrent)

    def run(self, err, savecovs=False):
        """Running the uncertainty for the results.

        Parameters
        ----------
        err : array-like
            The sequence of uncertainties to run at.
        savecovs : bool
            boolean to determine to save covariance matrix. Default is False.
            If savecovs is True, return the covariance matrix for each final fit."""

        self.results = []
        covs = []
        for i, e in enumerate(err):
            print("---- Running for uncertainty %s (%i/%i) ----" % (e, i, len(err)))
            self.ppe.clearcalc()
            self.ppe.setvars(effective_dy=e)
            if savecovs:
                covs.append(self.ppe.extract())
            else:
                self.ppe.extract()
            dr = (self.ppe.extracted.r_cluster[-1] - self.ppe.extracted.r_cluster[0]) / (
                len(self.ppe.extracted.r_cluster) - 1
            )
            self.results.append([e, self.ppe.extracted.model, self.ppe.extracted.baseline, dr])

        for e, r, bl, dr in self.results:
            print("---- Results for uncertainty %s ----" % e)
            print(r)

        return covs
