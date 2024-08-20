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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms

from diffpy.srmise.modelcluster import ModelCluster
from diffpy.srmise.peakstability import PeakStability

logger = logging.getLogger("diffpy.srmise")


def eatkwds(*args, **kwds):
    """Convenience function to remove all keywords in args from kwds."""
    for k in args:
        if k in kwds:
            print("Keyword %s=%s ignored." % (k, kwds.pop(k)))
    return kwds


class MultimodelSelection(PeakStability):
    """Quick and dirty multimodel selection using AIC and its offspring."""

    def __init__(self):
        """ """
        self.dgs = np.array([])
        self.dgs_idx = {}

        self.classes = []
        self.classes_idx = []
        self.class_tolerance = None

        # Track the best models
        self.aics = {}
        self.aicweights = {}
        self.aicprobs = {}
        self.sortedprobs = {}

        # Track the best models (separated into classes)
        self.classweights = {}
        self.classprobs = {}
        self.sortedclassprobs = {}
        self.sortedclasses = {}  # dg->as self.classes, but with model indices sorted by best AIC

        PeakStability.__init__(self)
        return

    def makeaics(self, dgs, dr, filename=None):
        """Test quality of each model for all possible uncertainties.

        Parameters
        ----------
        dgs : array-like
            The array of uncertainties over which to test each model.
        dr : float
            The sampling rate to use.  This determines the actual data to use
            for testing, since sometimes the actual result is different than the
            nominal value.
        filename : str
            Optional file to save pickled results

        Returns
        -------
        None
        """
        aics_out = {}  # Version of self.aics that holds only the statistic, not the AIC object.
        self.dgs = np.array(dgs)
        for i, dg in enumerate(self.dgs):
            self.dgs_idx[dg] = i
            self.aics[dg] = []
            aics_out[dg] = []

        (r, y, dr, dy) = self.ppe.resampledata(dr)

        for model_idx in range(len(self.results)):
            print("Testing model %s of %s." % (model_idx, len(self.results)))

            result = self.results[model_idx]
            em = self.ppe.error_method

            # This section dependent on spaghetti code elsewhere in srmise!
            # Short cut evaluation of AICs which doesn't require calculating chi-square
            # over and over again.  This method assumes that the various uncertainties
            # being evaluated are all proportional to each other, and that constant of
            # proportionality can be determined from self.dgs.  It also depends on the
            # non-obvious behavior of modelevaluators.AIC (and its inheritors) that the
            # chi-square value is only recalculated if its current value is None.  That
            # code is old and pretty ugly, and if that functionality changes this will
            # also break.  That said, PeakStability, MultimodelSelection, and the entire
            # modelevaluators subpackage are in need of a rewrite, and so it would be
            # best to do them all at once.
            dg0 = self.dgs[0]
            mc = ModelCluster(result[1], result[2], r, y, dg0 * np.ones(len(r)), None, em, self.ppe.pf)
            em0 = mc.quality()

            for dg in self.dgs:
                em_instance = em()
                em_instance.chisq = em0.chisq * (dg0 / dg) ** 2  # rescale chi-square
                em_instance.evaluate(mc)  # evaluate AIC without recalculating chi-square
                self.aics[dg].append(em_instance)
                aics_out[dg].append(em_instance.stat)

        self.makeaicweights()
        self.makeaicprobs()
        self.makesortedprobs()

        if filename is not None:
            try:
                import cPickle as pickle
            except ImportError:
                import pickle
            out_s = open(filename, "wb")
            pickle.dump(aics_out, out_s)
            out_s.close()

        return

    def loadaics(self, filename):
        """Load file containing results of the testall method.

        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        None
        """
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        in_s = open(filename, "rb")
        aics_in = pickle.load(in_s)
        in_s.close()

        em = self.ppe.error_method

        self.aics = aics_in
        self.dgs = np.sort(aics_in.keys())

        for i, dg in enumerate(self.dgs):
            self.dgs_idx[dg] = i

        # Reconstruct in terms of AIC instances instead of the raw statistic found in file
        for dg in self.aics:
            for i, stat in enumerate(self.aics[dg]):
                em_instance = em()
                em_instance.stat = stat
                self.aics[dg][i] = em_instance

        self.makeaicweights()
        self.makeaicprobs()
        self.makesortedprobs()

        return

    def makeaicweights(self):
        """Make weights for the aic Modelevaluators."""
        self.aicweights = {}
        em = self.ppe.error_method

        for dg in self.dgs:
            self.aicweights[dg] = em.akaikeweights(self.aics[dg])

    def makeaicprobs(self):
        """Make probabilities for the sequence of AICs."""
        self.aicprobs = {}
        em = self.ppe.error_method

        for dg in self.dgs:
            self.aicprobs[dg] = em.akaikeprobs(self.aics[dg])

    def makesortedprobs(self):
        """Make probabilities for the sequence of AICs in a sorted order."""
        self.sortedprobs = {}

        for dg in self.dgs:
            self.sortedprobs[dg] = np.argsort(self.aicprobs[dg]).tolist()

    def animate_probs(self, step=False, duration=0.0, **kwds):
        """Show animation of extracted peaks from first to last.

        Parameters
        ----------
        step : bool
            Require keypress to show next plot, default is False.
        duration : float
            Minimum time in seconds to complete animation. Default is 0.

        Keywords passed to pyplot.plot()"""
        if duration > 0:
            import time

            sleeptime = duration / len(self.dgs)

        plt.ion()
        plt.subplot(211)
        best_idx = self.sortedprobs[self.dgs[0]][-1]
        (line,) = plt.plot(self.dgs, self.aicprobs[self.dgs[0]])
        vline = plt.axvline(self.dgs[0])
        (dot,) = plt.plot(self.dgs[best_idx], self.aicprobs[self.dgs[0]][best_idx], "ro")

        plt.subplot(212)
        self.setcurrent(best_idx)
        plt.plot(*self.ppe.extracted.plottable())
        for dg in self.dgs[1:]:
            plt.ioff()
            line.set_ydata(self.aicprobs[dg])
            vline.set_xdata(dg)
            if self.sortedprobs[dg][-1] != best_idx:
                best_idx = self.sortedprobs[dg][-1]
                s = slice(0, len(self.ppe.extracted.model))
                self.ppe.extracted.replacepeaks(self.results[best_idx][1], s)
                plt.cla()
                plt.plot(*self.ppe.extracted.plottable())
            dot.set_xdata(self.dgs[best_idx])
            dot.set_ydata(self.aicprobs[dg][best_idx])
            plt.ion()
            plt.draw()
            if step:
                input()
            if duration > 0:
                time.sleep(sleeptime)

    def animate_classprobs(self, step=False, duration=0.0, **kwds):
        """Show animation of extracted peaks from first to last.

        Parameters
        ----------
        step : bool
            Require keypress to show next plot, default is False.
        duration : float
            Minimum time in seconds to complete animation. Default is 0.

        Keywords passed to pyplot.plot()"""
        if duration > 0:
            import time

            sleeptime = duration / len(self.dgs)

        plt.ion()
        ax1 = plt.subplot(211)
        bestclass_idx = self.sortedclassprobs[self.dgs[0]][-1]
        best_idx = self.sortedclasses[self.dgs[0]][bestclass_idx][-1]
        arrow_left = len(self.classes) - 1
        arrow_right = arrow_left + 0.05 * arrow_left
        (line,) = plt.plot(range(len(self.classes)), self.classprobs[self.dgs[0]])
        (dot,) = plt.plot(self.dgs[best_idx], self.classprobs[self.dgs[0]][bestclass_idx], "ro")
        plt.axvline(arrow_left, color="k")
        ax2 = ax1.twinx()
        ax2.set_ylim(self.dgs[0], self.dgs[-1])
        ax2.set_ylabel("dg")
        ax1.set_xlim(right=arrow_right)
        ax2.set_xlim(right=arrow_right)
        dgarrow = ax2.annotate(
            "",
            (arrow_right, self.dgs[0]),
            (arrow_left, self.dgs[0]),
            arrowprops=dict(arrowstyle="-|>"),
        )

        plt.subplot(212)
        self.setcurrent(best_idx)
        tempcluster = ModelCluster(self.ppe.extracted)
        val = tempcluster.plottable()
        minval = np.min(val[1::2])
        [r, res] = tempcluster.plottable_residual()
        plt.plot(*val)
        plt.plot(r, minval - np.max(res) + res)
        for dg in self.dgs[1:]:
            plt.ioff()
            line.set_ydata(self.classprobs[dg])
            dgarrow.xy = (arrow_right, dg)
            dgarrow.xytext = (arrow_left, dg)
            if self.sortedclassprobs[dg][-1] != bestclass_idx:
                bestclass_idx = self.sortedclassprobs[dg][-1]
                best_idx = self.sortedclasses[dg][bestclass_idx][-1]
                s = slice(0, len(tempcluster.model))
                tempcluster.replacepeaks(self.results[best_idx][1], s)
                plt.cla()
                val = tempcluster.plottable()
                minval = np.min(val[1::2])
                [r, res] = tempcluster.plottable_residual()
                plt.plot(*val)
                plt.plot(r, minval - np.max(res) + res)
            dot.set_xdata(bestclass_idx)
            dot.set_ydata(self.classprobs[dg][bestclass_idx])
            plt.ion()
            plt.draw()
            if step:
                input()
            if duration > 0:
                time.sleep(sleeptime)

    def classify(self, r, tolerance=0.05):
        """Find classes of models that are essentially the same.

        Same is defined as having peaks and baselines which all match (to within
        the specified tolerance).  This is calculated as the fraction of the
        difference between the squared values to the squared values of the peak
        (or baseline) itself.  The method is agnostic about free and fixed parameters.

        This will fail in the following cases:
        1) The corresponding peaks are in the wrong order, even if just by a little
        2) The exemplar (first model) of each class isn't the best representative
        3) The parameters vary so smoothly there aren't actually definite classes

        Parameters
        ----------
        r : array-like
            The r values over which to evaluate the models
        tolerance : float
            The fraction below which models are considered the same

        Returns
        -------
        None
        """
        self.classes = []
        self.classes_idx = {}
        self.class_tolerance = None

        classes = []  # each element is a list of the models (result indices) in the class
        classes_idx = {}  # given an integer corresponding to a model, return its class
        epsqval = {}  # holds the squared value of each class' exemplar peaks
        ebsqval = {}  # holds the squared value of each class exemplar baselines

        for i in range(len(self.results)):
            peaks = self.results[i][1]
            baseline = self.results[i][2]
            bsqval = baseline.value(r) ** 2
            psqval = [p.value(r) ** 2 for p in peaks]
            added_to_class = False

            for c in range(len(classes)):
                exemplar_peaks = self.results[classes[c][0]][1]
                exemplar_baseline = self.results[classes[c][0]][2]

                # Check baseline type and number of parameters
                if type(baseline) is not type(exemplar_baseline):
                    continue
                if baseline.npars() != exemplar_baseline.npars():
                    continue

                # check peak types and number of parameters
                badpeak = False
                if len(peaks) != len(exemplar_peaks):
                    continue
                for p, ep in zip(peaks, exemplar_peaks):
                    if type(p) is not type(ep):
                        badpeak = True
                        break
                    if p.npars() != ep.npars():
                        badpeak = True
                        break
                if badpeak:
                    continue

                # check peak values
                for p, ep in zip(psqval, epsqval[c]):
                    basediff = np.abs(np.sum(p - ep))
                    # if basediff > tolerance*np.sum(ep):
                    if basediff > tolerance * np.sum(ep) or basediff > tolerance * np.sum(p):
                        badpeak = True
                        break
                if badpeak:
                    continue

                # check baseline values
                basediff = np.abs(np.sum(bsqval - ebsqval[c]))
                # if basediff > tolerance*np.sum(ebsqval[c]):
                if basediff > tolerance * np.sum(ebsqval[c]) or basediff > tolerance * np.sum(bsqval):
                    continue

                # that's all the checks, add to current class
                added_to_class = True
                classes[c].append(i)
                classes_idx[i] = c
                break

            if added_to_class is False:
                # make a new class with the current model as exemplar
                classes.append([i])
                classnum = len(classes) - 1
                classes_idx[i] = classnum
                epsqval[classnum] = psqval
                ebsqval[classnum] = bsqval

        self.classes = classes
        self.classes_idx = classes_idx
        self.class_tolerance = tolerance

        self.makesortedclasses()
        self.makeclassweights()
        self.makeclassprobs()
        self.makesortedclassprobs()
        return

    def makesortedclasses(self):
        self.sortedclasses = {}

        for dg in self.dgs:
            bestinclass = []
            for cls in self.classes:
                temp = [self.aics[dg][c] for c in cls]
                # poor man's argsort, since numpy is slow on the non-numeric aic objects.
                # Could sort on .stat instead, but that's a trap if I ever use
                # other methods that are sorted in a different fashion.
                best_idx = sorted(range(len(temp)), key=temp.__getitem__)
                bestinclass.append([cls[b] for b in best_idx])
            self.sortedclasses[dg] = bestinclass

    def makeclassweights(self):
        """Make weights for all classes."""
        self.classweights = {}
        em = self.ppe.error_method

        for dg in self.dgs:
            bestinclass = [cls[-1] for cls in self.sortedclasses[dg]]
            self.classweights[dg] = em.akaikeweights([self.aics[dg][b] for b in bestinclass])

    def makeclassprobs(self):
        """Make probabilities for all classes."""
        self.classprobs = {}
        em = self.ppe.error_method

        for dg in self.dgs:
            bestinclass = [cls[-1] for cls in self.sortedclasses[dg]]
            self.classprobs[dg] = em.akaikeprobs([self.aics[dg][b] for b in bestinclass])

    def makesortedclassprobs(self):
        """Make probabilities for all classes in sorted order."""
        self.sortedclassprobs = {}

        for dg in self.dgs:
            self.sortedclassprobs[dg] = np.argsort(self.classprobs[dg]).tolist()

    def dg_key(self, dg_in):
        """Return the dg value usable as a key nearest to dg_in.

        Parameters
        ----------
        dg_in : The uncertainties of the model

        Returns
        -------
        float
            The dg value usable as a key nearest to dg_in."""
        idx = (np.abs(self.dgs - dg_in)).argmin()
        return self.dgs[idx]

    def bestclasses(self, dgs=None):
        """Return the best classes for all models.

        Parameters
        ----------
        dgs : array-like, optional
            The uncertainties of the models, by default None

        Returns
        -------
        array-like
            The best classes for all models."""
        if dgs is None:
            dgs = self.dgs
        best = []
        for dg in dgs:
            best.append(self.sortedclassprobs[dg][-1])
        return np.unique(best)

    def bestmodels(self, dgs=None):
        """Return the best models for all models.

        Parameters
        ----------
        dgs : array-like, optional
            The uncertainties of the models, by default None

        Returns
        -------
        array-like
            Sequence of best model
        """
        if dgs is None:
            dgs = self.dgs
        best = []
        for dg in dgs:
            bestclass_idx = self.sortedclassprobs[dg][-1]
            best.append(self.sortedclasses[dg][bestclass_idx][-1])
        return np.unique(best)

    def classbestdgs(self, cls, dgs=None):
        """Return the best uncertainties for the models.

        Parameters
        ----------
        cls : ModelEvaluator Class
            Override corder with a specific class index, or None to ignore classes entirely.
        dgs : array-like, optional
            The uncertainties of the models, by default None

        Returns
        -------
        array-like
            Sequence of best uncertainties for the models."""
        if dgs is None:
            dgs = self.dgs
        bestdgs = []
        for dg in self.dgs:
            if self.sortedclassprobs[dg][-1] == cls:
                bestdgs.append(dg)
        return bestdgs

    def modelbestdgs(self, model, dgs=None):
        """Return uncertainties where given model has greatest Akaike probability.

        Parameters
        ----------
        model : ModelEvaluator Class
            The model evaluator class to use
        dgs : array-like, optional
            The uncertainties of the models, by default None

        Returns
        -------
        array-like
            The uncertainties where given model has greatest Akaike probability
        """
        if dgs is None:
            dgs = self.dgs
        bestdgs = []
        cls = self.classes_idx[model]
        bestclassdgs = self.classbestdgs(cls, dgs)
        for dg in bestclassdgs:
            if self.sortedclasses[dg][cls][-1] == model:
                bestdgs.append(dg)
        return bestdgs

    def plot3dclassprobs(self, **kwds):
        """Return 3D plot of class probabilities.

        Keywords
        --------
        dGs - Sequence of dG values to plot. Default is all values.
        highlight - Sequence of dG values to highlight on plot.  Default is [].
        classes - Sequence of indices of classes to plot. Default is all classes.
        probfilter - [float1, float2]. Only show classes with maximum probability in given range.
                     Default is [0., 1.]
        class_size - Report the size of each class as a "number" or "fraction".  Default is "number".
        norm - A colors normalization for displaying number/fraction of models in class. Default is "auto".
               If equal to "full" determined by the total number of models.
               If equal to "auto" determined by the number of models in displayed classes.
        cmap - A colormap or registered colormap name.  Default is cm.jet.
            If class_size is "number" and norm is either "auto"
            or "full" the map is converted to an indexed colormap.
        highlight_cmap - A colormap or registered colormap name for coloring highlights.  Default is cm.gray.
        title - True, False, or a string.  Defaults to True, which displays some basic information about the graph.
        p_alpha - Probability graph alpha. (Colorbar remains opaque).  Default is 0.7.
        figure - A matplotlib.figure.Figure instance.  Default is the current figure.
        subplot - Specify a subplot. Default is 111.
        cbpos - Explicit position for colorbar given as (l,b,w,h) in axes coordinates.
                Does not resize other elements. Note that this overrides all colorbar
                keywords except orientation.
        scale - Scales the dG shown on the graph.

        All other keywords are passed to the colorbar.

        Returns
        -------
        a dictionary containing the following figure elements:
        "fig" - The figure
        "axis" - The image axis
        "cbaxis" - The colorbar axis, if it exists.
        "cb" - The colorbar, if it exists."""

        from matplotlib import cm, colorbar, colors
        from matplotlib.collections import PolyCollection

        fig = kwds.pop("figure", plt.gcf())
        ax = fig.add_subplot(kwds.pop("subplot", 111), projection="3d")

        kwds.copy()

        # Resolve keywords (title resolved later)
        dGs = kwds.pop("dGs", self.dgs)
        highlight = kwds.pop("highlight", [])
        classes = kwds.pop("classes", range(len(self.classes)))
        probfilter = kwds.pop("probfilter", [0.0, 1.0])
        class_size = kwds.pop("class_size", "number")
        norm = kwds.pop("norm", "auto")
        cmap = kwds.pop("cmap", cm.get_cmap("jet"))
        highlight_cmap = kwds.pop("highlight_cmap", cm.get_cmap("gray"))
        title = kwds.pop("title", True)
        p_alpha = kwds.pop("p_alpha", 0.7)
        scale = kwds.pop("scale", 1.0)

        xs = dGs * scale
        verts = []
        zs = []
        zlabels = []
        for i in classes:
            ys = [self.classprobs[dG][i] for dG in dGs]

            maxys = np.max(ys)
            if maxys >= probfilter[0] and maxys <= probfilter[1]:
                p0, p1 = ((xs[0], 0), (xs[-1], 0))  # points to close the vertices
                verts.append(np.concatenate([[p0], zip(xs, ys), [p1], [p0]]))
                zlabels.append(i)

        #  Define face colors
        fc = np.array([len(self.classes[z]) for z in zlabels])
        if class_size == "fraction":
            fc = fc / float(len(self.results))

        # Index the colormap if necessary
        if class_size == "number":
            if norm == "auto":
                indexedcolors = cmap(np.linspace(0.0, 1.0, np.max(fc)))
                cmap = colors.ListedColormap(indexedcolors)
            elif norm == "full":
                indexedcolors = cmap(np.linspace(0.0, 1.0, len(self.results)))
                cmap = colors.ListedColormap(indexedcolors)
            # A user-specified norm cannot be used to index a colormap.

        # Create proper norms for "auto" and "full" types.
        if norm == "auto":
            if class_size == "number":
                mic = np.min(fc)
                mac = np.max(fc)
                nc = mac - mic + 1
                norm = colors.BoundaryNorm(np.linspace(mic, mac + 1, nc + 1), nc)
            if class_size == "fraction":
                norm = colors.Normalize()
                norm.autoscale(fc)
        elif norm == "full":
            mcolor = len(self.results)
            if class_size == "number":
                norm = colors.BoundaryNorm(np.linspace(0, mcolor + 1, mcolor + 2), mcolor + 1)
            if class_size == "fraction":
                norm = colors.Normalize(0.0, 1.0)

        zs = np.arange(len(zlabels))

        poly = PolyCollection(verts, facecolors=cmap(norm(fc)), closed=False)
        poly.set_alpha(p_alpha)
        ax.add_collection3d(poly, zs=zs, zdir="y")

        # Highlight values of interest
        color_idx = np.linspace(0, 1, len(highlight))
        for dG, ci in zip(highlight, color_idx):
            for z_logical, z_plot in zip(zlabels, zs):
                ax.plot(
                    [dG, dG],
                    [z_plot, z_plot],
                    [0, self.classprobs[dG][z_logical]],
                    color=highlight_cmap(ci),
                    alpha=p_alpha,
                )

        ax.set_xlabel("dG")
        ax.set_xlim3d(dGs[0] * scale, dGs[-1] * scale)
        ax.set_ylabel("Class")
        ax.set_ylim3d(zs[0], zs[-1])
        ax.set_yticks(zs)
        ax.set_yticklabels([str(z) for z in zlabels])
        ax.set_zlabel("Akaike probability")
        ax.set_zlim3d(0, 1)

        if title is True:
            title = (
                "Class probabilities\n\
                    Max probabilities in %s\n\
                    %i/%i classes with %i/%i models displayed"
                % (
                    probfilter,
                    len(zs),
                    len(self.classes),
                    np.sum([len(self.classes[z]) for z in zlabels]),
                    len(self.results),
                )
            )

        if title is not False:
            fig.suptitle(title)

        # Add colorbar
        if "cbpos" in kwds:
            cbpos = kwds.pop("cbpos")
            plt.tight_layout()  # do it before cbaxis, so colorbar is ignored.
            transAtoF = ax.transAxes + fig.transFigure.inverted()
            rect = transforms.Bbox.from_bounds(*cbpos).transformed(transAtoF).bounds
            cbaxis = fig.add_axes(rect)

            # Remove all colorbar.make_axes keywords except orientation
            kwds = eatkwds("fraction", "pad", "shrink", "aspect", "anchor", "panchor", **kwds)
        else:
            kwds.setdefault("shrink", 0.75)

            # In matplotlib 1.1.0 make_axes_gridspec ignores anchor and panchor keywords.
            # Eat these keywords for now.
            kwds = eatkwds("anchor", "panchor", **kwds)
            cbaxis, kwds = colorbar.make_axes_gridspec(ax, **kwds)  # gridspec allows tight_layout
            plt.tight_layout()  # do it after cbaxis, so colorbar isn't ignored

        cb = colorbar.ColorbarBase(cbaxis, cmap=cmap, norm=norm, **kwds)

        if class_size == "number":
            cb.set_label("Models in class")
        elif class_size == "fraction":
            cb.set_label("Fraction of models in class")

        return {"fig": fig, "axis": ax, "cb": cb, "cbaxis": cbaxis}

    def get_model(self, dG, **kwds):
        """Return index of best model of best class at given dG.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Keywords
        --------
        corder - Which class to get based on AIC. Ordered from best to worst from 0 (the default).
        morder - Which model to get based on AIC. Ordered from best to worst from 0 (the default).
                 Returns a model from a class, or from the collection of all models if classes are ignored.
        cls - Override corder with a specific class index, or None to ignore classes entirely.

        Returns
        -------
        int
            Index of best model of best class at given dG.
        """
        corder = kwds.pop("corder", 0)
        morder = kwds.pop("morder", 0)
        if "cls" in kwds:
            cls = kwds["cls"]
        else:
            cls = self.get_class(dG, corder=corder)

        if cls is None:
            return self.sortedprobs[dG][-1 - morder]
        else:
            return self.sortedclasses[dG][cls][-1 - morder]

    def get_class(self, dG, **kwds):
        """Return index of best class at given dG.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Keywords
        --------
        corder - Which class to get based on AIC. Ordered from best to worst from 0 (the default).

        Returns
        -------
        int
            Index of best model of best class at given dG.
        """
        corder = kwds.pop("corder", 0)
        return self.sortedclassprobs[dG][-1 - corder]  # index of corderth best class

    def get_prob(self, dG, **kwds):
        """Return Akaike probability of best model of best class at given dG.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Keywords
        --------
        corder - Which class to get based on AIC. Ordered from best to worst from 0 (the default).
        morder - Which model to get based on AIC. Ordered from best to worst from 0 (the default).
                 Returns a model from a class, or from the collection of all models if classes are ignored.
        cls - Override corder with a specific class index, or None to ignore classes entirely.

        Returns
        -------
        array-like
            The sequence of Akaike probability of best model of best class at given dG.
        """
        idx = self.get_model(dG, **kwds)
        if "cls" in kwds and kwds["cls"] is None:
            return self.aicprobs[dG][idx]
        else:
            cls_idx = self.classes_idx[idx]
            return self.classprobs[dG][cls_idx]

    def get_nfree(self, dG, **kwds):
        """Return number of free parameters of best model of best class at given dG.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Keywords
        --------
        corder - Which class to get based on AIC. Ordered from best to worst from 0 (the default).
        morder - Which model to get based on AIC. Ordered from best to worst from 0 (the default).
                 Returns a model from a class, or from the collection of all models if classes are ignored.
        cls - Override corder with a specific class index, or None to ignore classes entirely.

        Returns
        -------
        int
            Number of free parameters of best model of best class at given dG.
        """
        idx = self.get_model(dG, **kwds)
        model = self.results[idx][1]
        baseline = self.results[idx][2]
        return model.npars(count_fixed=False) + baseline.npars(count_fixed=False)

    def get_aic(self, dG, **kwds):
        """Return number of free parameters of best model of best class at given dG.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Keywords
        --------
        corder - Which class to get based on AIC. Ordered from best to worst from 0 (the default).
        morder - Which model to get based on AIC. Ordered from best to worst from 0 (the default).
                 Returns a model from a class, or from the collection of all models if classes are ignored.
        cls - Override corder with a specific class index, or None to ignore classes entirely.

        Returns
        -------
        int
            Number of free parameters of best model of best class at given dG.
        """
        idx = self.get_model(dG, **kwds)
        return self.aics[dG][idx].stat

    def get(self, dG, *args, **kwds):
        """Return tuple of values corresponding to string arguments for best model of best class at given dG.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Permissible arguments: "aic", "class", "dG", "model", "nfree", "prob"
        ("dG" simply returns the provided dG value)

        Keywords
        --------
        corder - Which class to get based on AIC. Ordered from best to worst from 0 (the default).
        morder - Which model to get based on AIC. Ordered from best to worst from 0 (the default).
                 Returns a model from a class, or from the collection of all models if classes are ignored.
        cls - Override corder with a specific class index, or None to ignore classes entirely.

        Returns
        -------
        tuple
            The values corresponding to string arguments for best model of best class at given dG.
        """
        fdict = {
            "aic": self.get_aic,
            "class": self.get_class,
            "dg": lambda x: x,
            "model": self.get_model,
            "nfree": self.get_nfree,
            "prob": self.get_prob,
        }
        values = []
        for a in args:
            values.append(fdict[a].__call__(dG, **kwds))
        return tuple(values)

    def maxprobdG_byclass(self, model):
        """Return the post-hoc dG for which the given model's Akaike probability is maximized.

        Each model is mapped to its class' best member.

        Parameters
        ----------
        model : array-like
            The model to get the post-hoc dG.

        Returns
        -------
        array-like
            The post-hoc dG for the given model where the given model's Akaike probability is maximized.
        """
        cls = self.classes_idx[model]
        probs = [self.classprobs[dg][cls] for dg in self.dgs]
        prob_idx = np.argmax(probs)
        return self.dgs[prob_idx]

    def maxprobdG_bymodel(self, model):
        """Return the post-hoc dG for which the given model's Akaike probability is maximized.
        Classes are not considered.

        Parameters
        ----------
        model : array-like
            The model to get the post-hoc dG.

        Returns
        -------
        array-like
            The post-hoc dG by the given model's Akaike probability is maximize
        """
        probs = [self.aicprobs[dg][model] for dg in self.dgs]
        prob_idx = np.argmax(probs)
        return self.dgs[prob_idx]

    def maxprobmodel_byclass(self, dG):
        """Calculate the model which maximizes probability at given dG.

        The best class is mapped to its best model.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Returns
        -------
        float
            The model mapped by class which maximizes probability at given dG."""
        cls = self.sortedclassprobs[dG][-1]
        m = self.sortedclasses[dG][cls][-1]
        return m

    def maxprobmodel_bymodel(self, dG):
        """Return the model which maximizes probability at given dG.
        Classes are not considered.

        Parameters
        ----------
        dG : array-like
            The uncertainty used to calculate probabilities

        Returns
        -------
        model : array-like
            The model which maximizes probability at given dG."""
        # Note that if there are identical models this returns the one of greatest dg.
        return self.sortedprobs[dG][-1]
