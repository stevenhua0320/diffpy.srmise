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
"""Defines ModelCluster class.

Classes
-------
ModelCluster: Associate a region of data to a model that describes it.
ModelCovariance: Helper class for model covariance.
"""

import logging
import re
import sys

import numpy as np

from diffpy.srmise import srmiselog
from diffpy.srmise.baselines.base import Baseline
from diffpy.srmise.modelparts import ModelParts
from diffpy.srmise.peaks.base import Peak, Peaks
from diffpy.srmise.srmiseerrors import (
    SrMiseDataFormatError,
    SrMiseEstimationError,
    SrMiseFitError,
    SrMiseUndefinedCovarianceError,
)

logger = logging.getLogger("diffpy.srmise")


class ModelCovariance(object):
    """Helper class preserves uncertainty info (full covariance matrix) for a fit model.

    This object preserves a light-weight "frozen" version of a model which can be used
    to interrogate the model about the uncertainties of its parameters.

    Note that all parameters defined by a model, including fixed ones, are included in the
    covariance matrix. Fixed parameters are defined to have 0 uncertainty.

    Methods
    -------
    """

    def __init__(self, *args, **kwds):
        """Intialize object."""
        self.cov = None  # The raw covariance matrix
        self.model = None  # ModelParts instance, so both peaks and baseline (if present)

        # Map i->[n1,n2,...] of the jth ModelPart to the n_i parameters in cov.
        self.mmap = {}

        # Map (i,j)->n of jth parameter of ith ModelPart to nth parameter in cov.
        self.pmap = {}

        # Map n->(i,j), inverse of pmap.
        self.ipmap = {}

    def clear(self):
        """Reset object."""
        self.cov = None
        self.model = None
        self.mmap = {}
        self.pmap = {}
        self.ipmap = {}

    def setcovariance(self, model, cov):
        """Set model and covariance.

        Automatically initializes covariance matrix to full set of parameters in model, even
        those listed as "fixed."

        Parameters
        ----------
        model :  ModelParts
            The ModelParts instance
        cov : ndarray
            The nxn covariance matrix for n model parameters. If the parameterization includes "fixed"
            parameters not included in the covariance matrix, the matrix is expanded to include these
            parameters with 0 uncertainty.
        """
        tempcov = np.array(cov)

        if tempcov.shape[0] != tempcov.shape[1]:
            emsg = "Parameter 'cov' must be a square matrix."
            raise ValueError(emsg)

        if tempcov.shape[0] != model.npars(True) and tempcov.shape[0] != model.npars(False):
            emsg = [
                "Parameter 'cov' must be an nxn matrix, where n is equal to the number of free ",
                "parameters in the model, or the total number of parameters (fixed and free) of ",
                "the model.",
            ]
            raise ValueError("".join(emsg))

        self.model = model.copy()

        # The free/fixed status of every parameter in the model
        free = np.concatenate([m.free for m in self.model]).ravel()

        # Create maps from model parts to index of corresponding covariance matrix row/column
        n = 0
        for i, m in enumerate(model):
            self.mmap[i] = n + np.arange(m.npars(True))
            for j, p in enumerate(m):
                self.pmap[(i, j)] = n
                self.ipmap[n] = (i, j)
                n += 1

        if n == tempcov.shape[0]:
            # All free and fixed parameters already in passed covariance matrix
            self.cov = tempcov
        else:
            # Create new covariance matrix, making sure to account for fixed pars
            self.cov = np.matrix(np.zeros((n, n)))

            i = 0
            rawi = 0
            for i in range(n):
                j = 0
                rawj = 0
                if free[i]:
                    for j in range(n):
                        if free[j]:
                            self.cov[i, j] = cov[rawi, rawj]
                            rawj += 1
                    rawi += 1

    def transform(self, in_format, out_format, **kwds):
        """Transform parameters and covariance matrix under specified change of variables.

        By default this change applies to all parameters of the model.  If the specified transformation
        is invalid for a given ModelPart the original parameterization is maintained for that part.

        This function assumes that the derivative of every variable under this change depends only on the
        other parameters in its own part.  Essentially, each model part has it own non-zero gradient matrix for
        any change of variables, but the gradient matrix between different parts is 0.

        Note that transformed parameters may mix previously fixed and free parameters, and consequently
        the number of each kind is not necessarily preserved.  The user is cautioned to interpret a transformed
        covariance matrix carefully.  For example, the apparent degrees of freedom in the transformed
        covariance matrix may not coincide with the degrees of freedom during model fitting.

        Parameters
        ----------
        in_format : str
            The current format of parameters
        out_format : str
            The new format for parameters

        Keywords
        --------
        parts - Specify which model part, by index, to transform.  Defaults to all parts.  Alternately,
                "peaks" to transform all but the last part, and "baseline" to convert only the last part.
        """
        if self.cov is None:
            emsg = "Cannot transform undefined covariance matrix."
            raise SrMiseUndefinedCovarianceError(emsg)

        if "parts" in kwds:
            if kwds["parts"] == "peaks":
                parts = range(len(self.model) - 1)
            elif kwds["parts"] == "baseline":
                parts = [-1]
            else:
                parts = kwds["parts"]
        else:
            parts = range(len(self.model))

        # Calculate V_y = G Transpose(V_x) G
        # where V_y is the covariance matrix in terms of the parameterization y,
        # V_x is the current covariance matrix, and G is the gradient matrix of
        # the variable transformation x->y.  That is, G_ij = dy_i/dx_j.
        #
        # Since we assume that the parameterization of each ModelPart is independent
        # of every other ModelPart, we create the full gradient matrix from the smaller
        # gradient submatrices of each ModelPart.

        g = np.identity(self.cov.shape[0])

        for i in parts:
            start = self.mmap[i][0]
            stop = self.mmap[i][-1] + 1
            p = self.model[i]
            try:
                subg = p.owner().transform_derivatives(p.pars, in_format, out_format)
            except NotImplementedError:
                logger.warning(
                    "Transformation gradient not implemented for part %i: %s.  Ignoring transformation."
                    % (i, str(p))
                )
                subg = np.identity(p.npars(True))
            except Exception as e:
                logger.warning(
                    "Transformation gradient failed for part %i: %s. "
                    "Failed with message %s. Ignoring transformation." % (i, str(p), str(e))
                )
                subg = np.identity(p.npars(True))

            # Now transform the parameters to match
            try:
                p.pars = p.owner().transform_parameters(p.pars, in_format, out_format)
            except Exception as e:
                logger.warning(
                    "Parameter transformation failed for part %i: %s. "
                    "Failed with message %s. Ignoring transformation." % (i, str(p), str(e))
                )
                subg = np.identity(p.npars(True))

            # Update the global gradient matrix
            g[start:stop, start:stop] = subg

        g = np.matrix(g)
        self.cov = np.array(g * np.matrix(self.cov).transpose() * g)

        return

    def getcorrelation(self, i, j):
        """Return the correlation between variables i and j, Corr_ij=Cov_ij/(sigma_i sigma_j)

        The variables may be specified as integers, or as a two-component tuple of integers (l, m)
        which indicate the mth parameter in peak l.

        The standard deviation of fixed parameters is 0, in which case the correlation is
        undefined, but return 0 for simplicity.

        Parameters
        ----------
        i : int
            The index of variable in peak mapping
        j : int
            The index of variable in peak mapping

        Returns
        -------
        float
            The correlation between variables i and j
        """
        if self.cov is None:
            emsg = "Cannot get correlation on undefined covariance matrix."
            raise SrMiseUndefinedCovarianceError(emsg)

        # Map peak/parameter tuples to the proper index
        i1 = self.pmap[i] if i in self.pmap else i
        j1 = self.pmap[j] if j in self.pmap else j

        if self.cov[i1, i1] == 0.0 or self.cov[j1, j1] == 0.0:
            return 0.0  # Avoiding undefined quantities is sensible in this context.
        else:
            return self.cov[i1, j1] / (np.sqrt(self.cov[i1, i1]) * np.sqrt(self.cov[j1, j1]))

    def getvalue(self, i):
        """Return value of parameter i.

        The variable may be specified as an integer, or as a two-component tuple of integers (l, m)
        which indicate the mth parameter of modelpart l.
        """
        (l, m) = i if i in self.pmap else self.ipmap[i]
        return self.model[l][m]

    def getuncertainty(self, i):
        """Return uncertainty of parameter i.

        The variable may be specified as an integer, or as a two-component tuple of integers (l, m)
        which indicate the mth parameter of modelpart l.

        Parameters
        ----------
        i : int
            The index of variable in peak mapping

        Returns
        -------
        float
            The uncertainty of variable at index i.
        """
        (l, m) = i if i in self.pmap else self.ipmap[i]
        return np.sqrt(self.getcovariance(i, i))

    def getcovariance(self, i, j):
        """Return the covariance between variables i and j.

        The variables may be specified as integers, or as a two-component tuple of integers (l, m)
        which indicate the mth parameter of modelpart l.

        Parameters
        ----------
        i : int
            The index of variable in peak mapping
        j : int
            The index of variable in peak mapping

        Returns
        -------
        float
            The covariance between variables at indeex i and j.
        """
        if self.cov is None:
            emsg = "Cannot get correlation on undefined covariance matrix."
            raise SrMiseUndefinedCovarianceError(emsg)

        # Map peak/parameter tuples to the proper index
        i1 = self.pmap[i] if i in self.pmap else i
        j1 = self.pmap[j] if j in self.pmap else j

        return self.cov[i1, j1]

    def get(self, i):
        """Return (value, uncertainty) tuple for parameter i.

        The variable may be specified as an integer, or as a two-component tuple of integers (l, m)
        which indicate the mth parameter of modelpart l.

        Parameters
        ----------
        i : int
            The index of variable in peak mapping

        Returns
        -------
        (float, float)
            The value and uncertainty of variable at index i.
        """
        return (self.getvalue(i), self.getuncertainty(i))

    def correlationwarning(self, threshold=0.8):
        """Report distinct variables with magnitude of correlation greater than threshold.

        Returns a list of tuples (i, j, c), where i and j are tuples indicating
        the modelpart and parameter indices of the correlated variables, and
        c is their correlation.

        Parameters
        ----------
        threshold : float
            A real number between 0 and 1.

        Returns
        -------
        tuple (i, j, c)
        Indices of the modelpart and their correlations.
        """
        if self.cov is None:
            emsg = "Cannot calculate correlation on undefined covariance matrix."
            raise SrMiseUndefinedCovarianceError(emsg)

        correlated = []
        for i in range(self.cov.shape[0]):
            for j in range(i + 1, self.cov.shape[0]):
                c = self.getcorrelation(i, j)
                if c and np.abs(c) > threshold:  # filter out None values
                    correlated.append((self.ipmap[i], self.ipmap[j], c))
        return correlated

    def __str__(self):
        """Return string of value (uncertainty) pairs for all parameters."""
        if self.model is None or self.cov is None:
            return "Model and/or Covariance matrix undefined."
        lines = []
        for i, m in enumerate(self.model):
            lines.append("  ".join([self.prettypar((i, j)) for j in range(len(m))]))
        return "\n".join(lines)

    def prettypar(self, i):
        """Return string 'value (uncertainty)' for parameter i.

        The variable may be specified as an integer, or as a two-component tuple of integers (l, m)
        which indicate the mth parameter of modelpart l.

        Parameters
        ----------
        i : int
            The index of variable in peak mapping

        Returns
        -------
        str
        'value (uncertainty)' for variable at index i.
        """
        if self.model is None or self.cov is None:
            return "Model and/or Covariance matrix undefined."
        k = i if i in self.ipmap else self.pmap[i]
        return "%.5e (%.5e)" % (self.getvalue(k), np.sqrt(self.getcovariance(k, k)))


# End of class ModelCovariance


class ModelCluster(object):
    """Associate a contiguous cluster of data with an appropriate model.

    A ModelCluster instance is the basic unit of diffpy.srmise, combining data and
    a model with the basic tools for controlling their interaction.

    Methods
    -------
    addexternalpeaks: Add peaks to model, and their value to the data.
    augment: Add peaks to model, only keeping those which improve model quality.
    change_slice: Change the range of the data considered within cluster.
    cleanfit: Remove extremely poor or non-contributing peaks from model.
    contingent_fit: Fit model to data if size of cluster has significantly increased
                    since previous fit.
    factory: Static method to recreate a ModelCluster from a string.
    join_adjacent: Static method to combine two ModelClusters.
    npars: Return total number of parameters in model.
    replacepeaks: Add and/or delete peaks in model
    deletepeak: Delete a single peak in model.
    estimatepeak: Add single peak to model with no peaks.
    fit: Fit the model to the data
    plottable: Return list of arguments for convenient plotting with matplotlib
    plottable_residual: Return list of argument for convenient plotting of residual
                        with matplotlib.
    prune: Remove peaks from model which maximize quality.
    reduce_to: If value(x)>y, change model to so that value(x)=y
    quality: Return ModelEvaluator instance that calculates quality of model to data.
    value: Return value of the model plus baseline
    valuebl: Return value of the baseline
    writestr: Return string representation of self.

    """

    def __init__(self, model, *args, **kwds):
        """Intialize explicitly, or from existing ModelCluster.

        Parameters
        ----------
        model : (lists of) ModelCluster instance
            The ModelCluster instances to be clustered.
            If it is None, then a ModelCluster object is created.
        baseline : Baseline object
            The Baseline object, if it is None, set to 0.
        r_data : array-like
            The numpy array of r coordinates
        y_data : array-like
            The numpy array of y values
        y_error : array-like
            The numpy array of uncertainties in y
        cluster_slice : slice object
            The slice object defining the range of cluster. If the input is None,
            then it will take the entire range.
        error_method : ErrorEvaluator subclass
            The error evaluator to use to calculate quality of model to data.
        peak_funcs : a sequence of PeakFunction instances
            The peak instances to use to calculate the cluster of data.
        """
        self.last_fit_size = 0
        self.slice = None
        self.size = None
        self.r_cluster = None
        self.y_cluster = None
        self.error_cluster = None
        self.never_fit = None

        if isinstance(model, ModelCluster):
            orig = model
            s = orig.slice
            self.model = orig.model.copy()
            if orig.baseline is None:
                self.baseline = None
            else:
                self.baseline = orig.baseline.copy()
            self.r_data = orig.r_data
            self.y_data = orig.y_data
            self.y_error = orig.y_error
            self.change_slice(slice(s.start, s.stop, s.step))
            self.error_method = orig.error_method
            self.peak_funcs = list(orig.peak_funcs)
            return
        else:  # Explicit creation
            if model is None:
                self.model = Peaks([])
            else:
                self.model = model

            self.baseline = args[0]
            self.r_data = args[1]
            self.y_data = args[2]
            self.y_error = args[3]

            # Sets self.slice, self.size, self.r_cluster, self.y_cluster,
            # and self.error_cluster.
            if args[4] is None:
                self.change_slice(slice(len(self.r_data)))
            else:
                self.change_slice(args[4])

            self.error_method = args[5]
            self.peak_funcs = args[6]
            return

    def copy(self):
        """Return copy of this ModelCluster.

        Equivalent to ModelCluster(self)"""
        return ModelCluster(self)

    def addexternalpeaks(self, peaks):
        """Add peaks (and their value) to self.

        Parameters
        ----------
        peaks : A Peaks object
            The peaks to be added

        Returns
        -------
        None
        """
        self.replacepeaks(peaks)
        self.y_data += peaks.value(self.r_data)
        self.y_cluster = self.y_data[self.slice]

    def writestr(self, **kwds):
        """Return partial string representation.

        Keywords
        --------
        pfbaselist - List of peak function bases. Otherwise, define list from self.
        blfbaselist - List of baseline function bases.Otherwise, define list from self.
        """
        from diffpy.srmise.basefunction import BaseFunction

        if "pfbaselist" in kwds:
            pfbaselist = kwds["pfbaselist"]
            writepf = False
        else:
            pfbaselist = [p.owner() for p in self.model]
            pfbaselist.extend(self.peak_funcs)
            pfbaselist = list(set(pfbaselist))
            pfbaselist = BaseFunction.safefunctionlist(pfbaselist)
            writepf = True

        if "blfbaselist" in kwds:
            blfbaselist = kwds["blfbaselist"]
            writeblf = False
        else:
            blfbaselist = BaseFunction.safefunctionlist([self.baseline.owner()])
            writeblf = True

        lines = []

        if self.peak_funcs is None:
            lines.append("peak_funcs=None")
        else:
            lines.append("peak_funcs=%s" % repr([pfbaselist.index(p) for p in self.peak_funcs]))
        if self.error_method is None:
            lines.append("ModelEvaluator=None")
        else:
            lines.append("ModelEvaluator=%s" % self.error_method.__name__)

        lines.append("slice=%s" % repr(self.slice))

        # Indexed baseline functions (unless externally provided)
        if writeblf:
            lines.append("## BaselineFunctions")
            for i, bf in enumerate(blfbaselist):
                lines.append("# BaselineFunction %s" % i)
                lines.append(bf.writestr(blfbaselist))

        # Indexed peak functions (unless externally provided)
        if writepf:
            lines.append("## PeakFunctions")
            for i, pf in enumerate(pfbaselist):
                lines.append("# PeakFunction %s" % i)
                lines.append(pf.writestr(pfbaselist))

        lines.append("# BaselineObject")
        if self.baseline is None:
            lines.append("None")
        else:
            lines.append(self.baseline.writestr(blfbaselist))

        lines.append("## ModelPeaks")
        if self.model is None:
            lines.append("None")
        else:
            for m in self.model:
                lines.append("# ModelPeak")
                lines.append(m.writestr(pfbaselist))

        # Raw data in modelcluster.
        lines.append("### start data")
        lines.append("#L r y dy")
        for i in range(len(self.r_data)):
            lines.append("%g %g %g" % (self.r_data[i], self.y_data[i], self.y_error[i]))

        datastring = "\n".join(lines) + "\n"
        return datastring

    @staticmethod
    def factory(mcstr, **kwds):
        """Create ModelCluster from string.

        Keywords
        --------
        pfbaselist : List of peak function bases
        blfbaselist : List of baseline function bases
        """
        from diffpy.srmise.basefunction import BaseFunction

        if "pfbaselist" in kwds:
            readpf = False
            pfbaselist = kwds["pfbaselist"]
        else:
            readpf = True

        if "blfbaselist" in kwds:
            readblf = False
            blfbaselist = kwds["blfbaselist"]
        else:
            readblf = True

        # The major components are:
        # - Header
        # - BaselineFunctions (optional)
        # - Peakfunctions (optional)
        # - BaselineObject
        # - ModelPeaks
        # - StartData

        # find data section, and what information it contains
        res = re.search(r"^#+ start data\s*(?:#.*\s+)*", mcstr, re.M)
        if res:
            start_data = mcstr[res.end() :].strip()
            start_data_info = mcstr[res.start() : res.end()]
            header = mcstr[: res.start()]
        res = re.search(r"^(#+L.*)$", start_data_info, re.M)
        if res:
            start_data_info = start_data_info[res.start() : res.end()].strip()
        hasr = False
        hasy = False
        hasdy = False
        res = re.search(r"\br\b", start_data_info)
        if res:
            hasr = True
        res = re.search(r"\by\b", start_data_info)
        if res:
            hasy = True
        res = re.search(r"\bdy\b", start_data_info)
        if res:
            hasdy = True

        # Model
        res = re.search(r"^#+ ModelPeaks.*$", header, re.M)
        if res:
            model_peaks = header[res.end() :].strip()
            header = header[: res.start()]

        # Baseline Object
        res = re.search(r"^#+ BaselineObject\s*(?:#.*\s+)*", header, re.M)
        if res:
            baselineobject = header[res.end() :].strip()
            header = header[: res.start()]

        # Peak functions
        if readpf:
            res = re.search(r"^#+ PeakFunctions.*$", header, re.M)
            if res:
                peakfunctions = header[res.end() :].strip()
                header = header[: res.start()]

        # Baseline functions
        if readblf:
            res = re.search(r"^#+ BaselineFunctions.*$", header, re.M)
            if res:
                baselinefunctions = header[res.end() :].strip()
                header = header[: res.start()]

        # Instantiating baseline functions
        if readblf:
            blfbaselist = []
            res = re.split(r"(?m)^#+ BaselineFunction \d+\s*(?:#.*\s+)*", baselinefunctions)
            for s in res[1:]:
                blfbaselist.append(BaseFunction.factory(s, blfbaselist))

        # Instantiating peak functions
        if readpf:
            pfbaselist = []
            res = re.split(r"(?m)^#+ PeakFunction \d+\s*(?:#.*\s+)*", peakfunctions)
            for s in res[1:]:
                pfbaselist.append(BaseFunction.factory(s, pfbaselist))

        # Instantiating header data
        # peak_funcs
        res = re.search(r"^peak_funcs=(.*)$", header, re.M)
        peak_funcs = eval(res.groups()[0].strip())
        if peak_funcs is not None:
            peak_funcs = [pfbaselist[i] for i in peak_funcs]

        # error_method
        res = re.search(r"^ModelEvaluator=(.*)$", header, re.M)
        __import__("diffpy.srmise.modelevaluators")
        module = sys.modules["diffpy.srmise.modelevaluators"]
        error_method = getattr(module, res.groups()[0].strip())

        # slice
        res = re.search(r"^slice=(.*)$", header, re.M)
        cluster_slice = eval(res.groups()[0].strip())

        # Instantiating BaselineObject
        if re.match(r"^None$", baselineobject):
            baseline = None
        else:
            baseline = Baseline.factory(baselineobject, blfbaselist)

        # Instantiating model
        model = Peaks()
        res = re.split(r"(?m)^#+ ModelPeak\s*(?:#.*\s+)*", model_peaks)
        for s in res[1:]:
            model.append(Peak.factory(s, pfbaselist))

        # Instantiating start data
        # read actual data - r, y, dy
        arrays = []
        if hasr:
            r_data = []
            arrays.append(r_data)
        else:
            r_data = None
        if hasy:
            y_data = []
            arrays.append(y_data)
        else:
            y_data = None
        if hasdy:
            y_error = []
            arrays.append(y_error)
        else:
            y_error = None
        # raise SrMiseDataFormatError if something goes wrong
        try:
            for line in start_data.split("\n"):
                lines = line.split()
                if len(arrays) != len(lines):
                    emsg = "Number of value fields does not match that given by '%s'" % start_data_info
                    raise IndexError(emsg)
                for a, v in zip(arrays, line.split()):
                    a.append(float(v))
        except (ValueError, IndexError) as err:
            raise SrMiseDataFormatError(err)
        if hasr:
            r_data = np.array(r_data)
        if hasy:
            y_data = np.array(y_data)
        if hasdy:
            y_error = np.array(y_error)

        return ModelCluster(
            model,
            baseline,
            r_data,
            y_data,
            y_error,
            cluster_slice,
            error_method,
            peak_funcs,
        )

    @staticmethod
    def join_adjacent(m1, m2):
        """Create new ModelCluster from two adjacent ones.

        Suspected duplicate peaks are removed, and peaks may be adjusted if
        their sum where the clusters meet exceeds the data.  m1 and m2 are
        unchanged.

        Parameters
        ----------
        m1 : ModelCluster instance
            The first ModelCluster instance.
        m2 : ModelCluster instance
            The second ModelCluster instance.

        Returns
        -------
        ModelCluster instance
            The new ModelCluster instance between m1 and m2.
        """
        # Check for members that must be shared.
        if not (m1.r_data is m2.r_data):
            emsg = "Cannot join ModelClusters that do not share r_data."
            raise ValueError(emsg)
        if not (m1.y_data is m2.y_data):
            emsg = "Cannot join ModelClusters that do not share y_data."
            raise ValueError(emsg)
        if not (m1.y_error is m2.y_error):
            emsg = "Cannot join ModelClusters that do not share y_error."
            raise ValueError(emsg)
        if not (m1.error_method is m2.error_method):
            emsg = "Cannot join ModelClusters that do not share error_method."
            raise ValueError(emsg)
        if not (m1.baseline == m2.baseline):
            emsg = "Cannot join ModelClusters that do not have equivalent baselines."
            raise ValueError(emsg)

        m1_ids = m1.slice.indices(len(m1.r_data))
        m2_ids = m2.slice.indices(len(m1.r_data))

        if m1_ids[0] < m2_ids[0]:
            left = m1
            right = m2
            left_ids = m1_ids
            right_ids = m2_ids
        else:
            left = m2
            right = m1
            left_ids = m2_ids
            right_ids = m1_ids

        if not right_ids[0] == left_ids[1]:
            raise ValueError("Given ModelClusters are not adjacent.")

        new_slice = slice(left_ids[0], right_ids[1], 1)

        # Approximately where the clusters meet.
        border_x = 0.5 * (left.r_data[left_ids[1] - 1] + right.r_data[right_ids[0]])
        border_y = 0.5 * (left.y_data[left_ids[1] - 1] + right.y_data[right_ids[0]])

        if len(m1.model) > 0 and len(m2.model) > 0:
            new_model = left.model.copy()
            new_model.extend(right.model.copy())
            new_ids = new_model.argsort(key="position")
            new_model = Peaks([new_model[i] for i in new_ids])

            # Compare raw order with sorted order.  Peaks which are *both* out
            # of the expected order AND located on the "wrong" side of
            # border_x are removed.  The highly unlikely case of two peaks
            # exactly at the border is also handled.
            for i in reversed(range(len(new_model))):
                if new_model[i]["position"] == border_x and i > 0 and new_model[i - 1]["position"] == border_x:
                    del new_model[i]
                elif new_ids[i] != i:
                    if (new_model[i]["position"] > border_x and new_ids[i] < len(left.model)) and (
                        new_model[i]["position"] < border_x and new_ids[i] >= len(left.model)
                    ):
                        del new_model[i]

            # Likely to improve any future fitting
            new_model.match_at(border_x, border_y)
        elif len(m1.model) > 0:
            new_model = m1.model.copy()
        else:  # Only m2 has entries, or both are empty
            new_model = m2.model.copy()

        peak_funcs = list(set(m1.peak_funcs) | set(m2.peak_funcs))  # "Union"
        return ModelCluster(
            new_model,
            m1.baseline,
            m1.r_data,
            m1.y_data,
            m1.y_error,
            new_slice,
            m1.error_method,
            peak_funcs,
        )

    def change_slice(self, new_slice):
        """Change the slice which represents the extent of a cluster.

        Parameters
        ----------
        new_slice : slice object
            The new slice to change.

        Returns
        -------
        None
        """
        old_slice = self.slice
        self.slice = new_slice
        self.r_cluster = self.r_data[new_slice]
        self.y_cluster = self.y_data[new_slice]
        self.error_cluster = self.y_error[new_slice]
        self.size = len(self.r_cluster)

        # Determine if any data in the cluster is above the error threshold.
        # If not, then this cluster is never fit.  This is updated as the
        # cluster expands.  This is not necessarily a strict measure because
        # data removed from a cluster are not considered when updating, but
        # this does not occur in the normal operation of peak extraction.
        # Therefore never_fit will only be incorrect if the removed data
        # includes the only parts rising above the error threshold, so a
        # cluster is never wrongly skipped, but it might be fit when it isn't
        # necessary.
        y_data_nobl = self.y_data - self.valuebl(self.r_data)
        y_cluster_nobl = y_data_nobl[new_slice]
        if old_slice is None:
            self.never_fit = max(y_cluster_nobl - self.error_cluster) < 0
        else:
            # check if slice has expanded on the left
            if self.never_fit and self.slice.start < old_slice.start:
                left_slice = slice(self.slice.start, old_slice.start)
                self.never_fit = max(y_data_nobl[left_slice] - self.y_error[left_slice]) < 0
            # check if slice has expanded on the right
            if self.never_fit and self.slice.stop > old_slice.stop:
                right_slice = slice(old_slice.stop, self.slice.stop)
                self.never_fit = max(y_data_nobl[right_slice] - self.y_error[right_slice]) < 0

        return

    def npars(self, count_baseline=True, count_fixed=True):
        """Return number of parameters in model and baseline.

        Parameters
        ----------
        count_baseline : bool
            The boolean determines whether to count parameters from baseline. Default is True.
        count_fixed : bool
        The boolean determines whether to include non-free parameters. Default is True.

        Returns
        -------
        n : int
            The number of parameters in model and baseline.
        """
        n = self.model.npars(count_fixed=count_fixed)
        if count_baseline and self.baseline is not None:
            n += self.baseline.npars(count_fixed=count_fixed)
        return n

    def replacepeaks(self, newpeaks, delslice=slice(0, 0)):
        """Replace peaks given by delslice by those in newpeaks.

        Parameters
        ----------
        newpeaks : Peak instance
            The peak that id added to each existing peak to cluster.
        delslice : Peak instance
            The existing peaks given by slice object are deleted.

        Returns
        -------
        None
        """
        for p in self.model[delslice]:
            if not p.removable:
                raise ValueError("delslice specified non-removable peaks.")
        self.model[delslice] = newpeaks
        self.model.sort(key="position")
        return

    def deletepeak(self, idx):
        """Delete the peak at the given index.

        Parameters
        ----------
        idx : int
            Index of peak to delete.

        Returns
        -------
        None
        """
        self.replacepeaks([], slice(idx, idx + 1))

    def estimatepeak(self):
        """Attempt to add single peak to empty cluster.  Return True if successful.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        # STUB!!! ###
        # Currently only a single peak function is supported.  Dynamic
        # selection from multiple types may require additional support
        # within peak functions themselves.  The simplest method would
        # be estimating and fitting each possible peak type to the data
        # and seeing which works best, but for small clusters evaluating
        # model quality is generally unreliable, and most peak shapes will
        # be approxiately equally good anyway.
        if len(self.model) > 0:
            # throw some exception
            pass
        selected = self.peak_funcs[0]
        estimate = selected.estimate_parameters(self.r_cluster, self.y_cluster - self.valuebl())

        if estimate is not None:
            newpeak = selected.actualize(estimate, "internal")
            logger.info("Estimate: %s" % newpeak)
            self.replacepeaks(Peaks([newpeak]))
            return True
        else:
            return False

    def fit(
        self,
        justify=False,
        ntrials=0,
        fitbaseline=False,
        estimate=True,
        cov=None,
        cov_format="default_output",
    ):
        """Perform a chi-square fit of the model to data in cluster.

        Parameters
        ----------
        justify : bool
            Revert to initial model (if one exists) if new model
            has only a single peak and the quality of the fit suggests
            additional peaks are present. Default is False.
        ntrials : int
            The maximum number of function evaluations.
            '0' indicates the fitting algorithm's default.
        fitbaseline : bool
            Whether to fit baseline along with peaks. Default is False.
        estimate : bool
            Estimate a single peak from data if model is empty. Default is True.
        cov : ModelCovariance or None
            Optional ModelCovariance object preserves covariance information.
        cov_format : str
            Parameterization to use in cov.

        Returns
        -------
        ModelEvaluator or None
        If fitting changes a model, return ModelEvaluator instance.  Otherwise
        return None.
        """
        if self.never_fit:
            return None
        if len(self.model) == 0:
            # Attempt to add a first peak to the cluster
            if estimate:
                try:
                    self.estimatepeak()
                except SrMiseEstimationError:
                    logger.info("Fit: No model to fit, estimation not possible.")
                    return
            else:
                logger.info("Fit: No model to fit.")
                return

        orig_qual = self.quality()
        orig_model = self.model.copy()
        if self.baseline is None:
            orig_baseline = None
        else:
            orig_baseline = self.baseline.copy()
        self.last_fit_size = self.size

        if fitbaseline and self.baseline is not None and self.baseline.npars(count_fixed=False) > 0:
            y_datafit = self.y_data
            fmodel = ModelParts(self.model)
            fmodel.append(self.baseline)
        else:
            y_datafit = self.y_data - self.valuebl(self.r_data)
            fmodel = self.model

        try:
            fmodel.fit(
                self.r_data,
                y_datafit,
                self.y_error,
                self.slice,
                ntrials,
                cov,
                cov_format,
            )
        except SrMiseFitError as e:
            logger.debug("Error while fitting cluster: %s\nReverting to original model.", e)
            self.model = orig_model
            self.baseline = orig_baseline
            return None

        if fitbaseline and self.baseline is not None and self.baseline.npars(count_fixed=False) > 0:
            self.model = Peaks(fmodel[:-1])
            self.baseline = fmodel[-1]
        else:
            self.model = fmodel

        self.model.sort()
        self.cleanfit()
        new_qual = self.quality()

        # Test for fit improvement
        if new_qual < orig_qual:
            # either fit blew up (and leastsq didn't notice) or the fit had already converged.
            msg = [
                "ModelCluster.fit() warning: fit seems not to have improved.",
                "Reverting to original model.",
                "----------",
                "New Quality: %s",
                "Original Quality: %s" "%s",
                "----------",
            ]
            logger.debug("\n".join(msg), new_qual.stat, orig_qual.stat, self.model)

            self.model = orig_model
            self.baseline = orig_baseline
            return None

        # Short version: When justify=True a single peak is no longer fit
        # after a second peak could potentially explain its residual.
        #
        # This is a tricky point.  It is often the case while fitting that an
        # intially good peak becomes less accurate due to greater overlap at
        # the edges of the cluster, even as its (calculated) quality improves.
        # This may make combining clusters later more difficult, and so test
        # if the degree by which the new fit is off could perhaps be adequately
        # explained by adding a second peak instead.  If so, reverting to the
        # original fit is less likely to obscure any hidden peaks.
        if justify and len(self.model) == 1 and len(orig_model) > 0:
            min_npars = min([p.npars for p in self.peak_funcs])
            if new_qual.growth_justified(self, min_npars):
                msg = [
                    "ModelCluster.fit(): Fit over current cluster better explained by additional peaks.",
                    "Reverting to original model.",
                ]
                logger.debug("\n".join(msg))

                self.model = orig_model
                self.baseline = orig_baseline
                return None
        return new_qual

    def contingent_fit(self, minpoints, growth_threshold):
        """Fit cluster if it has grown sufficiently large since its last fit.

        Parameters
        ----------
        minpoints : int
            The minimum number of points an empty cluster requires to fit.
        growth_threshold : float
            Fit non-empty model if (currentsize/oldsize) >= this value.

        Returns
        -------
        ModelEvaluator or None
            Return ModelEvaluator instance if fit changed, otherwise None.
        """
        if self.never_fit:
            return None
        if (self.last_fit_size > 0 and float(self.size) / self.last_fit_size >= growth_threshold) or (
            self.last_fit_size == 0 and self.size >= minpoints
        ):
            return self.fit(justify=True)
        return None

    def cleanfit(self):
        """Remove poor-quality peaks in the fit.  Return number removed."""
        # Find peaks located outside the cluster
        pos = np.array([p["position"] for p in self.model])
        left_idx = pos.searchsorted(self.r_cluster[0])
        right_idx = pos.searchsorted(self.r_cluster[-1])
        outside_idx = range(0, left_idx)
        outside_idx.extend(range(right_idx, len(self.model)))
        # inside_idx = range(left_idx, right_idx)

        # Identify outside peaks that contribute < error everywhere in cluster.
        # Must check entire cluster and not just nearest endpoint because not
        # every peak function can be assumed to have its greatest contribution
        # there, and errors are not necessarily constant.
        outside_idx = [
            i
            for i in outside_idx
            if (self.model[i].removable and max(self.model[i].value(self.r_cluster) - self.error_cluster) < 0)
        ]

        # TODO: Check for peaks that have blown up.
        # Remember to check if a peak is removable.
        blown_idx = []

        # NaN is too serious not to remove, even if removable is False, but I should look
        # into better handling anyway.
        nan_idx = [i for i in range(len(self.model)) if np.isnan(self.model[i].pars).any()]

        if len(outside_idx) > 0:
            msg = ["Following peaks outside cluster made no contribution within it and were removed:"]
            msg.extend([str(self.model[i]) for i in outside_idx])
            logger.debug("\n".join(msg))

        if len(nan_idx) > 0:
            msg = ["Following peaks include non-numerical parameters and were removed:"]
            msg.extend([str(self.model[i]) for i in nan_idx])
            logger.debug("\n".join(msg))

        #        # TODO: Uncomment when there's a point!
        #        if len(blown_idx) > 0:
        #            msg = ["Following peaks inside cluster were too large and had to be removed:"]
        #            msg.extend([str(self.model[i]) for i in blown_idx])
        #            logger.info("\n".join(msg))

        # A peak can only be removed once.
        to_remove = list(set(outside_idx) | set(blown_idx) | set(nan_idx))
        to_remove.sort()

        logger.debug("Clean fits to remove: %s", to_remove)

        for i in reversed(to_remove):
            del self.model[i]
        return len(to_remove)

    def reduce_to(self, x, y):
        """Reduce model(x)>y to model(x)=y if hidden peaks are unlikely.

        This serves as an initial parameter estimate more likely to match
        with peaks on the other side of x.  Peaks with fixed parameters or
        a maximum very close to x may prevent optimal results.

        Parameters
        ----------
        x : array-like
            The position at which to match
        y : array-like
            The height to match.

        Returns
        -------
        ModelEvaluator or None
            Return ModelEvaluator instance if fit changed, otherwise None."""
        # No reduction neccessary
        if self.model.value(x) < y:
            logger.debug("reduce_to: No reduction necessary.")
            return None
        orig_model = self.model.copy()
        self.model.match_at(x, y - self.valuebl(x))
        quality = self.fit()

        # Did reduction help?
        if quality is None:
            logger.debug("reduce_to: Reduction failed, reverting.")
            self.model = orig_model
            return None
        min_npars = min([p.npars for p in self.peak_funcs])
        if quality.growth_justified(self, min_npars):
            logger.debug("reduce_to: Reduction not justified, reverting.")
            self.model = orig_model
            return None

        logger.debug("reduce_to: Reduction successful.")
        return quality

    def value(self, r=None):
        """Return value of baseline+model over cluster.

        Parameters
        ----------
        r : array-like, optional
            value(s) over which to calculate the baseline's value.
            The default is over the entire cluster.

        Returns
        -------
        float
            The value of baseline+model over cluster.
        """
        if len(self.model) == 0:
            return self.valuebl(r)
        else:
            if r is None:
                return self.valuebl(r) + (self.model.value(self.r_data, self.slice)[self.slice])
            else:
                return self.valuebl(r) + (self.model.value(r))

    def valuebl(self, r=None):
        """Return baseline's value over cluster.

        If no baseline exists its value is 0 everywhere.

        Parameters
        ----------
        r - value(s) over which to calculate the baseline's value.
            The default is over the entire cluster.

        Returns
        -------
        float
            The value of baseline's value.
        """
        if self.baseline is None:
            if r is None:
                return np.zeros(self.size)
            else:
                return r * 0.0
        else:
            if r is None:
                return self.baseline.value(self.r_data, self.slice)[self.slice]
            else:
                return self.baseline.value(r)

    def residual(self):
        """Return residual of model over cluster."""
        return self.y_cluster - self.value()

    def quality(self, evaluator=None, **kwds):
        """Return ModelEvaluator instance containing calculated quality of the model.

        ModelEvaluator objects may be compared as though they were numerical
        quantities.  Its raw value is given by the 'stat' member.  For more
        details see ModelEvaluator documentation.

        Parameters
        ----------
        evaluator : ModelEvaluator class or None
            The ModelEvaluator class to use. Default is None.

        Keywords
        --------
        kwds - Keyword arguments passed the the ModelEvaluator's evaluate() method.

        Returns
        -------
        ModelEvaluator instance
            The ModelEvaluator instance with quality calculated
        """
        if evaluator is None:
            evaluator_inst = self.error_method()
        else:
            evaluator_inst = evaluator()
        evaluator_inst.evaluate(self, **kwds)
        return evaluator_inst

    def plottable(self, joined=False):
        """Return sequence suitable for plotting cluster model+baseline with matplotlib.

        Parameters
        ----------
        joined : bool
            Return sum of all peaks if joined is True, or each one individually if False.

        Returns
        -------
        array-like
            A sequence of plottable objects.
        """
        if joined:
            return [self.r_cluster, self.y_cluster, self.r_cluster, self.value()]
        else:
            toreturn = [self.r_cluster, self.y_cluster]
            bl = self.valuebl()
            toreturn.extend([self.r_cluster for i in range(2 * len(self.model))])
            for i, p in enumerate(self.model):
                toreturn[2 * i + 3] = bl + p.value(self.r_data, self.slice)[self.slice]
            return toreturn

    def plottable_residual(self):
        """Return sequence suitable for plotting cluster residual with matplotlib.

        Returns
        -------
        array-like
            A sequence of plottable clusters and residuals.
        """
        return [self.r_cluster, self.residual()]

    def augment(self, source):
        """Add peaks from another ModelCluster that improve this one's quality.

        Parameters
        ----------
        source : ModelCluster instance
            The ModelCluster instance to augment the model's quality.

        Returns
        -------
        None
        """
        best_model = self.model.copy()
        best_qual = self.quality()
        source_model = source.model.copy()

        msg = [
            "==== Augmenting model ====",
            "Original fit:",
            "%s",
            "w/ quality: %s",
            "New model fits:",
            "%s",
        ]
        logger.debug("\n".join(msg), best_model, best_qual.stat, source_model)

        # Each iteration improves best_model by adding the peak from
        # source_model to best_model that most improves its quality, breaking
        # when no further improvement is possible.  This is a quick downhill
        # search, and isn't meant to be exhaustive.
        while len(source_model) > 0:
            test_models = [best_model.copy() for s in source_model]
            for t, s in zip(test_models, source_model):
                t.append(s)
            test_quals = np.array([None for t in test_models])

            for i in range(len(test_models)):
                self.model = test_models[i]
                self.fit()
                test_quals[i] = self.quality()

            args = test_quals.argsort()
            if test_quals[args[-1]] > best_qual:
                best_qual = test_quals[args[-1]]
                best_model = test_models[args[-1]]
                del source_model[args[-1]]
            else:
                break  # Best possible model has been found.
        self.replacepeaks(best_model, slice(len(self.model)))

        # TODO: Do I need this? If test_model contains peaks
        # by reference, the fit peaks will change as well.
        self.fit()
        msg = ["Best model after fit is:", "%s", "w/ quality: %s", "================="]
        logger.debug("\n".join(msg), self.model, best_qual.stat)

        return

    def __str__(self):
        """Return string representation of the cluster."""
        return "\n".join(
            [
                "Slice: %s" % self.slice,
                "Quality: %s" % self.quality().stat,
                "Baseline: %s" % self.baseline,
                "Peaks:\n%s" % self.model,
            ]
        )

    def prune(self):
        """Remove peaks until model quality no longer improves.

        Peaks are removed in a greedy fashion, and the best possible model is
        by no means guaranteed.

        Due to the somewhat exploratory nature of prune many non-convergent
        fits will generally be performed, but it severely restricts the number
        of function evaluations permitted during fitting, and so fits that do
        not converge rapidly are abandoned.  Nevertheless, occasionally this
        method will take an unusually long time to complete.
        """
        if len(self.model) == 0:
            return

        tracer = srmiselog.tracer
        tracer.pushc()

        y_nobl = self.y_cluster - self.valuebl()
        prune_mc = ModelCluster(
            None,
            None,
            self.r_cluster,
            y_nobl,
            self.error_cluster,
            None,
            self.error_method,
            self.peak_funcs,
        )
        orig_model = self.model.copy()

        peak_range = 3  # number of peaks on either side of deleted peak to fit
        check_models = []
        for m in orig_model:
            if m.removable:
                check_models.append(orig_model)
            else:
                check_models.append(None)

        best_model = self.model.copy()
        best_qual = self.quality()

        msg = ["====Pruning fits:====", "Original model:", "%s", "w/ quality: %s"]
        logger.info("\n".join(msg), best_model, best_qual.stat)

        # Main prune loop ####
        while check_models.count(None) < len(check_models):

            # Cache value of individual peaks for best current model.
            best_modely = []
            for b in best_model:
                best_modely.append(b.value(self.r_cluster))

            check_qual = np.array([])
            check_qualidx = np.array([], dtype=np.int32)

            check_models = []
            for m in best_model:
                if m.removable:
                    check_models.append(best_model)
                else:
                    check_models.append(None)

            # Check the ith model
            # Use orig_model for initial parameters of each fit, but best_model for the parameters
            # of those peaks not being fit.  This means we are always fitting to the best y data
            # yet found, but that how this fit is achieved is less biased by the order in which
            # peaks are removed.
            for i in range(len(check_models)):
                if check_models[i] is not None:
                    # Create model with ith peak removed, and distant peaks effectively fixed
                    lo = max(i - peak_range, 0)
                    hi = min(i + peak_range + 1, len(best_model))
                    check_models[i] = best_model[lo:i].copy()
                    check_models[i].extend(best_model[i + 1 : hi].copy())
                    prune_mc.model = check_models[i]

                    msg = ["len(check_models): %s", "len(best_model): %s", "i: %s"]
                    logger.debug("\n".join(msg), len(check_models), len(best_model), i)

                    addpars = best_model.npars() - check_models[i].npars() - best_model[i].npars(count_fixed=False)

                    # Remove contribution of (effectively) fixed peaks
                    y = np.array(y_nobl)
                    if lo > 0:
                        logger.debug("len(sum): %s", len(np.sum(best_modely[:lo], axis=0)))
                        y -= np.sum(best_modely[:lo], axis=0)
                    if hi < len(best_modely):
                        y -= np.sum(best_modely[hi:], axis=0)
                    prune_mc.y_data = y
                    prune_mc.y_cluster = y

                    msg = [
                        "",
                        "--- %s ---",
                        "Removed peak: %s",
                        "Starting model:",
                        "%s",
                    ]
                    logger.debug("\n".join(msg), i, best_model[i], prune_mc.model)

                    prune_mc.fit(ntrials=int(np.sqrt(len(y)) + 50), estimate=False)

                    qual = prune_mc.quality(kshift=addpars)
                    check_qual = np.append(check_qual, qual)
                    check_qualidx = np.append(check_qualidx, i)

                    msg = ["Found model:", "%s", "addpars: %s", "qual: %s"]
                    logger.debug("\n".join(msg), prune_mc.model, addpars, qual.stat)

                    # Do not check this peak in the future if quality decreased.
                    if qual < best_qual:
                        check_models[i] = None

            arg = check_qual.argsort()

            msg = [
                " - Finished round of pruning -",
                "best_qual: %s",
                "check_qual: %s",
                "sorted check_qual: %s",
            ]
            logger.debug("\n".join(msg), best_qual.stat, [c.stat for c in check_qual], arg)

            arg = arg[-1]
            newbest_qual = check_qual[arg]
            newbest_qualidx = check_qualidx[arg]
            if newbest_qual > best_qual:
                lo = max(newbest_qualidx - peak_range, 0)
                hi = min(newbest_qualidx + peak_range + 1, len(orig_model))
                bmtemp = best_model[:lo]
                bmtemp.extend(check_models[newbest_qualidx])
                bmtemp.extend(best_model[hi:])
                prune_mc.model = bmtemp
                prune_mc.y_data = y_nobl
                prune_mc.y_cluster = y_nobl

                logger.debug("New best model (before final fit):\n%s", prune_mc.model)

                prune_mc.fit(estimate=False)
                best_qual = prune_mc.quality()
                best_model = prune_mc.model

                self.model = best_model
                tracer.emit(self)

                msg = ["New best model:", "%s", "best_qual: %s"]
                logger.debug("\n".join(msg), best_model, best_qual.stat)

                if len(best_model) > 0:
                    del check_models[newbest_qualidx]
                    del orig_model[newbest_qualidx]
                else:
                    # Handles the case where all remaining peaks were removed by cleanfit()
                    # since updating check_models will not cause the loop to terminate in
                    # that case.
                    break
            else:
                break

        msg = [
            "Best model after pruning is:",
            "%s",
            "w/ quality: %s",
            "=================",
        ]
        logger.info("\n".join(msg), self.model, self.quality().stat)

        tracer.popc()

        return


# simple test code
if __name__ == "__main__":
    from numpy.random import randn

    from diffpy.srmise.modelevaluators.aicc import AICc
    from diffpy.srmise.peaks.gaussianoverr import GaussianOverR

    pf = GaussianOverR(0.7)
    res = 0.01

    pars = [[3, 0.2, 10], [3.5, 0.2, 10]]
    ideal_peaks = Peaks([pf.actualize(p, "pwa") for p in pars])

    r = np.arange(2, 4, res)
    y = ideal_peaks.value(r) + randn(len(r))

    err = np.ones(len(r))
    evaluator = AICc()

    guesspars = [[2.9, 0.15, 5], [3.6, 0.3, 5]]
    guess_peaks = Peaks([pf.actualize(p, "pwa") for p in guesspars])
    cluster = ModelCluster(guess_peaks, None, r, y, err, None, AICc, [pf])

    print("--- Actual Peak parameters ---")
    print(ideal_peaks)

    print("\n--- Before fit ---")
    print(cluster)

    cluster.fit()

    print("\n--- After fit ---")
    print(cluster)
