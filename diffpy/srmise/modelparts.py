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
"""Module for representing instances of mathematical functions.

Classes
-------
ModelPart: Superclass of Peak and Baseline
ModelParts: Collection (list) of ModelPart instances.
"""

import numpy as np
from scipy.optimize import leastsq
from diffpy.srmise.srmiseerrors import *
from diffpy.srmise import srmiselog

import logging
logger = logging.getLogger("diffpy.srmise")

import matplotlib.pyplot as plt

# Output of scipy.optimize.leastsq for a single parameter changed in scipy 0.8.0
# Before it returned a scalar, later it returned an array of length 1.
import pkg_resources as pr
__spv__ = pr.get_distribution('scipy').version
__oldleastsqbehavior__ = (pr.parse_version(__spv__) < pr.parse_version('0.8.0'))


class ModelParts(list):
    """A collection of ModelPart instances.

    Methods
    -------
    copy: Return deep copy
    fit: Fit to given data
    npars: Return total number of parameters
    pack_freepars: Update free parameters with values in given sequence
    residual: Return residual of model
    residual_jacobian: Return jacobian of residual of model
    transform: Change format of parameters.
    value: Return value of model
    unpack_freepars: Return sequence containing value of all free parameters
    """

    def __init__(self, *args, **kwds):
        list.__init__(self, *args, **kwds)

    def fit(self, r, y, y_error, range=None, ntrials=0, cov=None, cov_format="default_output"):
        """Chi-square fit of all free parameters to given data.

        There must be at least as many free parameters as data points.
        Fitting is performed with the MINPACK leastsq() routine exposed by scipy.

        Parameters
        r - Sequence of r values over which to fit
        y - Sequence of y values over which to fit
        y_error - Sequence of uncertainties in y
        range - Slice object specifying region of r and y over which to fit.
                Fits over all the data by default.
        ntrials - The maximum number of function evaluations while fitting.
        cov - Optional ModelCovariance object preserves covariance information.
        cov_format - Parameterization to use in cov.
        """
        freepars = self.unpack_freepars()
        if len(freepars) >= len(r):
            emsg = "Cannot fit model with " + str(len(freepars)) +\
                   " free parametersbut only "+str(len(r)) + " data points."
            raise SrMiseFitError(emsg)
        if len(freepars) == 0:
            #emsg = "Cannot fit model with no free parameters."
            #raise SrMiseFitError(emsg)
            return

        if range == None:
            range = slice(None)

        args = (r, y, y_error, range)

        if srmiselog.liveplots:
            plt.figure(1)
            plt.ioff()
            plt.subplot(211)
            plt.cla()
            plt.title("Before")
            plt.plot(r, y, label="_nolabel_")
            plt.plot(r, (y-self.value(r, range=range))-1.1*(max(y) - min(y)), label="_nolabel_")
            for p in self:
                plt.plot(r, p.value(r, range=range), label=str(p))
            plt.ion()

        try:
            f = leastsq(
                self.residual,               # minimize this function
                freepars,                    # initial parameters
                args=args,                   # arguments to residual, residual_jacobian
                Dfun=self.residual_jacobian, # explicit Jacobian
                col_deriv=1,                 # order of derivatives in Jacobian
                full_output=1,
                maxfev=ntrials)
        except NotImplementedError:
            # TODO: Figure out if is worth checking for residual_jacobian
            # before leastsq().  This exception will either occur almost never
            # or extremely frequently, and the extra evaluations will add up.
            logger.info("One or more functions do not define residual_jacobian().")
            f = leastsq(
                self.residual, # minimize this function
                freepars,      # initial parameters
                args=args,     # arguments to residual
                col_deriv=1,   # order of derivatives in Jacobian
                full_output=1,
                maxfev=ntrials)
        except Exception:
            # Sadly, KeyboardInterrupt, etc. is reraised as minpack.error
            # Not much I can do about that, though.
            import traceback
            emsg = "Unexpected error in modelparts.fit().  Original exception:\n" +\
                   traceback.format_exc() + "End original exception."
            raise SrMiseFitError(emsg)

        result = f[0]
        if __oldleastsqbehavior__ and len(freepars) == 1:
            # leastsq returns a scalar when there is only one parameter
            result = np.array([result])

        self.pack_freepars(result)

        if srmiselog.liveplots:
            plt.draw()
            plt.ioff()
            plt.figure(1)
            plt.subplot(212)
            plt.cla()
            plt.title("After")
            plt.ion()
            plt.plot(r, y,
                     r, (y-self.value(r, range=range))-1.1*(max(y) - min(y)),
                     *[i for sublist in [[r, p.value(r, range=range)] for p in self] for i in sublist])
            plt.draw()

            if srmiselog.wait:
                print "Press 'Enter' to continue...",
                raw_input()

        if f[4] not in (1,2,3,4):
            emsg = "Fit did not succeed -- " + str(f[3])
            raise SrMiseFitError(emsg)

        # clean up parameters
        for p in self:
                p.pars = p.owner().transform_parameters(p.pars, in_format="internal", out_format="internal")

        # Supply estimated covariance matrix if requested.
        # The precise relationship between f[1] and estimated covariance matrix is a little unclear from
        # the documentation of leastsq.  This is the interpretation given by scipy.optimize.curve_fit,
        # which is a wrapper around leastsq.
        if cov is not None:
            pcov = f[1]
            fvec = f[2]["fvec"]
            dof = len(r) - len(freepars)
            cov.setcovariance(self, pcov*np.sum(fvec**2)/dof)
            try:
                cov.transform(in_format="internal", out_format=cov_format)
            except SrMiseUndefinedCovarianceError as e:
                logger.warn("Covariance not defined.  Fit may not have converged.")


        return
#### Notes on the fit f
# f[0] = solution
# f[1] = Uses the fjac and ipvt optional outputs to construct an estimate of the jacobian around the solution.
#        None if a singular matrix encountered (indicates very flat curvature in some direction).
#        This matrix must be multiplied by the residual variance to get the covariance of the parameter
#        estimates - see curve fit.
# f[2] = dictionary{nfev: int, fvec: array(), fjac: array(), ipvt: array(), qtf: array()}
#        nfev - The number of function calls made
#        fvec - function (residual) evaluated at solution
#        fjac - "a permutation of the R matrix of a QR factorization of the final Jacobian."
#        ipvt - integer array defining a permutation matrix P such that fjac*P=QR
#        qtf - transpose(q)*fvec
# f[3] = message about results of fit
# f[4] = integer flag.  Fit was successful on 1,2,3, or 4.  Otherwise unsuccessful.

    def npars(self, count_fixed=True):
        """Return total number of parameters in all parts.

        Parameters
        count_fixed - Boolean which determines if fixed parameters are
                      are included in the count.
        """
        n = 0
        for p in self:
            n+=p.npars(count_fixed=count_fixed)
        return n

    def pack_freepars(self, freepars):
        """Update parameters with values from sequence of freepars."""
        if np.isnan(freepars).any():
            emsg = "Non-numeric free parameters."
            raise ValueError(emsg)
        freeidx = 0
        for p in self:
            freeidx += p.update(freepars[freeidx:])

    def residual(self, freepars, r, y_expected, y_error, range=None):
        """Calculate residual of all parameters.

        Parameters
        freepars - sequence of free parameters
        r - the input domain
        y_expected - sequence of expected values
        y_error - sequence of uncertainties in y-variable
        range - Slice object specifying region of r and y over which to fit.
                All the data by default.
        """
        self.pack_freepars(freepars)
        total = self.value(r, range)
        try:
            if range is None:
                range = slice(0, len(r))
            return (y_expected[range]-total[range])/y_error[range]
        except TypeError:
            return (y_expected-total)/y_error

    def residual_jacobian(self, freepars, r, y_expected, y_error, range=None):
        """Calculate the Jacobian of freepars.

        Parameters
        freepars - sequence of free parameters
        r - the input domain
        y_expected - sequence of expected values
        y_error - sequence of uncertainties in y-variable
        range - Slice object specifying region of r and y over which to fit.
                All the data by default.
        """
        if len(freepars) == 0:
            raise ValueError("Argument freepars has length 0.  The Jacobian "
                             "is only defined with >=1 free parameters.")

        self.pack_freepars(freepars)
        tempJac=[]
        for p in self:
            tempJac[len(tempJac):] = p.jacobian(r, range)
        # Since the residual is (expected - calculated) the jacobian
        # of the residual has a minus sign.
        jac=-np.array([j for j in tempJac if j is not None])
        try:
            if range is None:
                range = slice(0, len(r))
            return jac[:,range]/y_error[range]
        except TypeError:
            return jac/y_error

    def value(self, r, range=None):
        """Calculate total value of all parts over range.

        Parameters
        r - the input domain
        range - Slice object specifying region of r and y over which to fit.
                All the data by default.
        """
        total = r * 0.
        for p in self:
            total += p.value(r, range)
        return total

    def unpack_freepars(self):
        """Return array of all free parameters."""
        #To check: ravel() sometimes returns a reference and othertimes a copy.
        #          Do I need to use flatten() instead?
        return np.concatenate([p.compress() for p in self]).ravel()

    def covariance(self, format="internal", **kwds):
        """Return estimated covariance matrix of the model.

        The covariance matrix may be given in terms of any parameterization
        defined by the formats for each individual ModelPart.

        Parameters
        format - The format ("internal" by default) to use for all ModelParts.
                 This may be overridden for specific peaks as shown below.

        Keywords
        f0 - The format of the 0th ModelPart
        f1 - The format of the 1st ModelPart
        etc.
        """
        formats = [format for p in self]

        for k, v in kwds.items():
            try:
                idx = int(k[1:])
            except ValueError:
                emsg = "Invalid format keyword '%s'.  They must be specified as 'f0', 'f1', etc." %k
                raise ValueError(emsg)

            formats[int(k[1:])] = v



        return


    def copy(self):
        """Return deep copy of this ModelParts.

        The original and the copy are completely independent, except each
        ModelPart and its copy still reference the same owner."""
        return type(self).__call__([p.copy() for p in self])

    def __str__(self):
        """Return string representation of this ModelParts."""
        return ''.join([str(p)+"\n" for p in self])

    def __getslice__(self, i, j):
        """Extends list.__getslice__"""
        return self.__class__(list.__getslice__(self, i, j))

    def transform(self, in_format="internal", out_format="internal"):
        """Transforms format of parameters in this modelpart.

        Parameters
        in_format - The format the parameters are already in.
        out_format - The format the parameters are transformed to.
        """
        for p in self:
            try:
                p.pars = p.owner().transform_parameters(p.pars, in_format, out_format)
            except ValueError:
                logger.info("Invalid parameter transformation: Ignoring %s->%s for function of type %s." %(in_format, out_format, p.owner().getmodule()))

# End of class ModelParts

class ModelPart(object):
    """Represents a single part (instance of some function) of a model.

    Members
    -------
    pars - Array containing the parameters of this model part
    free - Array containing boolean values defining whether the corresponding parameter
           is free or not.
    removable - Boolean determining whether or not this model part can be
                removed during extraction.
    static_owner - Boolean determines if owner can be changed with changeowner()

    Methods
    -------
    changeowner - Change the owner of self
    copy - Return deep copy of self
    compress - Return parameters with non-free parameters removed
    jacobian - Return jacobian
    getfree - Return free parameter by index or keyword define by owner
    npars - Return number of parameters in self
    owner - Return self.owner
    setfree - Set a free parameter by index or keyword defined by owner
    update - Update free parameters with values in given sequence
    value - Return value
    writestr - Return string representation of self
    """

    def __init__(self, owner, pars, free=None, removable=True, static_owner=False):
        """Set instance members.

        Parameters
        owner - an instance of a BaseFunction subclass
        pars - Sequence of parameters which specify the function explicitly
        free - Sequence of Boolean variables.  If False, the corresponding
               parameter will not be changed.
        removable - Boolean determines whether this part can be removed.
        static_owner - Whether or not the part can be changed with
                       changeowner()

        Note that free and removable are not mutually exclusive.  If any
        pars are not free but removable=True then the part may be removed, but
        the held parameters for this part will remain unchanged until then.
        """
        self._owner = owner

        if len(pars) != owner.npars:
            emsg = "The length of pars must equal the number of parameters "+\
                   "specified by the model part owner."
            raise ValueError(emsg)
        self.pars = np.array(pars[:]) # pars[:] in case pars is a ModelPart

        if free is None:
            self.free = np.array([True for p in pars], dtype=bool)
        else:
            self.free = np.array(free, dtype=bool)
        if len(self.free) != owner.npars:
            emsg = "The length of free must be equal to the number of "+\
                   "parameters specified by the model part owner."
            raise ValueError(emsg)

        self.removable = removable
        self.static_owner = static_owner

    def changeowner(self, owner):
        """Change the owner of this part.

        Does not change the parameters associated with this model part. Raises
        SrMiseStaticOwnerError if this peak has been declared to have a static
        owner, or if the number of parameters is incompatible.

        Parameters
        owner - an instance of a BaseFunction subclass
        """
        if self.static_owner and self._owner is not owner:
            emsg = "Cannot change owner if static_owner is True."
            raise SrMiseStaticOwnerError(emsg)
        if self._owner.npars != owner.npars:
            emsg = "New owner specifies different number of parameters than "+\
                   "original owner."
            raise SrMiseStaticOwnerError(emsg)
        self._owner = owner

    def compress(self):
        """Return part parameters with non-free values removed."""
        return self.pars[self.free]

    def jacobian(self, r, range=None):
        """Return jacobian of this part over r.

        Parameters
        r - the input domain
        range - Slice object specifying region of r and y over which to fit.
                All the data by default.
        """
        return self._owner.jacobian(self, r, range)

    def owner(self):
        """Return the BaseFunction subclass instance which owns this part."""
        return self._owner

    def update(self, freepars):
        """Sequentially update free parameters from freepars.

        Parameters
        freepars - sequence of new parameter values.  May contain more
                   parameters than can actually be updated.

        Return number of parameters updated from freepars.
        """
        numfree = self.npars(count_fixed=False)
        if len(freepars) < numfree:
            pass # raise "freepars does not have enough elements to
                 # update every unheld parameter."
        # TODO: Check if I need to make copies here, or if references
        #       to parameters are safe.
        self.pars[self.free] = freepars[:numfree]
        return numfree

    def value(self, r, range=None):
        """Return value of peak over r.

        Parameters
        r - the input domain
        range - Slice object specifying region of r and y over which to fit.
                All the data by default.
        """
        return self._owner.value(self, r, range)

    def copy(self):
        """Return a deep copy of this ModelPart.

        The original and the copy are completely independent, except they both
        reference the same owner."""
        return type(self).__call__(self._owner, self.pars, self.free, self.removable, self.static_owner)

    def __getitem__(self, key_or_idx):
        """Return parameter of peak corresponding with key_or_idx.

        Parameters
        key_or_idx - An integer index, slice, or key from owner's parameter
                     dictionary.
        """
        try:
            return self.pars[key_or_idx]
        except ValueError:
            return self.pars[self._owner.parameterdict[key_or_idx]]

    def getfree(self, key_or_idx):
        """Return value of free corresponding with key_or_idx.

        Parameters
        key_or_idx - An integer index, slice, or key from owner's parameter
                     dictionary."""
        try:
            return self.free[key_or_idx]
        except ValueError:
            return self.free[self._owner.parameterdict[key_or_idx]]

    def setfree(self, key_or_idx, value):
        """Set value of free corresponding with key_or_idx.

        Parameters
        key_or_idx - An integer index, slice, or key from owner's parameter
                     dictionary.
        value: A boolean"""
        try:
            self.free[key_or_idx] = value
        except ValueError:
            self.free[self._owner.parameterdict[key_or_idx]] = value

    def __len__(self):
        """Return number of parameters, including any fixed ones."""
        return self._owner.npars

    def npars(self, count_fixed=True):
        """Return total number of parameters in all parts.

        Parameters
        count_fixed - Boolean which determines if fixed parameters are
                      are included in the count."""
        if count_fixed:
            return self._owner.npars
        else:
            return (self.free == True).sum()

    def __str__(self):
        """Return string representation of ModelPart parameters."""
        return str(self._owner.transform_parameters(self.pars, in_format="internal", out_format="default_output"))

    def __eq__(self, other):
        """   """
        if hasattr(other, "_owner"):
            return ((self._owner is other._owner) and
                    np.all(self.pars == other.pars) and
                    np.all(self.free == other.free) and
                    self.removable == other.removable)
        else:
            return False

    def __ne__(self, other):
        """   """
        return not self == other

    def writestr(self, ownerlist):
        """Return string representation of ModelPart.

        The value of owner is determined by its index in ownerlist.

        Parameters
        ownerlist - List of owner functions
        """
        if self._owner not in ownerlist:
            emsg = "ownerlist does not contain this ModelPart's owner."
            raise ValueError(emsg)
        lines = []
        lines.append("owner=%s" %repr(ownerlist.index(self._owner)))

        #Lists/numpy arrays don't give full representation of long lists
        lines.append("pars=[%s]" %", ".join([repr(p) for p in self.pars]))
        lines.append("free=[%s]" %", ".join([repr(f) for f in self.free]))
        lines.append("removable=%s" %repr(self.removable))
        lines.append("static_owner=%s" %repr(self.static_owner))
        datastring = "\n".join(lines)+"\n"
        return datastring

# End of class ModelPart

# simple test code
if __name__ == '__main__':

    pass
