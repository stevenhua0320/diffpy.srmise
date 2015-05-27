#!/usr/bin/env python
##############################################################################
#
# diffpy.srmise     by Luke Granlund
#                   (c) 2015 trustees of the Michigan State University.
#                   All rights reserved.
#
# File coded by:    Luke Granlund
#
# See LICENSE.txt for license information.
#
##############################################################################
"""Example of peak extraction demonstrating non-default values for many extraction parameters.

This example shows how to extract peaks from a crystalline PDF with unreliable
uncertainties, and shows how various extraction variables may be set.  In
particular, choosing a peak function and defining baseline parameters explicitly.

This script is equivalent to running
 srmise data/TiO2_fine_qmax26.gr --range 1.5 10. --dg 0.35 --qmax 26 --resolution .05 --bpoly1=-0.65 0c --save output/TiO2_parameterdetail.srmise --pwa output/TiO2_parameterdetail.pwa --plot
at the command line.
"""

import matplotlib.pyplot as plt

from diffpy.srmise import PDFPeakExtraction
from diffpy.srmise.baselines import Polynomial
from diffpy.srmise.peaks import GaussianOverR

def run(plot=True):
    
    ## Initialize peak extraction
    # Create peak extraction object
    ppe = PDFPeakExtraction()
    
    # Load the PDF from a file
    ppe.loadpdf("data/TiO2_fine_qmax26.gr")

    ## Set up extraction parameters.
    # In this section we'll examine the major extraction parameters in detail.
    # diffpy.srmise strives to provide reasonable default values for these
    # parameters.  For normal use setting the range, baseline, and uncertainty
    # should be sufficient.
    kwds = {}     
    
    # Range defaults to the entire PDF if not specified.
    kwds["rng"] = [1.5, 10.]
    
    # diffpy.srmise selects model complexity based primarily on the uncertainty
    # of the PDF.  Note that very small uncertainties (<1%) can make peak
    # extraction excessively slow.  In general, the smaller the uncertainty the
    # more complex the model.  PDFs which report no uncertainty, or report
    # unreliable values must be assigned one.  By default, a PDF which does not
    # report uncertainties uses 5% the maximum minus minimum values. Common
    # causes of unreliable uncertainties include oversampling (uncertainties in
    # nearby data are strongly correlated, as for this PDF) and/or
    # integrated diffraction patterns obtained by a method that also introduces
    # correlation to the 1D diffraction pattern.  Consequently, the assumption
    # of both least-squares fitting and the Akaike Information Criterion that
    # the data are at least approximately independently distributed is not
    # valid.  In this case results obtained by diffpy.srmise may be useful,
    # especially when they can be intrepreted in light of prior knowledge, but
    # strong statistical conclusions cannot be drawn.  For additional
    # discussion of this subtle yet important issue see:
    # [1] Egami and Billinge. (2012). Underneath the Bragg Peaks: Structural
    #     Analysis of Complex Materials (2nd ed.). Oxford: Pergamon Press.
    # [2] Granlund, et al. (2015) Acta Crystallographica A, 71(4), ?-?.
    #     doi:10.1107/S2053273315005276
    # [3] Yang, et al. (2014). Journal of Applied Crystallography, 47(4),
    #     1273-1283. doi:10.1107/S1600576714010516
    kwds["dg"] = 0.35  # Play with this value!

    # As a crystal PDF, a linear baseline crossing the origin is appropriate.
    # Here we define the linear baseline B(r) = -.5*r + 0, and explicitly set
    # the y-intercept as a fixed parameter which will not be fit.  For
    # crystal PDFs the theoretical value of the slope is -4*pi*rho0, where
    # rho0 is the number density.  Nevertheless, imperfect normalization of the
    # PDF means the experimental baseline is proportional to that value.
    blfunc = Polynomial(degree=1)
    slope = -.65 # Play with this value!
    y_intercept = 0.
    kwds["baseline"] = blfunc.actualize([slope, y_intercept], free=[True, False])

    # The pf (peakfunction) parameter allows setting the shape of peaks to be
    # extracted.  Termination effects are added automatically to the peak
    # function during extraction.  In the harmonic approximation of atomic
    # interactions peaks in the PDF are well approximated by a Gaussian/r.
    # (Note, however, that the values used for peak parameters -- namely
    # position, width, and area -- are for the Gaussian itself).  diffpy.srmise
    # uses width-limited peaks to reduce the likelihood of extracting
    # unphysically wide peaks in regions of high overlap.  The parameter
    # indicates the max fwhm permitted.  By default, diffpy.srmise uses a
    # maximum width of 0.7, which is generally reasonable if the r-axis of the
    # PDF is given in angstroms.  Models where many peaks reach the maximum
    # width, and models that are very sensitive to the choice in maximum width,
    # are strong signs that diffpy.srmise is having difficulty finding peaks
    # which are sufficiently constrained by the data.
    kwds["pf"] = [GaussianOverR(0.7)]

    # PDFs typically report the value of qmax (i.e. the maximum momentum
    # transfer q in the measurement), but it can be specified explicitly also.
    # If the PDF does not report qmax, diffpy.srmise attempts to estimate it
    # directly from the data.  This estimate can also be used by setting qmax
    # to "automatic".  An infinite qmax can be specified by setting qmax to 0,
    # In that case the Nyquist rate is 0 (infinite resolution), and diffpy.srmise
    # does not consider Nyquist sampling or termination effects.
    kwds["qmax"] = 26.0 

    # This parameter governs whether diffpy.srmise attempts to find a model
    # on a Nyquist-sampled grid with dr=pi/qmax, which is a grid where data
    # uncertainties are least correlated without loss of information.  By
    # default this parameter is True whenever qmax > 0, and generally it
    # should not need to be changed.  Setting it to False allows extracted
    # models retain more complexity because the data appear to have more
    # statistically independent points than they truly do.  For a detailed
    # discussion of Nyquist sampling and the PDF see:
    # [4] Farrow et al. (2011). Physical Review B, 84(13), 134105.
    #     doi:10.1103/PhysRevB.84.134105
    kwds["nyquist"] = True
    
    # This parameter dictates the data be oversampled by at least this factor
    # (relative to the Nyquist rate) during the early stages of peak
    # extraction. If the input PDF is even more finely sampled, that level of
    # sampling is used instead.  The default value of 4.0 is ad hoc, but has
    # been empirically sufficient.  Increasing this value may help the peak-
    # finding and clustering process, but reduces speed.
    kwds["supersample"] = 4.0
    
    # The cres (clustering resolution) parameter governs the sensitivity of the
    # clustering method used by diffpy.srmise.  In short, when the data are
    # being clustered, data which are further than the clustering resolution
    # from any other cluster (measured along the r-axis) are considered to be a
    # new cluster rather than a member of an existing one.  The default value
    # is the Nyquist sampling interval pi/qmax, and on most data it should not
    # greatly impact model complexity.  In some cases making it smaller may
    # help the peak-finding process.  Here it is roughly half the Nyquist
    # interval.
    kwds["cres"] = 0.05
    
    # Apply peak extraction parameters.
    ppe.setvars(**kwds)

    ## Perform peak extraction
    ppe.extract()
    
    ## Save output
    # The write() method saves a file which preserves all aspects of peak
    # extraction and its results, by convention using the .srmise extension,
    # and which can later be read by diffpy.srmise.
    #
    # The writepwa() method saves a file intended as a human-readable summary.
    # In particular, it reports the position, width (as full-width
    # half-maximum), and area of of extracted peaks.  The reported values
    # are for Gaussians in the radial distribution function (RDF) corresponding
    # to this PDF.
    ppe.write("output/TiO2_parameterdetail.srmise")
    ppe.writepwa("output/TiO2_parameterdetail.pwa")

    ## Plot results.
    # Display plot of extracted peak.  It is also possible to plot an existing
    # .srmise file from the command line using
    #     srmise output/TiO2_parameterdetail.srmise --no-extract --plot
    # or, for a somewhat prettier plot,
    #     srmiseplot output/TiO2_parameterdetail.srmise --show
    if plot:
        ppe.plot()
        plt.show()

if __name__ == '__main__':
    run()
