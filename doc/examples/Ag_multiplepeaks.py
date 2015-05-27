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
"""Example of extracting multiple peaks and accessing results programatically.

This example shows how to extract a range of peaks from a simple crystalline
PDF using a crystalline baseline obtained from an existing trial.  It shows
how to access the value and uncertainty of peak parameters, including
transforming between different peak parameterizations.  Finally, it shows how
to evaluate the model on an arbitrary grid.

The peaks extracted by this script are equivalent to those obtained running
 srmise data/Ag_nyquist_qmax30.gr --range 2. 10. --bsrmise=output/Ag_singlepeak.srmise --save output/Ag_multiplepeaks.srmise --pwa output/Ag_multiplepeaks.pwa --plot
at the command line.
"""

import numpy as np
import matplotlib.pyplot as plt

from diffpy.srmise import PDFPeakExtraction
from diffpy.srmise import ModelCovariance

def run(plot=True):
    
    ## Initialize peak extraction
    # Create peak extraction object
    ppe = PDFPeakExtraction()
    
    # Load the PDF from a file
    ppe.loadpdf("data/Ag_nyquist_qmax30.gr")
    
    # Obtain baseline from a saved diffpy.srmise trial.  This is not the
    # initial baseline estimate from the previous example, but the baseline
    # after both it and the extracted peaks have been fit to the data.
    ppebl = PDFPeakExtraction()
    ppebl.read("output/Ag_singlepeak.srmise")
    baseline = ppebl.extracted.baseline

    ## Set up extraction parameters.
    # Peaks are extracted between 2 and 10 angstroms, using the baseline
    # from the isolated peak example.
    kwds = {} 
    kwds["rng"] = [2.0, 10.]
    kwds["baseline"] = baseline
    
    # Apply peak extraction parameters.
    ppe.setvars(**kwds)

    ## Perform peak extraction, and retain object containing a copy of the
    # model and the full covariance matrix.
    cov = ppe.extract()
    
    ## Accessing results of extraction
    #
    # Model parameters are organized using a nested structure, with a list
    # of peaks each of which is a list of parameters, similar to the the
    # following schematic.
    #     Peak
    #         Position
    #         Width
    #         Area
    #     Peak
    #         Position
    #         Width
    #         Area*
    #     ...
    #     Baseline
    #         Slope
    #         Intercept
    # By convention, the baseline is the final "peak."  The ModelCovariance
    # object returned by extract() can return information about any peak by
    # using the appropriate tuple of indices (i,j).  That is, (i,j) denotes
    # the jth parameter of the ith peak.  For example, the starred parameter
    # above is the area (index = 2) of the next nearest neighbor (index = 1)
    # peak. Thus, this peak can be referenced as (1,2).  Several examples are
    # presented below.
    print "\n------ Examples of accessing peak extraction results ------"
    
    # The value and uncertainties of the nearest-neighbor peak parameters.
    # Note that a covariance matrix estimated from a PDF with unreliable
    # or ad hoc PDF uncertainties is likewise unreliable or ad hoc.
    position = "%f +/- %f" %cov.get((0,0))
    width = "%f +/- %f" %cov.get((0,1))
    area = "%f +/- %f" %cov.get((0,2))
    print  "Nearest-neighbor peak: position=%s, width=%s, area=%s" %(position, width, area)
    print "Covariance of width and area for nearest-neighbor peak: ", cov.getcovariance((0,1),(0,2))
    
    # It is also possible to iterate over peaks directly without using indices.
    # For example, to calculate the total peak area:
    total_area = 0
    for peak in cov.model[:-1]: # Exclude last element, which is the baseline.
        total_area += peak["area"]    # The "position" and "width" keywords are also
                                # available for the GaussianOverR peak function.
    print "Total area of extracted peaks: ", total_area
    
    # Baseline parameters.
    print "The linear baseline B(r)=%f*r + %f" % tuple(par for par in cov.model[-1])
    
    # Highly-correlated parameters can indicate difficulties constraining the
    # fit.  This function lists all pairs of parameters with an absolute value
    # of correlation which exceeds a given threshold.
    print "Correlations > 0.8:"
    print "\n".join(str(c) for c in cov.correlationwarning(.8))

    ## Different Parameterizations
    # Peaks and baselines may have equivalent parameterizations that are useful
    # in different situations.  For example, the types defined by the
    # GaussianOverR peak function are:
    #   "internal" - Used in diffpy.srmise calculations, explicitly enforces a
    #                maximum peak width
    #   "pwa" - The position, width (full-width at half-maximum), area.
    #   "mu_sigma_area" - The position, width (the distribution standard
    #                     deviation sigma), area.
    #   "default_output" - Defines default format to use in most user-facing
    #                      scenarios. Maps to the "pwa" parameterization.
    #   "default_input" - Defines default format to use when specifying peak
    #                     parameters.  Maps to the "internal" parameterization.
    # All diffpy.srmise peak and baseline functions are required to have the
    # "internal", "default_output", and "default_input" formats.  In many
    # cases, such as polynomial baselines, all of these are equivalent.
    #
    # Suppose you want to know peak widths in terms of the standard deviation
    # sigma of the Gaussian distribution.  It is then appropriate to convert
    # all peaks to the "mu_sigma_area" format.  Valid options for the "parts"
    # keyword are "peaks", "baseline", or a sequence of indices (e.g. [1,2,3]
    # would transform the second, third, and fourth peaks).  If the keyword
    # is omitted, the transformation is attempted for all parts of the fit.
    cov.transform(in_format="pwa", out_format="mu_sigma_area", parts="peaks")
    print "Width (sigma) of nearest-neighbor peak: %f +/- %f" %cov.get((0,1))


    # A .srmise file does not save the full covariance matrix, so it must be
    # recalculated when loading from these files.  For example, here is the
    # nearest-neighbor peak in the file which we used to define the initial
    # baseline.
    cov2 = ModelCovariance()
    ppebl.extracted.fit(fitbaseline=True, cov=cov2, cov_format="default_output")
    position2 = "%f +/- %f" %cov.get((0,0))
    width2 = "%f +/- %f" %cov.get((0,1))
    area2 = "%f +/- %f" %cov.get((0,2))
    print  "Nearest-neighbor peak from file: position=%s, width=%s, area=%s"  \
           %(position2, width2, area2)


    ## Save output
    ppe.write("output/Ag_multiplepeaks.srmise")
    ppe.writepwa("output/Ag_multiplepeaks.pwa")


    ## Evaluating a model.
    # Although the ModelCovariance object is useful, the model used for fitting
    # can be directly accessed through PDFPeakExtraction as well, albeit
    # without uncertainties.  This is particularly useful when evaluating a
    # model since the parameters stay in the "internal" format used for
    # calculations.  For example, here we plot the data and every second peak
    # on an arbitrary grid.  Unlike with ModelCovariance, the baseline and
    # peaks are kept separate.
    if plot:
        plt.figure()
        grid = np.arange(2, 10, .01)
        baseline = ppe.extracted.baseline
        everysecondpeak = ppe.extracted.model[::2]
        plt.plot(ppe.x, ppe.y, 'o')
        for peak in everysecondpeak:
            plt.plot(grid, baseline.value(grid) + peak.value(grid))
        plt.xlim(2, 10)
        plt.show()

if __name__ == '__main__':
    run()
