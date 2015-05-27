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
"""Example of extracting an isolated peak from a crystalline PDF.

This example shows how to extract an isolated peak from a simple crystalline
PDF with accurate experimentally-determined uncertainties using a crystalline
baseline estimated from the data.  This is the simplest use case
for diffpy.srmise, and covers initializing diffpy.srmise, defining extraction
parameters, running peak extraction, and saving the results.

This script is equivalent to running
 srmise data/Ag_nyquist_qmax30.gr --range 2. 3.5 --baseline=Polynomial(degree=1) --save output/Ag_singlepeak.srmise --pwa output/Ag_singlepeak.pwa --plot
at the command line.
"""

import matplotlib.pyplot as plt

from diffpy.srmise import PDFPeakExtraction
from diffpy.srmise.baselines import Polynomial

def run(plot=True):
    
    ## Initialize peak extraction
    # Create peak extraction object
    ppe = PDFPeakExtraction()
    
    # Load the PDF from a file
    ppe.loadpdf("data/Ag_nyquist_qmax30.gr")

    ## Set up extraction parameters.
    # For convenience we add all parameters to a dictionary before passing them
    # to the extraction object.
    #
    # The "rng" (range) parameter defines the region over which peaks will be
    # extracted and fit.  For the well isolated nearest-neighbor silver peak,
    # which occurs near 2.9 angstroms, it is sufficient to perform extraction
    # between 2 and 3.5 angstroms.
    #
    # The "baseline" parameter lets us define the PDF baseline, which is
    # linear for a crystal.  If a linear baseline is specified without
    # numerical parameters diffpy.srmise attempts to estimate them from the
    # data, and this is usually sufficient when peaks do not overlap much.
    kwds = {} 
    kwds["rng"] = [2.0, 3.5]
    kwds["baseline"] = Polynomial(degree=1)
    
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
    # In particular, it reports the position, width (as full-width at
    # half-maximum), and area of of extracted peaks.  The reported values
    # are for Gaussians in the radial distribution function (RDF) corresponding
    # to this PDF.
    ppe.write("output/Ag_singlepeak.srmise")
    ppe.writepwa("output/Ag_singlepeak.pwa")

    ## Plot results.
    # Display plot of extracted peak.  It is also possible to plot an existing
    # .srmise file from the command line using
    #     srmise output/Ag_singlepeak.srmise --no-extract --plot
    # or, for a somewhat prettier plot,
    #     srmiseplot output/Ag_singlepeak.srmise --show
    if plot:
        ppe.plot()
        plt.show()

if __name__ == '__main__':
    run()
