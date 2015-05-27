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

This example shows how to specify initial peaks in order to guide the results of
peak extraction.
"""

import matplotlib.pyplot as plt

from diffpy.srmise import PDFPeakExtraction
from diffpy.srmise.baselines import Polynomial
from diffpy.srmise.peaks import GaussianOverR, Peaks

def run(plot=True):
    
    
    ## Initialize peak extraction and all parameters from existing trial.
    ppe = PDFPeakExtraction()
    ppe.read("output/TiO2_parameterdetail.srmise")
    
    # Clear the result of the previous fit, and make sure "initial_peaks" is
    # set to the default value, which is empty.
    ppe.clearcalc()
    ppe.defaultvars("initial_peaks")

    ## Add initial peaks
    # There are two basic ways to quickly specify initial peaks.  The first
    # is simply supplying the approximate position of the peak, and letting
    # diffpy.srmise estimate the peak parameters.  The estimate_peak() method
    # performs this task, taking into consideration all initial_peaks which
    # have already been specified, and then adding the resulting peak to
    # initial_peaks.  However, by default these peaks could be removed during
    # pruning.  To prevent that in this case, each peak is marked as non-
    # removable.
    rough_guess = [2.0, 2.8, 3.5, 4.0, 4.5, 5.0, 5.5, 5.8]
    for g in rough_guess:
        ppe.estimate_peak(g)
    for p in ppe.initial_peaks:
        p.removable = False
    if plot:
        plt.figure(1)
        ppe.plot()
        plt.suptitle("1. Rough guesses.")

    # The built-in estimation routine cannot handle every case, but it is also
    # possible to provide explicit parameters for the initial peaks.  In this
    # case peaks are created from the same GaussianOverR used during
    # extraction, but one could use a different peak function from
    # diffpy.srmise.peaks if desired.  The peak parameters are given in terms
    # terms of position, width (fwhm), and area, and it is important to specify
    # that format is being used so they are correctly changed into the
    # internal parameterization.
    explicit_guess = [[6.25, .3, 3], [6.5, .3, 3], [6.85, 0.3, 10], [7.1, 0.3, 10], [7.45, 0.3, 20]]
    peak_function = ppe.pf[0]
    explicit_peaks = Peaks([peak_function.actualize(e, removable=False, in_format="pwa") for e in explicit_guess])
    ppe.addpeaks(explicit_peaks)
    if plot:
        plt.figure(2)
        ppe.plot()
        plt.suptitle("2. Rough and explicit guesses.")

    ## Perform peak extraction
    # The initial peaks defined above don't cover the entire range of the PDF,
    # but can still be found using the standard peak extraction method to fill
    # in any gaps.
    ppe.extract()
    if plot:
        plt.figure(3)
        ppe.plot()
        plt.suptitle("3. Results after peak extraction.")
        
    
    ## Save output
    ppe.write("output/TiO2_initialpeaks.srmise")
    ppe.writepwa("output/TiO2_initialpeaks.pwa")

    if plot:
        plt.show()

if __name__ == '__main__':
    run()
