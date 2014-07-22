#!/usr/bin/env python

import numpy as np

from diffpy.srmise.mise import PDFPeakExtraction, MultimodelSelection
import diffpy.srmise.mise.miselog as ml

def run(plot=True):

    # Suppress mundane output
    ml.setlevel("warning")

    # Initialize extraction object from existing SrMise trial.
    ppe = PDFPeakExtraction()
    ppe.read("C60.srmise")
    
    # Prep for multimodel selection using this extraction object.
    ms = MultimodelSelection()
    ms.setppe(ppe)
    
    # Define the values of dg to assume in order to generate multiple models.
    # For the purpose of illustration we'll use 10 evenly-spaced values of dg
    # where 1% < dg < 5% of max gr value.
    rmin = 1.
    rmax = 7.25
    r = np.array(ppe.x)
    gr = np.array(ppe.y)
    mask = np.where((r>=rmin) & (r<=rmax))
    r = r[mask]
    gr = gr[mask]
    grmax = np.max(gr)
    print "Maximum height of G(r) within %s <= r <= %s angstroms" %(rmin, rmax)
    print "G(r): %s at r=%s" %(grmax, r[np.argmax(gr)])
    dgs = np.linspace(.01*grmax, .05*grmax, 10)

    # Extraction keywords are inherited from C60.srmise, but explicitly
    # overwrite whatever extraction range it used.
    kwds={}
    kwds["rng"] = [rmin, rmax]
    ppe.setvars(**kwds)

    # Perform peak extraction for each of the assumed uncertainties.
    ms.run(dgs)
    
    # Evaluate and save AIC for all models using Nyquist sampling.
    # The file "ps.dat" saves the results of extraction, the
    # file "ms.dat" saves the AIC values of each model when evaluated
    # over each of the uncertainties assumed when creating models.
    dr = np.pi/ppe.qmax
    ms.save("C60ps.dat")
    ms.makeaics(dgs, dr, filename="C60ms.dat")

if __name__ == '__main__':
    run()
