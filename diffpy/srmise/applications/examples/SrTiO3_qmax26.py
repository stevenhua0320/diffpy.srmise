#!/usr/bin/env python
# This script is equivalent to:
# srmise SrTiO3_qmax26.gr --range 1.5 10 --bcrystal .0839486  --no-nyquist --dg 0.3 --save SrTiO3_qmax26.srmise -p

import numpy as np
import matplotlib.pyplot as plt

from diffpy.srmise.mise import PDFPeakExtraction
from diffpy.srmise.mise.baselines import Polynomial
from diffpy.srmise.mise.peaks import GaussianOverR
import diffpy.srmise.mise.miselog as ml

def run(plot=True):
    ml.setlevel("info")

    ppe = PDFPeakExtraction()
    ppe.loadpdf("SrTiO3_qmax26.gr")

    blf = Polynomial(degree=1)

    rho0 = .0839486

    kwds={}
    kwds["rng"] = [1.5, 10.]
    kwds["baseline"] = blf.actualize([-4*np.pi*rho0, 0.])
    kwds["nyquist"] = False
    kwds["dg"] = 0.3

    ppe.setvars(**kwds)

    ppe.extract()
    ppe.write("SrTiO3_qmax26.srmise")

    if plot:
        ppe.plot()
        plt.show()

if __name__ == '__main__':
    run()
