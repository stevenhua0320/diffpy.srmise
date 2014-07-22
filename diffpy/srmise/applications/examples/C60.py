#!/usr/bin/env python
# This script is equivalent to:
# srmise C60.gr --range 1. 8. --bseq C60baseline.dat --dg 750 --save C60.srmise -p

import numpy as np
import matplotlib.pyplot as plt

from diffpy.srmise.mise import PDFPeakExtraction
from diffpy.srmise.mise.baselines import FromSequence
import diffpy.srmise.mise.miselog as ml

def run(plot=True):
    ml.setlevel("info")

    ppe = PDFPeakExtraction()
    ppe.loadpdf("C60.gr")
    blf = FromSequence("C60baseline.dat")

    kwds={}
    kwds["rng"] = [1., 8.]
    kwds["baseline"] = blf.actualize([])
    kwds["dg"] = 5000
    kwds["nyquist"] = True
    kwds["cres"] = 0.05

    ppe.setvars(**kwds)

    ppe.extract()
    ppe.write("C60.srmise")

    if plot:
        ppe.plot()
        plt.show()

if __name__ == '__main__':
    run()
