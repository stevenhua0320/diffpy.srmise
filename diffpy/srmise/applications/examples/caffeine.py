#!/usr/bin/env python
# This script is equivalent to:
# srmise caffeine.gr --range 1. 5. --baseline Polynomial(degree=1) --dg-mode absolute --dg .2 --save caffeine.srmise -p

import numpy as np
import matplotlib.pyplot as plt

from diffpy.srmise.mise import PDFPeakExtraction
from diffpy.srmise.mise.baselines import Polynomial
import diffpy.srmise.mise.miselog as ml

def run(plot=True):
    ml.setlevel("info")

    ppe = PDFPeakExtraction()
    ppe.loadpdf("caffeine.gr")

    kwds={}
    kwds["rng"] = [1., 5.]
    kwds["baseline"] = Polynomial(degree=1)
    kwds["dg"] = 0.2

    ppe.setvars(**kwds)

    ppe.extract()
    ppe.write("caffeine.srmise")

    if plot:
        ppe.plot()
        plt.show()

if __name__ == '__main__':
    run()
