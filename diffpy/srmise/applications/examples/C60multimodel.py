#!/usr/bin/env python

import numpy as np
import optparse, sys
import matplotlib.pyplot as plt

from diffpy.srmise.mise import MultimodelSelection
from diffpy.srmise.applications.plot import makeplot
import diffpy.srmise.mise.miselog as ml

defaultstabfile = "C60ps.dat"
defaultmsfile = "C60ms.dat"

deftol = 0.1 # default tolerance when classifying models

ml.setlevel("warning")
ms = MultimodelSelection()

def console(stabfile=defaultstabfile, msfile=defaultmsfile):
    """Return MultimodelSelection object loaded from 'ms.dat'."""
    ms.load(stabfile)
    ms.loadaics(msfile)
    dr = np.pi/ms.ppe.qmax
    (r,y,dr2,dy) = ms.ppe.resampledata(dr)
    ms.classify(r, tolerance=deftol)
    return ms


def main(altargs=None):
    """Run multimodeling with command-line options.

    One can override the default argument list used by the command line
    (sys.argv[1:]) by providing a sequence."""

    # configure options parsing
    parser = optparse.OptionParser("%prog [options].\n\n For convenient access from "
                                   "console get loaded MultimodelSelection object "
                                   "using the module's console() function.")
    parser.add_option("--animate", type="float",
            help="Animate AIC probabilities over at least given number of seconds. (Experimental).")
    parser.add_option("-s", "--save", type="string", default="bm",
            help="Save any pwas and plots to file based on given string plus an "
                 "appropriate file extension. Default 'C60bm' (best model).")
    parser.add_option("--loadaics", type="string", default=defaultmsfile,
            help="Load AICs from specified file")
    parser.add_option("--loadruns", type="string", default=defaultstabfile,
            help="Load extracted models from specified file.")
    parser.add_option("-t", "--tolerance", type="float", default=deftol,
            help="Tolerance for model disimilarity.")
    parser.add_option("-r", "--report", action="store_true",
            help="Generate report about the models.")
    parser.add_option("--plotbest", action="store_true",
            help="Show plots of best models.")
    parser.add_option("--plotprobs", action="store_true",
            help="Show plot of Akaike probabilities by class vs. dG")
    parser.add_option("--savefig", action="store_true",
            help="Save plotted figures.")
    parser.add_option("--savepwa", action="store_true",
            help="Save best models to pwa.")
    parser.allow_interspersed_args = True

    if altargs is None:
        opts, args = parser.parse_args(sys.argv[1:])
    else:
        opts, args = parser.parse_args(altargs)

    # Load all extracted peaks and AIC values from SrMise runs.
    ms.load(opts.loadruns)
    ms.loadaics(opts.loadaics)

    # Ensure Nyquist sampling
    dr = np.pi/ms.ppe.qmax
    (r,y,dr2,dy) = ms.ppe.resampledata(dr)

    # All models are placed into classes.  Models in the same class
    # should be essentially identical (same peak parameters, etc.)
    # up to a small tolerance determined by comparing individual peaks.
    ms.classify(r, tolerance=opts.tolerance)

    # Save pwa files for the best models.
    if opts.savepwa:
        bm = ms.bestmodels() # Get indices for models identified as best.
        for m in bm:
            filename = opts.save+str(m)+".pwa"
            cls = ms.classes_idx[m] # Get index of class to which model belongs.
            bestdgs = ms.modelbestdgs(m) # Get the uncertainties where this model has greatest Akaike probability.
            maxprob = np.max([ms.classprobs[dg][cls] for dg in bestdgs]) # Get the model's greatest Akaike probability.

            msg = ["This is a best model determined by MultiModelSelection",
                  "Model: %i (of %i)",
                  "Class: %i (of %i, tolerance=%f)",
                  "Best model for uncertainties: %f-%f",
                  "Max Akaike probability: %g"]
            msg = "\n".join(msg) %(m, len(ms.classes_idx), cls, len(ms.classes), opts.tolerance, np.min(bestdgs), np.max(bestdgs), maxprob)
            ms.setcurrent(m) # Make this the active model.
            ms.ppe.writepwa(filename, msg)

    if opts.report:

        print "------- Report --------"

        # Key facts
        print "Number of models: ", len(ms.results)
        print "Number of classes (tol=%s): " %opts.tolerance, len(ms.classes)
        print "Range of dgs: %s-%s" %(ms.dgs[0], ms.dgs[-1])
        print "Number of (Nyquist) data points: ", len(r)
        bm = ms.bestmodels()
        print "Best models (class, #peaks):"
        for b in bm:
            print " %i (%i, %i)" %(b, ms.classes_idx[b], len(ms.results[b][1]))

    # Animated 2D plot showing how the Akaike probabilities of models vary as the post-hoc
    # uncertainty dG is changed.
    if opts.animate is not None:
        ms.animate_classprobs(duration=opts.animate)

    # 3D plot of Akaike probabilities: probability vs. dG vs. class.
    if opts.plotprobs:
        figdict = ms.plot3dclassprobs(probfilter=[0.0, 1.], figure=plt.figure())
        plt.tight_layout()

        if opts.savefig:
            plt.savefig(opts.save+"probs.png", format="png", bbox_inches="tight")

    # Plots of all the models identified as best for at least some dG.
    if opts.plotbest:
        for bm in ms.bestmodels():
            plt.figure()
            ms.setcurrent(bm)
            figdict = makeplot(ms)

            if opts.savefig:
                plt.savefig(opts.save+str(bm)+".png", format="png")

    if opts.plotprobs or opts.plotbest or opts.animate:
        plt.show()

if __name__ == '__main__':
    main()
