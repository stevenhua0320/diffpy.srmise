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
"""Example of second step in multimodel approach to peak extraction, which is
an AIC-based analysis of many models.

The multimodel approach generates many models of varying complexity by assuming
a range of experimental uncertainties are physically plausible.  This example
shows how to analyze multiple models obtained from a nanoparticle PDF using a
custom baseline.  The Akaike Information Criterion (AIC) can be used to see
which models are relatively more likely to describe the experimental data.  For
complex PDFs, especially, there are many sets of peaks which are physically
distinct yet appear to fit the experimental data similarly well.  Here we
calculate the Akaike weights, which is a measure of the likelihood that a
given model is the best model (in the sense of Kullback-Leibler information
divergence) relative to all the other ones in the same comparison.

NOTE: The multimodeling API used here is expected to change drastically in a
future version of diffpy.srmise.

For more information on the multimodeling approach taken here see
[1] Granlund, et al. (2015) Acta Crystallographica A, 71(4), ?-?.
    doi:10.1107/S2053273315005276
The standard reference of AIC-based multimodel selection is
[2] Burnham and Anderson. (2002). Model Selection and Multimodel Inference.
    New York, NY: Springer. doi:10.1007/b97636
"""
import numpy as np
import optparse, sys
import matplotlib.pyplot as plt

from diffpy.srmise import MultimodelSelection
from diffpy.srmise.applications.plot import makeplot
import diffpy.srmise.srmiselog as sml

# distances from ideal (unrefined) C60 
dcif = np.array(map(float, str.split('''
    1.44 2.329968944 2.494153163 2.88 3.595985339
    3.704477734 4.132591264 4.520339129 4.659937888
    4.877358006 5.209968944 5.405310018 5.522583786
    5.818426502 6.099937888 6.164518388 6.529777754
    6.686673127 6.745638756 6.989906831 7.136693738
    ''')))

def run(plot=True):

    # Suppress mundane output
    sml.setlevel("warning")

    ## Create multimodeling object and load diffpy.srmise results from file.
    ms = MultimodelSelection()
    ms.load("output/C60_models.dat")
    ms.loadaics("output/C60_aics.dat")

    ## Use Nyquist sampling
    # Standard AIC analysis assumes the data have independent uncertainties.
    # Nyquist sampling minimizes correlations in the PDF, which is the closest
    # approximation to independence possible for the PDF.
    dr = np.pi/ms.ppe.qmax
    (r,y,dr2,dy) = ms.ppe.resampledata(dr)

    ## Classify models
    # All models are placed into classes.  Models in the same class
    # should be essentially identical (same peak parameters, etc.)
    # up to a small tolerance determined by comparing individual peaks. The
    # best model in each class essentially stands in for all the other models
    # in a class in the rest of the analysis.  This step reduces a major source
    # of model redundancy, which otherwise weakens AIC-based analysis.
    tolerance = 0.2 
    ms.classify(r, tolerance)

    ## Find "best" models.
    # In short, models with greatest Akaike probability.  If the
    # experimental uncertainty of the PDF is known, one should evaluate all
    # the models using that uncertainty.  If the reported uncertainty is
    # unreliable (the case for this particular PDF) we can only determine what
    # models are best contingent on whether or not the value of dg used to
    # calculate the Akaike probability is actually correct.  One approach is
    # identifying all models that have greatest Akaike probability for at 
    # least one value of dg.
    #
    # A more subtle analysis would focus not just on models with greatest
    # Akaike probability, but the set of models with non-negligible
    # probability at any given dg.  The details of the AIC method are beyond
    # the scope of this example, however.
    bm = ms.bestmodels() # Get indices for models identified as best.
    for m in bm:
        filename = "output/C60_multimodel"+str(m)+".pwa"
        cls = ms.classes_idx[m] # Get index of class to which model belongs.
        bestdgs = ms.modelbestdgs(m) # Get the uncertainties where this model has greatest Akaike probability.
        maxprob = np.max([ms.classprobs[dg][cls] for dg in bestdgs]) # Get the model's greatest Akaike probability.

        msg = ["This is a best model determined by MultiModelSelection",
              "Model: %i (of %i)",
              "Class: %i (of %i, tolerance=%f)",
              "Best model for uncertainties: %f-%f",
              "Max Akaike probability: %g"]
        msg = "\n".join(msg) %(m, len(ms.classes_idx), cls, len(ms.classes), tolerance, np.min(bestdgs), np.max(bestdgs), maxprob)
        ms.setcurrent(m) # Make this the active model.
        ms.ppe.writepwa(filename, msg)


    ## Summarize various facts about the analysis.
    print "------- Report --------"
    # Key facts
    print "Number of models: ", len(ms.results)
    print "Number of classes (tol=%s): " %tolerance, len(ms.classes)
    print "Range of dgs: %s-%s" %(ms.dgs[0], ms.dgs[-1])
    print "Number of (Nyquist) data points: ", len(r)
    bm = ms.bestmodels()
    print "Best models (class, #peaks):"
    for b in bm:
        print " %i (%i, %i)" %(b, ms.classes_idx[b], len(ms.results[b][1]))

    ## 3D plot of Akaike probabilities
    # This plot shows the Akaike probabilities of classes that make a
    # non-negligible contribution to the total, and how these vary as the
    # classes are evaluated assuming different values for the experimental
    # uncertainty.
    if plot:
        figdict = ms.plot3dclassprobs(probfilter=[0.0, 1.], figure=plt.figure())
        plt.tight_layout()
        # Uncomment to save figure
        #plt.savefig("output/C60_multimodelAICprobs.png", format="png", bbox_inches="tight")

        
    # Plots of all the models identified as best for at least some dG.
    if plot:
        for bm in ms.bestmodels():
            plt.figure()
            ms.setcurrent(bm)
            figdict = makeplot(ms, dcif)
            # Uncomment to save figure.
            #plt.savefig("output/C60_multimodel"+str(bm)+".png", format="png")


    if plot:
        plt.show()

if __name__ == '__main__':
    run()
