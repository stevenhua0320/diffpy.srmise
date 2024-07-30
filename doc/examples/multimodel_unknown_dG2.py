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
"""AIC-driven multimodel analysis of nanoparticle PDF with unknown
uncertainties.

The multimodel approach generates many models of varying complexity by assuming
a range of experimental uncertainties are physically plausible.  This example
shows how to analyze multiple models obtained from a C60 nanoparticle PDF with
unreliable uncertainties.  The Akaike Information Criterion (AIC) can be used to
see which models are relatively more likely to describe the experimental data.
For complex PDFs, especially, there are many sets of peaks which are physically
distinct yet appear to fit the experimental data similarly well.  Here we
calculate the Akaike probabilities, which are a measure of the likelihood that a
given model is the best model (in the sense of Kullback-Leibler information
divergence) relative to all the other ones in the same comparison.  This
analysis reflects ignorance of the experimental uncertainties by evaluating
the Akaike probabilities for a range of assumed uncertainties, returning models
which are selected as best at least once.  This is a weaker analysis than
possible when the uncertainties are known.

NOTE: The multimodeling API used here is expected to change drastically in a
future version of diffpy.srmise.

For more information on the multimodeling approach taken here see
[1] Granlund, et al. (2015) Acta Crystallographica A, 71(4), 392-409.
    doi:10.1107/S2053273315005276
The standard reference of AIC-based multimodel selection is
[2] Burnham and Anderson. (2002). Model Selection and Multimodel Inference.
    New York, NY: Springer. doi:10.1007/b97636
"""
import matplotlib.pyplot as plt
import numpy as np

import diffpy.srmise.srmiselog as sml
from diffpy.srmise import MultimodelSelection
from diffpy.srmise.applications.plot import makeplot

# distances from ideal (unrefined) C60
dcif = np.array(
    [
        1.44,
        2.329968944,
        2.494153163,
        2.88,
        3.595985339,
        3.704477734,
        4.132591264,
        4.520339129,
        4.659937888,
        4.877358006,
        5.209968944,
        5.405310018,
        5.522583786,
        5.818426502,
        6.099937888,
        6.164518388,
        6.529777754,
        6.686673127,
        6.745638756,
        6.989906831,
        7.136693738,
    ]
)


def run(plot=True):

    # Suppress mundane output
    sml.setlevel("warning")

    # Create multimodeling object and load diffpy.srmise results from file.
    ms = MultimodelSelection()
    ms.load("output/unknown_dG_models.dat")
    ms.loadaics("output/unknown_dG_aics.dat")

    # Use Nyquist sampling
    # Standard AIC analysis assumes the data have independent uncertainties.
    # Nyquist sampling minimizes correlations in the PDF, which is the closest
    # approximation to independence possible for the PDF.
    dr = np.pi / ms.ppe.qmax
    (r, y, dr2, dy) = ms.ppe.resampledata(dr)

    # Classify models
    # All models are placed into classes.  Models in the same class
    # should be essentially identical (same peak parameters, etc.)
    # up to a small tolerance determined by comparing individual peaks. The
    # best model in each class essentially stands in for all the other models
    # in a class in the rest of the analysis.  A tolerance of 0 indicates the
    # models must be exactly identical.  Increasing the tolerance allows
    # increasingly different models to be classified as "identical."  This step
    # reduces a major source of model redundancy, which otherwise weakens
    # AIC-based analysis.  As a rule of thumb, AIC-based analysis is robust
    # to redundant poor models (since they contribute very little to the Akaike
    # probabilities in any case), but redundant good models can significantly
    # alter how models are ranked.  See Granlund (2015) for details.
    tolerance = 0.2
    ms.classify(r, tolerance)

    # Summarize various facts about the analysis.
    num_models = len(ms.results)
    num_classes = len(ms.classes)
    print("------- Multimodeling Summary --------")
    print("Models: %i" % num_models)
    print("Classes: %i (tol=%s)" % (num_classes, tolerance))
    print("Range of dgs: %f-%f" % (ms.dgs[0], ms.dgs[-1]))
    print("Nyquist-sampled data points: %i" % len(r))

    # Find "best" models.
    # In short, models with greatest Akaike probability.  Akaike probabilities
    # can only be validly compared if they were calculated for identical data,
    # namely identical PDF values *and* uncertainties, and are only reliable
    # with respect to the actual experiment when using a Nyquist-sampled PDF
    # with experimentally determined uncertainties.
    #
    # In the present case the PDF uncertainties are not reliable, and so the
    # analysis cannot be performed by specifying the experimental uncertainty
    # dG.  Instead, perform a weaker analysis, calculating the Akaike
    # probabilities for a range of assumed dG, and identifying classes which
    # have greatest probability at least once.  The classes identified in this
    # way have no particular information-theoretic relationship, but if the
    # actual experimental uncertainty is in the interval tested, the best
    # class at the experimental uncertainty is among them.

    # Get classes which are best for one or more dG, and the specific dG in that
    # interval at which they attain greatest Akaike probability.
    best_classes = np.unique([ms.get_class(dG) for dG in ms.dgs])
    best_dGs = []
    for cls in best_classes:
        cls_probs = [ms.get_prob(dG) if ms.get_class(dG) == cls else 0 for dG in ms.dgs]
        dG = ms.dgs[np.argmax(cls_probs)]
        best_dGs.append(dG)

    print("\n--------- Best models for at least one dG ---------" % dG)
    print("   Best dG  Model  Class  Free       AIC     Prob  File")
    for dG in best_dGs:

        # Generate information about best model.
        # The get(dG, *args, **kwds) method returns a tuple of values
        # corresponding to string arguments for the best model in best class at
        # given dG. When the corder keyword is given it returns the model from
        # the corderth best class (where 0 is best, 1 is next best, etc.)
        # "model" -> index of model
        # "class" -> index of class
        # "nfree" -> number of free parameters in corresponding model
        # "aic" -> The AIC for this model given uncertainty dG
        # "prob" -> The AIC probability given uncertainty dG
        # These all have dedicated getter functions.
        (model, cls, nfree, aic, prob) = ms.get(dG, "model", "class", "nfree", "aic", "prob")

        filename_base = "output/unknown_dG_m" + str(model)

        # print(info for this model
        print(
            "%10.4e  %5i  %5i  %4i  %10.4e %6.3f  %s" % (dG, model, cls, nfree, aic, prob, filename_base + ".pwa")
        )

        # A message added as a comment to saved .pwa file.
        best_from = [dg for dg in ms.dgs if ms.get_class(dg) == cls]
        msg = [
            "Multimodeling Summary",
            "---------------------",
            "Model: %i (of %i)" % (model, num_models),
            "Class: %i (of %i, tol=%s)" % (cls, num_classes, tolerance),
            "Best model from dG: %s-%s" % (best_from[0], best_from[-1]),
            "Evaluated at dG: %s" % dG,
            "Akaike probability: %g" % prob,
        ]
        msg = "\n".join(msg)

        # Make this the active model
        ms.setcurrent(model)

        # Save .pwa
        ms.ppe.writepwa(filename_base + ".pwa", msg)

        # Plot this model
        if plot:
            plt.figure()
            makeplot(ms.ppe, dcif)
            plt.title("Model %i/Class %i (Best dG=%f, AIC prob=%f)" % (model, cls, dG, prob))
            # Uncomment line below to save figures.
            # plt.savefig(filename_base + ".png", format="png")

    # 3D plot of Akaike probabilities
    # This plot shows the Akaike probabilities of all classes as a function
    # of assumed uncertainty dG.  This gives a rough sense of how the models
    # selected by an AIC-based analysis would vary if the experimental
    # uncertainties contributing to the observed G(r) were different.  Models
    # are highlighted at the various dG values found above.
    if plot:
        plt.figure()
        ms.plot3dclassprobs(probfilter=[0.1, 1.0], highlight=best_dGs)
        plt.tight_layout()
        # Uncomment line below to save figure.
        # plt.savefig("output/unknown_dG_probs.png", format="png", bbox_inches="tight")

    if plot:
        plt.show()


if __name__ == "__main__":
    run()
