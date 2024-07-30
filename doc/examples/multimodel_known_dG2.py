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
"""AIC-driven multimodel analysis of crystalline PDF with known uncertainties.

The multimodel approach generates many models of varying complexity by assuming
a range of experimental uncertainties are physically plausible.  This example
shows how to analyze multiple models obtained (in the previous script) from a
crystalline silver PDF with experimentally determined uncertainties.  This
involves calculating the Akaike probabilities, which are a measure of the
likelihood that a given model is the best model (in the sense of
Kullback-Leibler divergence) relative to all the other ones in the same
comparison.

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

# distances from ideal Ag (refined to PDF)
dcif = np.array(
    [
        11.2394,
        11.608,
        11.9652,
        12.3121,
        12.6495,
        12.9781,
        13.2986,
        13.6116,
        13.9175,
        14.2168,
        14.51,
        14.7973,
    ]
)


def run(plot=True):

    # Suppress mundane output
    sml.setlevel("warning")

    # Create multimodeling object and load diffpy.srmise results from file.
    ms = MultimodelSelection()
    ms.load("output/known_dG_models.dat")
    ms.loadaics("output/known_dG_aics.dat")

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

    # Get dG usable as key in analysis.
    # The Akaike probabilities were calculated for many assumed values of the
    # experimental uncertainty dG, and each of these assumed dG is used as a
    # key when obtaining the corresponding results.  Numerical precision can
    # make recalculating the exact value difficult, so the dg_key method returns
    # the key closest to its argument.
    dG = ms.dg_key(np.mean(ms.ppe.dy))

    # Find "best" models.
    # In short, models with greatest Akaike probability.  Akaike probabilities
    # can only be validly compared if they were calculated for identical data,
    # namely identical PDF values *and* uncertainties, and are only reliable
    # with respect to the actual experiment when using a Nyquist-sampled PDF
    # with experimentally determined uncertainties.
    #
    # The present PDF satisifes these conditions, so the rankings below reflect
    # an AIC-based estimate of which of the tested models the data best support.
    print("\n--------- Model Rankings for dG = %f ---------" % dG)
    print("Rank  Model  Class  Free         AIC   Prob  File")
    for i in range(len(ms.classes)):

        # Generate information about best model in ith best class.
        # The get(dG, *args, **kwds) method returns a tuple of values
        # corresponding to string arguments for the best model in best class at
        # given dG. When the corder keyword is given it returns the model from
        # the corderth best class (where 0 is best, 1 is next best, etc.)
        # "model" -> index of model
        # "class" -> index of class
        # "nfree" -> number of free parameters in corresponding model
        # "aic" -> The AIC for this model given uncertainty dG
        # "prob" -> The AIC probability given uncertainty dG
        # These all have dedicated getter functions.  For example, the model
        # index can also be obtained using get_model(dG, corder=i)
        (model, cls, nfree, aic, prob) = ms.get(dG, "model", "class", "nfree", "aic", "prob", corder=i)

        filename_base = "output/known_dG_m" + str(model)

        # print(info for this model
        print(
            "%4i  %5i  %5i  %4i  %10.4e %6.3f  %s" % (i + 1, model, cls, nfree, aic, prob, filename_base + ".pwa")
        )

        # A message added as a comment to saved .pwa file.
        msg = [
            "Multimodeling Summary",
            "---------------------",
            "Evaluated at dG: %s" % dG,
            "Model: %i (of %i)" % (model, num_models),
            "Class: %i (of %i, tol=%s)" % (cls, num_classes, tolerance),
            "Akaike probability: %g" % prob,
            "Rank: %i" % (i + 1),
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
            plt.title("Model %i/Class %i (Rank %i, AIC prob=%f)" % (model, cls, i + 1, prob))
            # Uncomment line below to save figures.
            # plt.savefig(filename_base + ".png", format="png")

    # 3D plot of Akaike probabilities
    # This plot shows the Akaike probabilities of all classes as a function
    # of assumed uncertainty dG.  This gives a rough sense of how the models
    # selected by an AIC-based analysis would vary if the experimental
    # uncertainties contributing to the observed G(r) were different.  The
    # Akaike probabilities calculated for the actual experimental uncertainty
    # are highlighted.
    if plot:
        plt.figure()
        ms.plot3dclassprobs(probfilter=[0.0, 1.0], highlight=[dG])
        plt.tight_layout()
        # Uncomment line below to save figure.
        # plt.savefig("output/known_dG_probs.png", format="png", bbox_inches="tight")

    if plot:
        plt.show()


if __name__ == "__main__":
    run()
