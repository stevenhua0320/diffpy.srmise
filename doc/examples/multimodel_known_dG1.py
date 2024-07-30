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
"""Extract multiple models from crystalline PDF with known uncertainties for
use in later AIC-driven multimodeling analysis.

The multimodel approach generates many models of varying complexity by assuming
a range of experimental uncertainties are physically plausible.  This example
shows how to generate multiple models from a crystalline silver PDF with
experimentally determined uncertainties.  The Akaike Information Criterion (AIC)
will later be used to see which models are relatively more likely to describe
the experimental data.  For complex PDFs, especially, there are many sets of
peaks which are physically distinct yet appear to fit the experimental data
similarly well.  Multimodeling can help determine which models are worth
investigating first.

NOTE: The multimodeling API used here is expected to change drastically in a
future version of diffpy.srmise.

For more information on the multimodeling approach taken here see
[1] Granlund, et al. (2015) Acta Crystallographica A, 71(4), 392-409.
    doi:10.1107/S2053273315005276
The standard reference of AIC-based multimodel selection is
[2] Burnham and Anderson. (2002). Model Selection and Multimodel Inference.
    New York, NY: Springer. doi:10.1007/b97636
"""

import numpy as np

import diffpy.srmise.srmiselog as sml
from diffpy.srmise import MultimodelSelection, PDFPeakExtraction


def run(plot=True):

    # Suppress mundane output
    # When running scripts, especially involving multiple trials, it can be
    # useful to suppress many of the diffpy.srmise messages.  Valid levels
    # include "debug", "info" (the default), "warning", "error", and
    # "critical."  See diffpy.srmise.srmiselog for more information.
    sml.setlevel("warning")

    # Initialize peak extraction from saved trial
    ppe = PDFPeakExtraction()
    ppe.read("output/query_results.srmise")
    ppe.clearcalc()

    # Set up extraction parameters
    # All parameters loaded from .srmise file.
    # Setting new values will override the previous values.
    kwds = {}
    kwds["rng"] = [10.9, 15]  # Region of PDF with some overlap.
    ppe.setvars(**kwds)

    # Create multimodel selection object.
    # The MultimodelSelection class keeps track of the results of peak
    # extraction as the assumed uncertainty dg is varied.
    ms = MultimodelSelection()
    ms.setppe(ppe)

    # Define range of dg values
    # For the purpose of illustration use 15 evenly-spaced values of dg where
    # 50% < dg < 120% of mean experimental dG in extraction range.
    dg_mean = np.mean(ppe.dy[ppe.getrangeslice()])
    dgs = np.linspace(0.5 * dg_mean, 1.2 * dg_mean, 15)

    # Perform peak extraction for each of the assumed uncertainties.
    ms.run(dgs)

    # Save results
    # The file known_dG_models.dat saves the models generated above.  The file
    # known_dG_aics.dat saves the value of the AIC of each model when evaluated
    # on a Nyquist-sampled grid using each of the dg values used to generate
    # the models in the first place.
    dr = np.pi / ppe.qmax
    ms.save("output/known_dG_models.dat")
    ms.makeaics(dgs, dr, filename="output/known_dG_aics.dat")


if __name__ == "__main__":
    run()
