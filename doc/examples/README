Purpose
=======

This tutorial is an introduction to the SrMise library and command-line tool,
intended to expose new users and developers to the major use cases and options
anticipated for SrMise.

Generating interest in SrMise is another goal of these examples, and we hope
you will discover exciting ways to apply its capabilities to your scientific
goals. If you think SrMise may help you do so, please feel free to contact us
through the DiffPy website.

http://www.diffpy.org

Overview
========

SrMise is an implementation of the `ParSCAPE algorithm`_, which incorporates
standard chi-square fitting within an iterative clustering framework.  The
algorithm supposes that, in the absence of an atomic structure model, the model
complexity (informally, the number of extracted peaks) which can be
justifiably obtained from a PDF is primarily determined by the experimental
uncertainties.  The Akaike Information Criterion (AIC), summarized in the
manual, is the information-theoretic tool used to balance model complexity
with goodness-of-fit.

Three primary use cases are envisioned for SrMise:

1) Peak fitting, where user-specified peaks are fit to the experimental data.
2) Peak extraction, where the number of peaks and their parameters are
   estimated solely from the experimental data.
3) Multimodel selection, where multiple sets of peaks are ranked in an
   AIC-driven analysis to determine the most plausible sets to guide
   additional investigation.

Productively running SrMise requires, in basic, the following elements:

1) An experimental PDF.  Note that peak extraction, though not peak fitting,
   requires that all peaks of interest be positive.  This rules out peak
   extraction using SrMise for neutron PDFs obtained from samples containing
   elements with both positive and negative scattering factors.
2) The experimental uncertainties.  In principle these should be reported with
   the data, but in practice experimental uncertainties are frequently not
   reported, or are unreliable due to details of the data reduction process.
   In these cases the user should specify an *ad hoc* value.  In peak extraction
   an *ad hoc* uncertainty necessarily results in *ad hoc* model complexity, or,
   more precisely, a reasonable model complexity if the provided uncertainty
   is presumed correct.  (Even when the uncertainties are known, specifying an
   *ad hoc* value can be a pragmatic tool for exploring alternate models,
   especially in conjunction with multimodeling analysis.)  For both peak
   extraction and peak fitting the estimated uncertainties of peak parameters
   (i.e. location, width, intensity) are dependent on the experimental
   uncertainty.
3) The PDF baseline.  For crystalline samples the baseline is linear and can
   be readily estimated.  For nanoparticles more effort is required as SrMise
   includes explicit support for only a few basic shapes, although the user
   can define a baseline using arbitrary polynomials or an interpolating
   function constructed from a list of arbitrary numerical values.
4) The range over which to extract or fit peaks.  By default SrMise will use
   the entire PDF, but it is usually wise to restrict the range to the region
   of immediate interest.

The examples described below, though not exhaustive, go into detail about each
of these points.  They also cover other parameters for which good default
values can usually be estimated directly from the data.


Getting Started
===============

The examples are contained in the *doc/examples/* directory of the SrMise
`source distribution <https://github.com/diffpy/diffpy.srmise/releases>`_,
available as both a |zip| and |tar.gz| archive.  Download one of these files
(Windows users will generally favor the .zip, while Linux/Mac users the .tar.gz)
to a directory of your choosing.

Uncompress the archive.  If the downloaded file is *archivename.zip* or
*archivename.tar.gz* this will create a new directory *archivename* in its
current directory.  On Windows this can be accomplished by right-clicking
and choosing "Extract all".  On Linux/Mac OS X run, from the containing
directory, ::

    tar xvzf archivename.tar.gz

From a command window change to the *doc/examples* directory of the new
folder.  For example, a Windows' user who extracted *archivename.zip* in the
folder *C:\\Research* would type ::

    cd C:\Research\archivename\doc\examples

Every example below includes a Python script that may be run from this
directory.  While such scripts expose the full functionality of SrMise, for
many common tasks the command-line program ``srmise`` included with the package
is both sufficient and convenient, and the tutorial uses it to introduce many
fundamental concepts.  Its options may be examined in detail by running ::

    srmise --help

It is recommended to work through, in the order presented, at least the
command-line portion of each example.  Users looking for more detail should find
the copiously commented scripts helpful.

.. ~List of Examples~ ..

* Peak extraction of a single peak:
      extract_single_peak.py

* Summary of SrMise parameters:
      parameter_summary.py

* Peak extraction with initial peaks:
      extract_initial.py

* Peak fitting with intial peaks:
      fit_initial.py

* Querying SrMise results:
      query_results.py

* Multimodeling with known uncertainties:
      multimodel_known_dG1.py
      multimodel_known_dG2.py

* Multimodeling with unknown uncertainties:
      multimodel_unknown_dG1.py
      multimodel_unknown_dG2.py


.. ~Example 1~ ..

Peak extraction of a single peak
================================

| Script: extract_single_peak.py_
| Sample: `Ag`_

This introductory example shows how to extract the nearest-neighbor peak of an
X-ray PDF for crystalline (FCC) silver powder with experimentally determined
uncertainties.  It demonstrates basic input/output with SrMise, how to set the
region over which to extract peaks, and how to automatically estimate a linear
baseline.

First, plot the data without performing peak extraction.  The first argument
must be either a PDF (as here) or a .srmise file (described later) saved by
SrMise. ::

    srmise data/Ag_nyquist_qmax30.gr --no-extract --plot

This should result in an image very similar to the one below.  The top shows the
experimental data in blue.  The bottom shows the difference curve, which is
just the PDF itself since no baseline has been specified (it is identically 0),
and no peaks have been extracted.

|images/extract_single_peak1.png|

By default peak extraction is performed over the entire PDF, but often only
peaks in a particular region are of immediate interest.  In this case the
nearest-neighbor peak near 2.9 Å is well separated from all other peaks, and
performing peak extraction from around 2 Å to 3.5 Å will be sufficient.  To
restrict peak extraction to this interval use the ``--range`` option, which
accepts a pair of values. ::

    srmise data/Ag_nyquist_qmax30.gr --no-extract --plot --range 2 3.5

The PDF baseline of a crystal is linear, and a reasonable crystal baseline can
often be automatically estimated.  To estimate baseline parameters
automatically, specify the type of baseline to use with the ``--baseline``
option.  Here we specify a polynomial of degree 1, which is at present the only
baseline for which SrMise provides automatic estimation. Since the results of
peak extraction are conditioned on the baseline parameters, it is a good idea to
see whether they are reasonable. ::

    srmise data/Ag_nyquist_qmax30.gr --no-extract --plot --range 2 3.5
        --baseline "Polynomial(degree=1)"

|images/extract_single_peak2.png|

The estimated baseline looks reasonable, so it's time to perform peak extraction.
By default ``srmise`` performs extraction when run, so simply remove the
``--no-extract`` option. ::

    srmise data/Ag_nyquist_qmax30.gr --plot --range 2 3.5
        --baseline "Polynomial(degree=1)"

|images/extract_single_peak3.png|

The plot shows the fit to the data and difference curve.  The top inset shows
a vertical marker at the position of the extracted peak.  The console output
indicates the nearest-neighbor peak is located at 2.9007 ± 0.0019 Å, with width
(as full-width at half-maximum) 0.2672 ± 0.0049 Å, and intensity 9.8439 ±
0.1866.  (Results may vary slightly by platform.)  Since this PDF has
reliable uncertainties, the reported parameter uncertainties are quantitatively
sound.  Note also that these parameters are for a Gaussian in the radial
distribution function (RDF) corresponding to the experimental PDF, rather than
the Gaussian divided by radius which is the actual function being fit to the
PDF.

SrMise has two basic formats for saving data.  The first are .srmise files,
which record all information about the parameters and results of peak
extraction.  These files may be loaded by ``srmise`` and in Python scripts.  The
second are .pwa files, which are intended as a human readable summary, but do
not contain enough information to reproduce peak extraction results.  These
may be saved with the ``--save filename.srmise`` and ``--savepwa filename.pwa``
options.

The script gives results identical to the commands above, and also saves both a
.srmise and .pwa file in the *output* directory.  Verify this by running it. ::

    python extract_single_peak.py


.. ~Example 2~ ..

Summary of SrMise parameters
============================

| Script: parameter_summary.py_
| Sample: |TiO2|_

This example offers an overview of the SrMise extraction parameters, and
demonstrates their use by explicitly setting them to reasonable values in the
context of a titanium dioxide (rutile) X-ray PDF with unreliable uncertainties.

For brevity, code snippets below simply add an entry to the dictionary ``kwds``,
which sets SrMise parameters as part of the following pattern: ::

    from diffpy.srmise import PDFPeakExtraction

    ...

    ppe = PDFPeakExtraction() # Initializes main extraction object

    kwds = {}           # Dictionary for holding parameters
    ...                 # Code populating the dictionary
    ppe.setvars(**kwds) # Set parameters

Run and plot the results of this example with ::

    python parameter_summary.py

|images/parameter_summary1.png|


baseline
--------

| The PDF baseline.  Informally, a PDF is the baseline plus peaks.
| *Accepts* - Baseline with parameters, or BaselineFunction implementing
  estimation.
| *Default* - None (identically 0).  Users should specify one.

The PDF baseline is a function upon which peaks are added.  Crystalline
materials have a linear baseline, while the baseline of finite
nanomaterials is the shape-dependent "characteristic function", which is
the autocorrelation of the object.  The physical origin of the baseline
is unmeasured scattering below some minimum value of the experimental
momentum transfer, |Qmin|.  The effect of interparticle correlations is
sometimes also treated as part of the PDF baseline.  While linear baselines
are readily estimated, for other materials the user will need to exercise
judgement, as the results of peak extraction are generally conditioned upon
a reasonable choice of baseline.

Baselines may be specified by importing and instantiating the appropriate
classes, or by using a baseline loaded from an existing .srmise file.  The
following ``BaselineFunction``\ s are importable from
``diffpy.srmise.baselines``.

* Arbitrary. Any Python function which implements a simple interface.  For
  exploratory use only, as this baseline cannot be saved.  See the
  |Extending SrMise| documentation for information on creating new
  baselines.
* FromSequence.  Interpolated from lists of r and G(r) values. No
  parameters.
* NanoSpherical. Characteristic function of sphere. Radius and scale
  parameters.
* Polynomial(degree=1). Crystalline. Implements estimation.
* Polynomial(degree>1). An arbitrary polynomial.

---------

Example

The baseline G(r) = -0.65*r + 0, with the intercept fixed at 0, is visually
reasonable for the |TiO2| sample.  This baseline may be utilized from the
command-line with the ``--bpoly1 -0.65 0c`` options, or in a script as
follows: ::

    from diffpy.srmise.baselines import Polynomial

    ...

    blfunc = Polynomial(degree=1)
    slope = -.65
    intercept = 0.
    kwds["baseline"] = blfunc.actualize([slope, intercept], free=[True, False])

Run the following command to view this baseline. ::

    srmise data/TiO2_fine_qmax26.gr --bpoly1 -0.65 0c --range 0 10
        --no-extract --plot

|images/parameter_summary2.png|

cres
----

| The clustering resolution, which influences sensitivity of peak finding.
| *Accepts* - Value greater than PDF sampling rate dr.
| *Default* - Nyquist rate |pi/Qmax|, or 4*dr if |Qmax| = ∞.

The clustering resolution |d_c| determines when new clusters, and thus new
peak-like structures, are identified during the clustering phase of peak
extraction.  When a point is being clustered, it is added to an existing
cluster if the distance (along the r-axis) to the nearest cluster, d, is
less than |d_c|. (See image.)  Otherwise this point is the first in a new
cluster.  Note that SrMise oversamples the PDF during the clustering phase,
so values less than the Nyquist rate may be specified.

|images/parameter_summary3.png|

---------

Example

A clustering resolution of 0.05, about half the Nyquist sampling interval for
the |TiO2| PDF, is easily set from the command-line with the ``--cres 0.05``
option, or from a script: ::

    kwds["cres"] = 0.05


dg
--

| PDF uncertainty used during peak extraction.
| *Accepts* - Scalar or list of scalars > 0.
| *Default* - Value reported by PDF, otherwise 5% max value of PDF.

PDF reports reliable experimentally determined uncertainties, but otherwise
an *ad hoc* value must be specified for fitting.  This parameter is the
primary determinant of model complexity during peak extraction, and even
when the reported values are reliable using an *ad hoc* value can be helpful
in generating other plausible models.  This parameter can be set to a single
value, or a value for each point.  The uncertainties of most PDFs have
very little r-dependence, so using the same value for each data point often
gives points with nearly the correct relative weight.  This means the
refined value of peak parameters for a given model have very little
dependence on the absolute scale of the uncertainties.  The estimated
uncertainty of peak parameters, however, depends directly on the absolute
magnitude.

---------

Example

An *ad hoc* uncertainty of 0.35 (each point has equal weight) may be set for the
|TiO2| example from the command-line with the ``--dg 0.35`` option, or in a
script with: ::

    kwds["dg"] = 0.35

The command-line tool also includes the ``--dg-mode`` option, which exposes
several methods for setting more complex uncertainties concisely.  For details,
run ::

    srmise --help

initial_peaks
-------------

| Specifies peaks to include in model prior to extraction.
| *Accepts* - A ``Peaks`` instance containing any number of ``Peak`` instances.
| *Default* - Empty ``Peaks`` instance.

Initial peaks are held fixed for the early stages of peak extraction, which
conditions any additional peaks extracted.  In later stages initial peaks
have no special treatment, although they may be set as non-removable, which
prevents removal by pruning.

In peak fitting, this parameter specifies the peaks which are to be fit.

Two basic methods exist for setting peaks.  The first is a convenience function
which takes a position and attempts to estimate peak parameters.  The second
is manual specification, where the user provides initial values for all
peak parameters.

SrMise version |release| does not support setting initial_peaks from the
command-line.

---------

Example

Five initial peaks are specified for the |TiO2| sample, using the peak function
described in the corresponding section. The first two peaks are estimated from
position, and show the quality of estimated parameters in regions with little
peak overlap.  The other three peaks have manually specified parameters,
and occur in regions of somewhat greater overlap.  To aid convergence, the
widths of these latter peaks have been fixed at a reasonable value for a peak
arising from a single atomic pair distance.  Although initial_peaks may be set
directly, the  ``estimate_peak()`` and ``addpeaks()`` methods of
PDFPeakExtraction used below are often more convenient. ::

    from diffpy.srmise.peaks import GaussianOverR

    pf = GaussianOverR(maxwidth=0.7)

    ...

    ## Initial peaks from approximate positions.
    positions = [2.0, 4.5]
    for p in positions:
        ppe.estimate_peak(p) # adds to initial_peaks

    ## Initial peaks from explicit parameters.
    pars = [[6.2, 0.25, 2.6],[6.45, 0.25, 2.7],[7.15, 0.25, 5]]
    peaks = []
    for p in pars:
        peaks.append(pf.actualize(p, free=[True, False, True], in_format="pwa"))
    ppe.addpeaks(peaks) # adds to initial_peaks

|images/parameter_summary4.png|

While initial peaks are fixed during the early stages of peak extraction, in
later stages they are treated as any other peak identified by SrMise.  In
particular, they may be removed by pruning.  This can be prevented by setting
them as non-removable. ::

    ## Don't prune initial peaks
    for ip in ppe.initial_peaks:
        ip.removable = False


nyquist
-------

| Whether to evaluate results on Nyquist-sampled grid with dr = |pi/Qmax| .
| *Accepts* - True or False
| *Default* - True when |Qmax|>0, otherwise False.

When nyquist is False, the PDF's original sampling rate is used.  By the
Nyquist-Shannon sampling theorem, all PDFs sampled faster than the Nyquist
rate contain all the information which the experiment can provide.  Points
sampled much faster than the Nyquist rate are strongly correlated, however,
violating an assumption of chi-square fitting.  Nyquist sampling offers the
best approximation to independently-distributed uncertainties possible for a
PDF without loss of information.

---------

Example

Setting the Nyquist parameter explicitly is straightforward, although the
default value (True for this |TiO2| sample) is preferred in most cases.  From
the command line include the ``--nyquist`` or ``--no-nyquist`` option.  To use
Nyquist sampling in scripts, set ::

    kwds["nyquist"] = True

pf
--

| The peak function used for extracting peaks.
| *Accepts* - An instance of any class inheriting from PeakFunction.
| *Default* - GaussianOverR(maxwidth=0.7), reasonable if r-axis in Å.

The following peak functions are importable from ``diffpy.srmise.peaks``.

* GaussianOverR(maxwidth).  A Gaussian divided by radius r.  Maxwidth gives
  maximum full-width at half maximum, to reduce likelihood of unphysically
  wide peaks.
* Gaussian(maxwidth).  A Gaussian with a maximum width, as above.
* TerminationRipples(base_pf, qmax).  Modifies another peak function,
  base_pf, to include termination effects for given |Qmax|.  Peak
  extraction automatically applies termination ripples to peaks, but they
  should be specified explicitly if using SrMise for peak fitting.

---------

Example

The default peak function is reasonable for the |TiO2| example, but can be
explicitly specified from the command-line with ``--pf "GaussianOverR(0.7)"``.
In scripts, use ::

    from diffpy.srmise.peaks import GaussianOverR

    ...

    kwds["pf"] = GaussianOverR(0.7)


qmax
----

| The experimental maximum momentum transfer |Qmax|.
| *Accepts* - Any value ≥ 0, where 0 indicates an infinite value.  Also
  allows "automatic", which estimates |Qmax| from the data.
| *Default* - Value reported by data.  If not available, uses automatic
  estimate.

|Qmax| is responsible for the characteristic termination ripples observed
in the PDF.  SrMise models termination effects by taking the Fourier
transform of a peak, zeroing all components above |Qmax|, and performing the
inverse transform back to real-space.  PDFs where termination ripples were
smoothed during data reduction (e.g. using a Hann window) will be fit less
well.

---------

Example

For the |TiO2| PDF, |Qmax| = 26 |angstrom^-1|, which can be explicitly set
with ::

    kwds["qmax"] = 26.0

Alternately, to automatically estimate |Qmax| from the data (about 25.9
|angstrom^-1| in this case), use ::

    kwds["qmax"] = "automatic"

At the command-line, both the ``--qmax 26.0`` and ``--qmax automatic`` options
are valid.


rng
---

| The range over which to perform peak extraction or peak fitting.
| *Accepts* - A list [rmin, rmax], where rmin < rmax and neither fall
  outside the data.  May specify either as ``None`` to use default value.
| *Default* - The first and last r-values, respectively, in the PDF.

Users are encouraged to restrict fits to the regions of immediate interest.

---------

Example

To extract peaks from the |TiO2| sample between 1.5 and 10 Å, in scripts use ::

    kwds["rng"] = [1.5, 10]

At the command-line use ``--range 1.5 10``.


supersample
-----------

| Minimum degree to oversample PDF during early stages of peak extraction.
| *Accepts* - A value ≥ 1.
| *Default* - 4.0

Peak extraction oversamples the PDF during the early phases to assist in
peak finding.  This value specifies a multiple of the Nyquist rate,
equivalent to dividing the Nyquist sampling interval dr = |pi/Qmax| by this
value.  The supersample parameter has no effect if the input PDF is already
sampled faster than this.

Note that large degrees of supersampling, whether due to this parameter or
the input PDF, negatively impact the time required for chi-square fitting.

---------

Example

The default value is sufficient for the |TiO2| sample, but to set explicitly in
a script use ::

    kwds["supersample"] = 4.0

or ``--supersample 4.0`` at the command-line.


.. ~Example 3~ ..

Peak fitting
============

| Script: fit_initial.py_
| Sample: |C60|_

This example demonstrates peak fitting with a set of initial peaks on a |C60|
nanoparticle PDF with unreliable uncertainties.  An interpolated baseline is
created from a list of (r, G(r)) pairs contained in a file.  Note that the
command-line tool ``srmise`` does not currently support peak fitting.

The initial peaks are specified as in the previous example, by giving an
approximate list of peak positions to an estimation routine, or manually
specifying peak parameters.  Peak fitting never alters the peak function, so
termination effects are explicitly added to an existing peak function with
the following pattern. ::

    from diffpy.srmise.peaks import TerminationRipples

    ...

    # Return new peak function instance which adds termination effects to
    # existing peak function instance "base_pf" with maximum momentum transfer
    # "qmax".
    pf = TerminationRipples(base_pf, qmax)

The initial peaks used in this fit are shown below.  The last two peaks use
manually specified parameters.  Note that this PDF is unnormalized, so the
scale of the y-axis is arbitrary.

|images/fit_initial1.png|

By default, peak fitting occurs on a Nyquist-sampled grid when |qmax| > 0.  To
fit a finely-sampled PDF without resampling set "nyquist" to False.

Run the script to see the results of the fit. ::

    python fit_initial.py

|images/fit_initial2.png|


.. ~Example 4~ ..

Querying SrMise results
=======================

| Script: query_results.py_
| Sample: `Ag`_

In previous examples the results of peak extraction/fitting were read from the
console, but this is not always convenient.  This example demonstrates the
basic methods for querying SrMise objects about their parameters and values
within scripts.

First, visually check that the baseline obtained in the earlier silver example
(set using the ``--bsrmise filename.srmise`` option) is reasonable over a
larger range. ::

    srmise data/Ag_nyquist_qmax30.gr --no-extract --plot --range 2 10
        --bsrmise output/query_results.srmise

|images/query_results1.png|

Next, run ::

    python query_results.py

to perform peak extraction, the example analysis, and obtain the two plots
below.

|images/query_results2.png|

|images/query_results3.png|

Now the methods of the script are described.  The way to evaluate model
uncertainties with SrMise is with a ``ModelCovariance`` instance returned after
peak extraction (or fitting). ::

    cov = ppe.extract()

Model parameters denoted by a tuple ``(i,j)``, representing the *j*\ th
parameter of the *i*\ th peak, are passed to this object's methods.  For a
Gaussian over r, the order of peak parameters in SrMise is position, width, and
area.  Thus, the area of the nearest-neighbor peak is denoted by the tuple
(0,2). (Indexing is zero-based.) By convention, the last element (``i=-1``) is
the baseline. The methods of greatest interest are as follows. ::

    # Get (value, uncertainty) tuple for this parameter
    cov.get((i,j))

    # Get just the value of the parameter
    cov.getvalue((i,j))

    # Get just the uncertainty of the parameter
    cov.getuncertainty((i,j))

    # Get the covariance between two parameters
    cov.getcovariance((i1,j1), (i2,j2))

    # Get the correlation between two parameters
    cov.getcorrelation((i1,j1), (i2,j2))

Suppose, for example, one wants to emperically estimate the number of silver
atoms contributing to each occupied coordination shell of the FCC structure,
knowing that the coordination number (i.e. nearest neighbors) of an ideal FCC
structure is exactly 12.  For a monoelemental material the intensity of a peak
in a properly normalized PDF should equal the number of contributing atoms in
the corresponding shell.  Thus, the intensity of the nearest neighbor peak
permits an estimate of the PDF scale factor, and using that an estimate of other
shell's occupancy.  This simple procedure is implemented in the script using
model parameters and uncertainties obtained with the methods above.

Another useful capability is calculating the value of a model, in whole or
part.  Given a PDFPeakExtraction instance ``ppe`` and a numpy array ``r``, the
usual methods are ::

    # Return sum of peaks and baseline, evaluated on the current grid, or r.
    ppe.extracted.value()
    ppe.extracted.value(r)

    # Return residual (data - model) on the current grid.
    ppe.extracted.residual()

    # Return the baseline evaluated on the current grid, or r.
    ppe.extracted.valuebl()
    ppe.extracted.valuebl(r)

    # The ith peak evaluated on r
    ppe.extracted.model[i].value(r)

    # The baseline evaluated on r
    ppe.extracted.baseline.value(r)


.. ~Example 5~ ..

Multimodeling with known uncertainty
====================================

| Scripts: multimodel_known_dG1.py_, multimodel_known_dG2.py_
| Sample: `Ag`_

.. note ::

    This example is intended for advanced users.  The API for multimodel
    selection is expected to change dramatically in a future version of SrMise.
    At present multimodel selection in SrMise requires each point in the PDF be
    assigned identical uncertainty.  Using the mean uncertainty of a
    Nyquist-sampled PDF is suggested.

The Akaike information criterion (AIC) is a powerful but straightforward method
for multimodel selection, which in the context of SrMise means ranking
individual models (i.e. a set of peaks and baselines) by their "Akaike
probabilities."  These are, in brief, an asymptotic estimate of the expected
(with respect to the data) and relative (with respect to the models under
consideration) likelihood that a particular model is the best model in the sense
of least Kullback-Leibler divergence.  This approach has a basis in fundamental
concepts of information theory, and shares some conceptual similarities to
cross validation.  Qualitatively speaking, a good model is both simple (has
few parameters) and fits the data well (has a small residual, e.g. the
chi-square error), but improving one of these often comes with a cost to the
other.  The AIC manages this tradeoff between model complexity and
goodness-of-fit.

A formal introduction to these methods is beyond the scope of this example.
Investigators are encouraged to consult Burnham and Anderson's "Model Selection
and Multimodel Inference" (New York: Springer, 2002) for a general introduction
to AIC-based multimodeling.  See also the paper describing SrMise and the
`ParSCAPE algorithm`_ for details about this method as applied to peak
extraction from pair distribution functions.

The suggested approach to multimodel selection with SrMise is generating a
population of models of varying complexity by treating the experimental
uncertainty dG of PDF G(r) as a parameter, denoted dg, across multiple SrMise
trials.  These models may be very similar, and such model redundancy is reduced
by creating classes of similar models, namely those with the same number and
type of peak, and very similar peak parameters.  Each class, represented by the
constituent model with least chi-square error, is considered distinct for the
purpose of the multimodel analysis.  The Akaike probabilities are evaluated for
each class, and from these the investigator may identify the set of most
promising models for additional analysis.  The investigator's *a priori*
knowledge of a system, such that a particular model is unphysical, can be
leveraged by excluding that model and recalculating the Akaike probabilities.

Details of the multimodeling procedure are discussed in the comments of the
extraction and analysis scripts.  Run these, noting that the extraction script
may take several minutes to complete. ::

    python multimodel_known_dG1.py
    python multimodel_known_dG2.py

.. ~Example 6~ ..

Multimodeling with unknown uncertainty
======================================

| Scripts: multimodel_unknown_dG1.py_, multimodel_unknown_dG2.py_
| Sample: |C60|_

.. note ::

    This example is intended for advanced users.  The API for multimodel
    selection is expected to change dramatically in a future version of SrMise.
    At present multimodel selection in SrMise requires each point in the PDF
    be assigned identical uncertainty.

Multimodel selection with SrMise when experimental uncertainties are unknown is
challenging, and without an independent estimate of these uncertainties the
usual AIC-based multimodeling analysis of the previous example is not possible.
To be specific, the procedure in the previous example can be carried out under
the assumption that some *ad hoc* uncertainty is correct, but results are
clearly dependent upon that choice.

The approach taken here, detailed in the paper describing the
`ParSCAPE algorithm`_, is to evaluate the Akaike probabilities over a broad
range of *ad hoc* uncertainties, typically the same range used to generate the
population of models in the first place.  Then, identify the set of classes
which have greatest Akaike probability for at least one of the evaluated
uncertainties.  This choice embodies ignorance of experimentally-determined
uncertainties.  Unlike a standard AIC-based multimodeling analysis, these
classes have no particular information theoretic relationship with each other
since their Akaike probabilities were calculated assuming different
uncertainties.  However, if the true experimental uncertainty lies within the
interval tested, this set of classes necessarily contains the one that would be
selected as best were the experimental uncertainties known.  This is,
nevertheless, a significantly less powerful analysis than is possible when
the experimental uncertainties are known.

Details of the multimodeling procedure are discussed in the comments of the
extraction and analysis scripts.  Run these, noting that the extraction script
may take several minutes to complete. ::

    python multimodel_unknown_dG1.py
    python multimodel_unknown_dG2.py

.. ~PDF Info~ ..

PDF Information
===============

Information on the sample, experimental methods, and data reduction procedures
for the example PDFs are summarized below.  Special attention is given to why
each PDF does or does not report reliable uncertainties.


Ag
--

A synchotron X-ray PDF (|Qmax| = 30 |angstrom^-1|, Nyquist sampled) with
reliable experimentally-estimated uncertainties for a crystalline powder of
face-centered cubic silver.  The 2D diffraction pattern was measured
on an integrating detector.  A Q-space 1D pattern with nearly uncorrelated
experimentally-estimated uncertainties was obtained using `SrXplanar`_.  All
other data reduction was performed using `PDFgetX2`_.

Reliable experimental uncertainties were preserved during error propagation to
the PDF by transforming the 1D pattern to the minimally-correlated (Nyquist)
grid without intermediate resampling.


|C60|
-----

A synchotron X-ray PDF (|Qmax| = 21.3 |angstrom^-1|, finely sampled) for a
powder of buckminsterfullerene nanoparticles in a face-centered cubic
lattice, but with no fixed orientation at the lattice sites.  The 2D diffraction
pattern was measured on an integrating detector.  A 2\ *θ* 1D pattern without
propagated uncertainties was obtained using `FIT2D`_.  All other data reduction
was performed using `PDFgetX2`_.  This PDF is unnormalized, so the scale of the
y-axis is arbitrary.  The nanoparticle baseline used for testing this PDF with
SrMise is a fit to the observed interparticle contribution using an empirical
model of thin spherical shells of constant density in an FCC lattice.

This PDF has unreliable uncertainties.  Since the 1D pattern reports no
uncertainty, PDFgetX2 treats the uncertainty as equal to the square-root of the
values in the 1D pattern, which is invalid for integrating detectors.  Moreover,
the 1D pattern must be resampled onto a Q-space grid before the PDF can be
calculated, and this introduces correlations between points. Finally, the PDF is
itself oversampled, resulting in further correlations.


|TiO2|
------

A synchotron X-ray PDF (|Qmax| = 26 |angstrom^-1|, finely sampled) for a
crystalline powder of titanium dioxide (rutile).  The 2D diffraction pattern was
measured on an integrating detector.  A Q-space 1D pattern with nearly
uncorrelated experimentally-estimated uncertainties was obtained using
`SrXplanar`_.  All other data reduction was performed using `PDFgetX2`_.

Although the 1D diffraction pattern has reliable uncertainties, this PDF was
(illustratively) sampled faster than the Nyquist rate, introducing significant
correlations between nearby data points.  Resampling this PDF at the Nyquist
rate cannot recover reliable uncertainties unless the full variance-covariance
matrix has been preserved and is propagated during resampling.
