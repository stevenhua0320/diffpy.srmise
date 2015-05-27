.. _developers-guide-index:

####################################################
SrMise developer's documentation
####################################################

Software version |release|.

Last updated |today|.

Tool for unbiased peak extraction from atomic pair distribution functions.

The diffpy.srmise package is an implementation of the `ParSCAPE algorithm  
<https://dx.doi.org/10.1107/S2053273315005276>`_ for peak extraction from 
atomic pair distribution functions (PDFs).  It is designed to function even 
when *a priori* knowledge of the physical sample is limited, utilizing the 
Akaike Information Criterion (AIC) to estimate whether peaks are 
statistically justified relative to alternate models.  Three basic use cases 
are anticipated for diffpy.srmise.  The first is peak fitting a user-supplied 
collections of peaks.  The second is peak extraction from a PDF with no (or 
only partial) user-supplied peaks.  The third is an AIC-driven multimodeling 
analysis where the output of multiple diffpy.srmise trials are ranked. 

The framework for peak extraction defines peak-like clusters within the data, 
extracts a single peak within each cluster, and iteratively combines nearby 
clusters while performing a recursive search on the residual to identify 
occluded peaks.  Eventually this results in a single global cluster 
containing many peaks fit over all the data.  Over- and underfitting are 
discouraged by use of the AIC when adding or removing (during a pruning step) 
peaks.  Termination effects, which can lead to physically spurious peaks in 
the PDF, are incorporated in the mathematical peak model and the pruning step 
attempts to remove peaks which are fit better as termination ripples due to 
another peak. 

Where possible, diffpy.srmise provides physically reasonable default values 
for extraction parameters.  However, the PDF baseline should be estimated by 
the user before extraction, or by performing provisional peak extraction with 
varying baseline parameters.  The package defines a linear (crystalline) 
baseline, arbitrary polynomial baseline, a spherical nanoparticle baseline, 
and an arbitrary baseline interpolated from a list of user-supplied values.  
In addition, PDFs with accurate experimentally-determined uncertainties are 
necessary to provide the most reliable results, but historically such PDFs 
are rare.  In the absence of accurate uncertainties an ad hoc uncertainty 
must be specified. 


===================
Disclaimer
===================

.. literalinclude:: ../../../LICENSE.txt

.. literalinclude:: ../../../LICENSE_PDFgui.txt

================
Acknowledgments
================

Developers
-----------

diffpy.srmise is developed and maintained by

.. literalinclude:: ../../../AUTHORS.txt

The source code of *pdfdataset.py* was derived from diffpy.pdfgui.

======================================
Installation
======================================

See the `README.rst <https://github.com/diffpy/diffpy.srmise#requirements>`_
file included with the distribution.

======================================
Where next?
======================================

.. toctree::
   :maxdepth: 2

   examples.rst
   extending.rst
   
======================================
API
======================================

Detailed API documentation will be available in a future version of
diffpy.srmise.

* :ref:`genindex`
* :ref:`search`
