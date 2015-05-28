diffpy.srmise
========================================================================

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

For more information about the diffpy.srmise library, see the users manual at
http://diffpy.github.io/diffpy.srmise.

REQUIREMENTS
------------------------------------------------------------------------

The diffpy.srfit package requires Python 2.6 or 2.7 and the following software:

* ``setuptools`` - software distribution tools for Python
* ``NumPy`` - numerical mathematics and fast array operations for Python
* ``SciPy`` - scientific libraries for Python
* ``matplotlib`` - python plotting library

On Ubuntu Linux, the required software can easily be installed using
the system package manager::

   sudo apt-get install \
      python-setuptools python-numpy python-scipy python-matplotlib

For Mac OS X systems with the MacPorts package manager, the required
software can be installed with ::

   sudo port install \
      python27 py27-setuptools py27-numpy py27-scipy py27-matplotlib

When installing for MacPorts, make sure the MacPorts bin directory is the first
in the system PATH and that python27 is selected as the default Python version
in MacPorts::

   sudo port select --set python python27

For Windows systems, the easiest way to obtain ``setuptools`` if not already 
installed is downloading the ``pip`` setup script `get-pip.py 
<https://bootstrap.pypa.io/get-pip.py>`_ and running :: 

    python get-pip.py
    
It is recommended to install all other dependencies using prebuilt binaries.  
Visit http://www.scipy.org and http://www.matplotlib.org for instructions.  
Alternately, install a full Python distribution such as Python(x,y) or 
Enthought Canopy which already includes the required components. 

INSTALLATION
------------------------------------------------------------------------

The simplest way to obtain diffpy.srmise on Unix, Linux, and Mac systems is 
using ``easy_install`` or ``pip`` to download and install the latest release 
from the `Python Package Index <https://pypi.python.org>`_. :: 

   sudo pip diffpy.srmise

If you prefer to install from sources, make sure all required software packages
are in place and then run ::

   sudo python setup.py install

This installs diffpy.srmise for all users in the default system location. If 
administrator (root) access is not available, see the usage info from 
``python setup.py install --help`` for options to install to user-writable 
directories.

To install on Windows run either of the commands above omitting ``sudo``.  


DEVELOPMENT
------------------------------------------------------------------------

diffpy.srmise is open-source software developed with support of the Center of 
Research Excellence in Complex Materials at Michigan State University, in 
cooperation with the DiffPy-CMI complex modeling initiative at the Brookhaven 
National Laboratory.  The diffpy.srmise sources are hosted at 
https://github.com/diffpy/diffpy.srmise. 

Feel free to fork the project and contribute.  To install diffpy.srmise in a 
development mode, with its sources being directly used by Python rather than 
copied to a package directory, use :: 

   python setup.py develop --user


ACKNOWLEDGEMENT
------------------------------------------------------------------------

The source code of *pdfdataset.py* was derived from diffpy.pdfgui.


CONTACTS
------------------------------------------------------------------------

For more information on diffpy.srmise please visit the project web-page

http://www.diffpy.org

or email Prof. Simon Billinge at sb2896@columbia.edu.
