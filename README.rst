#########################
SrMise
#########################

`DiffPy project <http://www.diffpy.org>`_ tool for unbiased peak extraction from
atomic pair distribution functions.

SrMise is an implementation of the `ParSCAPE algorithm
<https://dx.doi.org/10.1107/S2053273315005276>`_ for peak extraction from
atomic pair distribution functions (PDFs).  It is designed to function even
when *a priori* knowledge of the physical sample is limited, utilizing the
Akaike Information Criterion (AIC) to estimate whether peaks are
statistically justified relative to alternate models.  Three basic use cases
are anticipated for SrMise.  The first is peak fitting a user-supplied
collections of peaks.  The second is peak extraction from a PDF with no (or
only partial) user-supplied peaks.  The third is an AIC-driven multimodeling
analysis where the output of multiple SrMise trials are ranked.

The framework for peak extraction defines peak-like clusters within the data,
extracts a single peak within each cluster, and iteratively combines nearby
clusters while performing a recursive search on the residual to identify
occluded peaks.  Eventually this results in a single global cluster
containing many peaks fit over all the data.  Over- and underfitting are
discouraged by use of the AIC when adding or, during a pruning step, removing
peaks.  Termination effects, which can lead to physically spurious peaks in
the PDF, are incorporated in the mathematical peak model and the pruning step
attempts to remove peaks which are fit better as termination ripples due to
another peak.

Where possible, SrMise provides physically reasonable default values
for extraction parameters.  However, the PDF baseline should be estimated by
the user before extraction, or by performing provisional peak extraction with
varying baseline parameters.  The package defines a linear (crystalline)
baseline, arbitrary polynomial baseline, a spherical nanoparticle baseline,
and an arbitrary baseline interpolated from a list of user-supplied values.
In addition, PDFs with accurate experimentally-determined uncertainties are
necessary to provide the most reliable results, but historically such PDFs
are rare.  In the absence of accurate uncertainties an *ad hoc* uncertainty
must be specified.

For more information about SrMise, see the users manual at
http://diffpy.github.io/diffpy.srmise.

Getting Started
=================

The diffpy.srmise package requires Python 2.6 or 2.7 and the following software:

* ``setuptools`` - software distribution tools for Python
* ``NumPy`` - numerical mathematics and fast array operations for Python
* ``SciPy`` - scientific libraries for Python
* ``matplotlib`` - python plotting library

See the `SrMise license <LICENSE.txt>`__ for terms and conditions of use.
Detailed installation instructions for the `Windows`_, `Mac OS X`_, and
`Linux`_ platforms follow.

Windows
-------

Several prebuilt Python distributions for Windows include all the
prerequisite software required to run SrMise, and installing one of these is the
simplest way to get started.  These distributions are usually free for
individual and/or academic use, but some also have commercial version.  Links to
executables, installation instructions, and licensing information
for some popular options are listed below.

* `Anaconda <http://www.continuum.io/downloads>`_
* `Enthought Canopy <https://www.enthought.com/products/canopy/>`_
* `Python(x,y) <https://code.google.com/p/pythonxy/>`_
* `WinPython <http://winpython.github.io>`_

Alternately, individual Windows executables for Python and the required
components can be downloaded and installed.  The official Windows releases of
Numpy and SciPy do not currently support 64-bit Python installations, so be
sure to download the 32-bit versions of these packages.

* `Python 2.6/2.7 <https://www.python.org/downloads/windows/>`_
* `NumPy <http://sourceforge.net/projects/numpy/files/NumPy/>`_
* `SciPy <http://sourceforge.net/projects/scipy/files/scipy/>`_
* `matplotlib <http://matplotlib.org/downloads.html>`_

After installing Python and the required packages, the simplest way to obtain
SrMise is using ``pip`` to download and install the latest release from the
`Python Package Index <https://pypi.python.org>`_ (PyPI).  Open a command window
by running ``cmd`` from the Start Menu's application search box (Windows 7/8/10)
or Run command (Windows Vista and earlier).  Verify that the
``pip`` program is installed by running ::

    pip --version

If this command is not found, download and run
`get-pip.py <https://bootstrap.pypa.io/get-pip.py>`_, which will install both it
and setuptools.  For example, if the file were downloaded to the desktop, a
Windows user named ``MyName`` should run the following from the command
line: ::

    cd C:\Users\MyName\Desktop
    python get-pip.py

Finally, install the latest version of SrMise by running ::

    pip install diffpy.srmise


Mac OS X
--------

For Mac OS X systems with the MacPorts package manager, the required
software can be installed with ::

   sudo port install \
      python27 py27-setuptools py27-numpy py27-scipy py27-matplotlib

When installing for MacPorts, make sure the MacPorts bin directory is the first
in the system PATH and that python27 is selected as the default Python version
in MacPorts::

   sudo port select --set python python27

The simplest way to obtain diffpy.srmise on Mac OS X systems
is using ``pip`` to download and install the latest release from
`PyPI <https://pypi.python.org>`_. ::

   sudo pip install diffpy.srmise

Those who prefer to install from sources may download them from the
`GitHub <https://github.com/diffpy/diffpy.srmise/releases>`__ or
`PyPI <https://pypi.python.org/pypi/diffpy.srmise>`__ pages for SrMise.
Uncompress them to a directory, and from that directory run ::

   sudo python setup.py install

This installs diffpy.srmise for all users in the default system location. If
administrator (root) access is not available, see the usage info from
``python setup.py install --help`` for options to install to user-writable
directories.


Linux
-----

On Ubuntu and Debian Linux, the required software can easily be installed using
the system package manager::

   sudo apt-get install \
      python-setuptools python-numpy python-scipy python-matplotlib

Similarly, on Fedora::

    sudo yum install python-setuptools numpy scipy python-matplotlib

For other Linux distributions consult the appropriate package manager.

The simplest way to obtain diffpy.srmise on Linux systems
is using ``pip`` to download and install the latest release from the
`PyPI <https://pypi.python.org>`_. ::

   sudo pip install diffpy.srmise

Those who prefer to install from sources may download them from the
`GitHub <https://github.com/diffpy/diffpy.srmise/releases>`__ or
`PyPI <https://pypi.python.org/pypi/diffpy.srmise>`__ pages for SrMise.
Uncompress them to a directory, and from that directory run ::

   sudo python setup.py install

This installs diffpy.srmise for all users in the default system location. If
administrator (root) access is not available, see the usage info from
``python setup.py install --help`` for options to install to user-writable
directories.


DEVELOPMENT
===========

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
===============

The source code of *pdfdataset.py* was derived from diffpy.pdfgui.


CONTACTS
========

For more information on SrMise please visit the DiffPy project web-page

http://www.diffpy.org/

or email Prof. Simon Billinge at sb2896@columbia.edu.
