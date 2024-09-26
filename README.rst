|Icon| |title|_
===============

.. |title| replace:: diffpy.srmise
.. _title: https://diffpy.github.io/diffpy.srmise

.. |Icon| image:: https://avatars.githubusercontent.com/diffpy
        :target: https://diffpy.github.io/diffpy.srmise
        :height: 100px

|PyPi| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/diffpy/diffpy.srmise/actions/workflows/main.yml/badge.svg
        :target: https://github.com/diffpy/diffpy.srmise/actions/workflows/main.yml

.. |Codecov| image:: https://codecov.io/gh/diffpy/diffpy.srmise/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/diffpy/diffpy.srmise

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/diffpy.srmise
        :target: https://anaconda.org/conda-forge/diffpy.srmise

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/diffpy.srmise
        :target: https://pypi.org/project/diffpy.srmise/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/diffpy.srmise
        :target: https://pypi.org/project/diffpy.srmise/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/diffpy/diffpy.srmise/issues

implementation of the ParSCAPE algorithm for peak extraction from atomic pair distribution functions (PDFs)

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

For more information about the diffpy.srmise library, please consult our `online documentation <https://diffpy.github.io/diffpy.srmise>`_.

Citation
--------

If you use diffpy.srmise in a scientific publication, we would like you to cite this package as

        L. Granlund, Billinge, S. J. L., and P. M. Duxbury. “Algorithm for systematic peak extraction from atomic
        pair distribution functions”. In: Acta Crystallogr. A 71.4 (2015), pp. 392–409. DOI:
        10.1107/S2053273315005276

Installation
------------

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
and install from the "conda-forge" channel of Conda packages.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``diffpy.srmise_env`` ::

        conda create -n diffpy.srmise_env python=3
        conda activate diffpy.srmise_env

Then, to fully install ``diffpy.srmise`` in our active environment, run ::

        conda install diffpy.srmise

Another option is to use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``diffpy.srmise_env`` environment, we will also have to install dependencies ::

        pip install -r https://raw.githubusercontent.com/diffpy/diffpy.srmise/main/requirements/run.txt

and then install the package ::

        pip install diffpy.srmise

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/diffpy/diffpy.srmise/>`_. Once installed, ``cd`` into your ``diffpy.srmise`` directory
and run the following ::

        pip install .

Support and Contribute
----------------------

`Diffpy user group <https://groups.google.com/g/diffpy-users>`_ is the discussion forum for general questions and discussions about the use of diffpy.srmise. Please join the diffpy.srmise users community by joining the Google group. The diffpy.srmise project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/diffpy/diffpy.srmise/issues>`_ and/or `submit a fix as a PR <https://github.com/diffpy/diffpy.srmise/pulls>`_. You can also post it to the `Diffpy user group <https://groups.google.com/g/diffpy-users>`_.

Feel free to fork the project and contribute. To install diffpy.srmise
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

Improvements and fixes are always appreciated.

Before contribuing, please read our `Code of Conduct <https://github.com/diffpy/diffpy.srmise/blob/main/CODE_OF_CONDUCT.rst>`_.

Contact
-------

For more information on diffpy.srmise please visit the project `web-page <https://diffpy.github.io/>`_ or email Prof. Simon Billinge at sb2896@columbia.edu.
