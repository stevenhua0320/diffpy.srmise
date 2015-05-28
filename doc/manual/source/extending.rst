.. _developers-guide-extending:

===================
Extending SrMise
===================

The :ref:`developers-guide-examples` give an overview of how to
use SrMise with the existing peak and baseline functions.  These inherit from
classes providing generic peak and baseline functionality, and from which
additional peaks and baselines can be implemented.  The process for adding
new peaks and baselines is summarized below, but see the source code for
additional details.

If you implement a peak or baseline likely to be of broad interest to the PDF
community, please consider submitting a pull request to the GitHub
`SrMise repository  <https://github.com/diffpy/diffpy.srmise>`_.

Organization of Functions
-------------------------

The ``BaseFunction`` class in ``diffpy.srmise.basefunction`` implements the
functionality common to all SrMise baseline and peak functions, which are
separately implemented in the ``diffpy.srmise.baselines`` and
``diffpy.srmise.peaks`` subpackages.  Specific baseline and peak functions
inherit from the ``BaselineFunction`` and ``PeakFunction`` classes in those
subpackges, as shown below.

* .. py:class:: BaseFunction

    + .. py:class:: BaselineFunction
    
        - .. py:class:: FromSequence
        - .. py:class:: NanoSpherical
        - .. py:class:: Polynomial
        - *etc.*
        
    + .. py:class:: PeakFunction
    
        - .. py:class:: Gaussian
        - .. py:class:: GaussianOverR
        - *etc.*

Adding Baselines
-------------------------------------

To add a baseline, create a new module which defines a class inheriting from
``diffpy.srmise.baselines.base.BaselineFunction``.  The class data and methods
which need to be implemented are summarized in the source code.


.. literalinclude:: ../../../diffpy/srmise/baselines/base.py
    :pyobject: BaselineFunction
    :end-before: __init__

The class methods should follow these specifications.  See existing baselines
for examples.

.. py:method:: estimate_parameters(r, y)

    Return a Numpy array of parameters estimated from the data.
    
    :param r: Grid on which the data are defined.
    :param y: The data.
    :type r: `Sequence`
    :type y: `Sequence`
    :returns: Estimated parameters
    :rtype: `numpy.ndarray`
    :raises: NotImplementedError if estimation has not been implemented.
    :raises: SrMiseEstimationError if estimation fails.


.. py:method:: _jacobian_raw(pars, r, free)
   
    Return Jacobian for parameters evaluated over `r`.
    
    :param pars: The parameters of the baseline.
    :param r: Scalar or grid on which to calculate the Jacobian.
    :param free: Boolean values indicating if corresponding parameter is free (True) or fixed (False).
    :type pars: `Sequence(float)`
    :type r: `int`, `float`, or `Sequence(int` or `float)`
    :type free: `Sequence(boolean)`
    :returns: List of Jacobian values (or None if parameter is not free) for each parameter evaluated at `r`.
    :rtype: `list(numpy.ndarray(float) or float or None)`

.. py:method:: _transform_derivativesraw(pars, in_format, out_format)

    Return the gradient matrix of `pars` represented in format 'out_format'.
    
    :param pars: The parameters of the baseline.
    :param in_format: The format of `pars`.
    :param out_format: The desired format of `pars`.
    :type pars: `Sequence(float)`
    :type in_format: `str`
    :type out_format: `str`
    :returns: The gradient matrix for the transformation.
    :rtype: `numpy.ndarray`

.. py:method:: _transform_parametersraw(pars, in_format, out_format)

    Return parameters transformed into format 'out_format'.
    
    :param pars: The parameters of the baseline.
    :param in_format: The format of `pars`.
    :param out_format: The desired format of `pars`.
    :type pars: `Sequence(float)`
    :type in_format: `str`
    :type out_format: `str`
    :returns: The transformed parameters.
    :rtype: `numpy.ndarray`
    
.. py:method:: _valueraw(pars, r)

    Return value of baseline with given parameters at r.
    
    :param pars: The parameters of the baseline.
    :param r: Scalar or grid on which to calculate the baseline.
    :type pars: `Sequence(float)`
    :type r: `int`, `float`, or `Sequence(int` or `float)`
    :returns: The value of the baseline.
    :rtype: `float` or `numpy.ndarray(float)`.


Adding Peaks
--------------------------

To add a new peak function, create a new module which defines a class
inheriting from ``diffpy.srmise.peaks.base.PeakFunction``.  Implementing a peak
function is nearly identical to implementing a baseline function, with the
following differences:

1) The ``estimate_parameters`` method is required.
2) The "position" key must be defined in the ``parameterdict`` class member.
3) Peak functions must implement the additional method ``scale_at``.
   
.. py:method:: scale_at(pars, r, scale)

    Return peak parameters such that the value at ``r`` is scaled by ``scale``
    while the position of the peak's maxima remains unchanged.
    
    :param pars: The parameters of the baseline.
    :param r: Position where the peak will be rescaled.
    :param scale: A scale factor > 0.
    :type pars: `Sequence(float)`
    :type r: `int` or `float`
    :type scale: `float`
    :returns: The adjusted peak parameters.
    :rtype: `numpy.ndarray(float)`.

