Transforming Flow Cytometry Data
================================

This tutorial focuses on how to perform basic transformations to flow cytometry data using ``FlowCal``, particularly by using the module :mod:`FlowCal.transform`

To start, navigate to the ``examples/FCFiles`` directory included with FlowCal, and open a ``python`` session therein. Then, import ``FlowCal`` as with any other python module.

>>> import FlowCal

Exponential Transformation
--------------------------

Start by loading file ``data_001.fcs`` into an ``FCSData`` object called ``s``.

>>> s = FlowCal.io.FCSData('data_001.fcs')

Let's now visualize the contents of the ``FL1`` channel. We will explore ``FlowCal``'s plotting functions in the :doc:`plotting tutorial </python_tutorials/plot>`, but for now let's just use ``matplotlib``'s ``hist`` function.

>>> import matplotlib.pyplot as plt
>>> plt.hist(s[:, 'FL1'], bins=100)
>>> plt.show()

.. image:: /_static/python_tutorial_transform_1.png

Note that the range of the x axis is from 0 to 1024. We know that this sample was acquired with a logarithmic amplifier, and fluorescence should cover values from 1 to 10000. However, data from an FCS file is normally stored in "channel units" or "channel numbers", which are essentially numbers as they come from the flow cytometer's detectors. To convert this data to arbitrary fluorescence units (a.u.), we use :func:`FlowCal.transform.exponentiate`.

>>> s_transformed = FlowCal.transform.exponentiate(s, channels='FL1')

``s_transformed`` now contains the same data as ``s``, except that the ``FL1`` channel has been transformed to a.u. Let's now look at the transformed data.

>>> import numpy as np
>>> bins = np.logspace(0, 4, 100)
>>> plt.hist(s_transformed[:, 'FL1'], bins=bins)
>>> plt.xscale('log')
>>> plt.show()

.. image:: /_static/python_tutorial_transform_2.png

We will explore a more convenient way to plot transformed data in the :doc:`plotting tutorial </python_tutorials/plot>`.

:func:`FlowCal.transform.exponentiate` can transform several channels at the same time. For example, to transform the forward scatter and side scatted channels, we can use the following.

>>> s_transformed = FlowCal.transform.exponentiate(s, channels=['FSC', 'SSC'])

Other Transformations
---------------------

``FlowCal`` includes the ability to transform flow cytometry data to Molecules of Equivalent Fluorophore (MEF), a unit independent of the acquisition settings. However, doing so is slightly more complex. For more information, consult the :doc:`MEF tutorial </python_tutorials/mef>`.