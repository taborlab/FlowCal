Calibrating Flow Cytometry Data to MEF
======================================

This tutorial focuses on how to transform flow cytometry data to Molecules of Equivalent Fluorophore (MEF) using ``FlowCal``, particularly by using the module :mod:`FlowCal.mef`. For more information on MEF calibration, see the section on :doc:`fundamentals of calibration</fundamentals/calibration>`.

To start, navigate to the ``examples/FCFiles`` directory included with FlowCal, and open a ``python`` session therein. Then, import ``FlowCal`` as with any other python module.

>>> import FlowCal

Also, import ``pyplot`` from ``matplotlib``

>>> import matplotlib.pyplot as plt

Working with Calibration Beads
------------------------------

As mentioned in the :doc:`fundamentals</fundamentals/calibration>` section, conversion to MEF requires measuring calibration beads. ``data_006.fcs`` in the ``examples`` folder contains beads data. Let's examine it.

>>> b = FlowCal.io.FCSData('data_006.fcs')
>>> b_g, __, c = FlowCal.gate.density2d(b,
...                                     channels=['FSC', 'SSC'],
...                                     gate_fraction=0.3,
...                                     full_output=True)
>>> FlowCal.plot.density_and_hist(b,
...                               gated_data=b_g,
...                               gate_contour=c,
...                               density_channels=['FSC', 'SSC'],
...                               density_params={'mode':'scatter'},
...                               hist_channels=['FL1'])
>>> plt.tight_layout()
>>> plt.show()

.. image:: https://www.dropbox.com/s/zfgt4xtvnvy07sv/python_tutorial_mef_1.png?raw=1

The ``FSC``/``SSC`` density plot shows two groups of events: the dense group in the middle-left corresponds to single beads, whereas the fainter cluster on the middle right corresponds to bead agglomerations. Only single beads should be used, so :func:`FlowCal.gate.density2d` is used here to identify single beads automatically. Looking at the ``FL1`` histogram, we can clearly distinguish 8 subpopulations with different fluorescence levels. Notice that the group with the highest fluorescence seems to have saturated fluorescence values.

MEF Transformation in ``FlowCal``
---------------------------------

We saw in the :doc:`transformation tutorial </python_tutorial/transform>` that a transformation function is needed to convert flow cytometry data from raw sensor numbers, as stored in FCS files, to fluorescence values in a.u. Similarly, ``FlowCal`` uses transformation functions to convert raw flow cytometry data to MEF. However, these functions have to be generated during analysis using a calibration bead sample. Once a function is generated, though, it can be used to convert many cell samples to MEF, provided that beads and samples have been acquired using the same settings.

Generating a transformation function from calibration beads data is a complicated process, therefore ``FlowCal`` has an entire module dedicated to it: :mod:`FlowCal.mef`. The main function in this module, :func:`FlowCal.mef.get_transform_fxn`, requires at least the following information: calibration beads data, the names of the channels for which a MEF transformation function should be generated, and manufacturer-provided MEF values of each subpopulation for each channel. Let's now use :func:`FlowCal.mef.get_transform_fxn` to obtain a transformation function.

>>> # Obtain transformation function
>>> # The following MEFL values were provided by the beads' manufacturer
>>> mef_values_fl1 = [0, 792, 2079, 6588, 16471, 47497, 137049, 271647]
>>> to_mef = FlowCal.mef.get_transform_fxn(b_g, 
...                                        mef_values=mef_values_fl1,
...                                        mef_channels='FL1',
...                                        plot=True)
>>> plt.show()

The argument ``plot`` instructs :func:`FlowCal.mef.get_transform_fxn` to generate and save plots showing the individual steps of bead data analysis. We will look at these plots and how to interpret them in the next section. We recommend to always generate these plots to confirm that the standard curve was generated properly.

Let's now use ``to_mef`` to transform fluroescence data to MEF.

>>> # Load sample
>>> s = FlowCal.io.FCSData('data_001.fcs')
>>>
>>> # Transform each channel as appropriate
>>> s_t = FlowCal.transform.to_rfi(s, channels=['FSC', 'SSC'])
>>> s_t = to_mef(s_t, channels='FL1')
>>>
>>> # Gate
>>> s_g = FlowCal.gate.high_low(s_t, channels=['FSC', 'SSC'])
>>> s_g = FlowCal.gate.density2d(s_g,
...                              channels=['FSC', 'SSC'],
...                              gate_fraction=0.5)
>>>
>>> # Plot histogram of transformed channel
>>> FlowCal.plot.hist1d(s_g, channel='FL1', log=True)
>>> plt.show()

.. image:: https://www.dropbox.com/s/lis33tp84cyjmy7/python_tutorial_mef_2.png?raw=1

``s_g`` now contains ``FL1`` fluorescence values in MEF units. Note that the values in the x axis of the histogram do not match the ones showed before in channel (raw) units or a.u.. This is always true in general. However, the range of approximately 4 decades is maintained.

Generation of a MEF Transformation Function
-------------------------------------------

We will now give a short description of the process that :func:`FlowCal.mef.get_transform_fxn` uses to generate a transformation function from beads data. We will also examine the plots produced by :func:`FlowCal.mef.get_transform_fxn` and discuss how these plots can reveal problems with the analysis. In the following, ``<beads_filename>`` refers to the file name of the FSC cotaining beads data, which was provided to :func:`FlowCal.mef.get_transform_fxn`. This discussion is parallel to the one in the :doc:`fundamentals of calibration</fundamentals/calibration>` document, but at a higher technical level.

Generating a MEF transformation function involves four steps:

1. Identification of Bead Subpopulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FlowCal`` uses a clustering algorithm to automatically identify the different subpopulations of beads. The algorithm will try to find as many populations as values are provided in ``mef_values``.

A plot with a default filename of ``clustering_<beads_filename>.png`` is generated by :func:`FlowCal.mef.get_transform_fxn` after the completion of this step. This plot is a histogram or scatter plot in which different subpopulations are shown in a different colors. Such plot is shown below, for ``data_006.fcs``.

.. image:: https://www.dropbox.com/s/hldzq9hqi0ky4ft/python_tutorial_mef_3.png?raw=1

It is always visually clear which events correspond to which groups, and the different colors should correspond to this expectation. If they don't, sometimes it helps to use a different set of fluorescence channels for clustering (see below), or to use a different gating fraction in the previous density gating step.

The default clustering algorithm is Gaussian Mixture Models, implemented in :func:`FlowCal.mef.clustering_gmm`. However, a function implementing another clustering algorithm can be provided to :func:`FlowCal.mef.get_transform_fxn` through the argument ``clustering_fxn``. In addition, the argument ``clustering_channels`` specifies which channels to use for clustering. This can be different than ``mef_channels``, the channels for which to generate a standard curve. A plot resulting from clustering with three fluroescence channels is shown below.

.. image:: https://www.dropbox.com/s/5b6ux2o9irmip4h/python_tutorial_mef_4.png?raw=1

2. Calculation of Population Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each channel in ``mef_channels``, a representative fluorescence value in channel units is calculated for each subpopulation. By default, the median of each population is used, but this can be customized using the ``statistic_fxn`` parameter.

3. Population Selection
~~~~~~~~~~~~~~~~~~~~~~~

For each channel in ``mef_channels``, subpopulations close to saturation are discarded.

A plot with a default filename of ``populations_<channel>_<beads_filename>.png`` is generated by :func:`FlowCal.mef.get_transform_fxn` for each channel in ``mef_channels`` after the completion of this step. This plot is a histogram showing each population, as identified in step one, with vertical lines showing their representative statistic as calculated from step 2, and with the discarded populations colored in grey. Such plot is shown below, for ``data_006.fcs`` and channel ``FL1``.

.. image:: https://www.dropbox.com/s/807o6k8vaowvacd/python_tutorial_mef_5.png?raw=1

By default, populations whose mean is closer than a few standard deviations from one of the edge values are discarded. This is encoded in the function :func:`FlowCal.mef.selection_std`. A different method can be used by providing a different function to :func:`FlowCal.mef.get_transform_fxn` through the argument ``selection_fxn``. This argument can even be None, in which case no populations are discarded. Finally, one can manually discard a population by using ``None`` as its MEF fluorescence value in ``mef_values``. Discarding populations specified in this way is performed in addition to ``selection_fxn``.

4. Standard Curve Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A bead fluorescence model is fitted to the fluorescence values of each subpopulation in channel units, as calculated in step 2, and in MEF units, as provided in ``mef_values``. A standard curve can then be calculated from the bead fluorescence model.

A plot with a default filename of ``std_crv_<channel>_<beads_filename>.png`` is generated by :func:`FlowCal.mef.get_transform_fxn` for each channel in ``mef_channels`` after the completion of this step. This plot shows the fluorescence values of each population in channel units and MEF, the fitted bead fluorescence model, and the resulting standard curve. Such plot is shown below, for ``data_006.fcs`` and channel ``FL1``.

.. image:: https://www.dropbox.com/s/guaqbclf0t69d4k/python_tutorial_mef_6.png?raw=1

It is worth noting that the bead fluorescence model and the standard curve are different, in that bead fluorescence is also affected by bead autofluorescence, its fluorescence when no fluorophore is present. To obtain the standard curve, autofluorescence is eliminated from the model. Such a model is fitted in :func:`FlowCal.mef.fit_beads_autofluorescence`, but a different model can be provided to :func:`FlowCal.mef.get_transform_fxn` using the argument ``fitting_fxn``.

After these steps, a transformation function is generated using the standard curve, and returned.

:func:`FlowCal.mef.get_transform_fxn` has more customization options. For more information, consult the reference.