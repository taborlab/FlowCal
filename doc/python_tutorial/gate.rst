Gating Flow Cytometry Data
==========================

This tutorial focuses on how to gate flow cytometry data using ``FlowCal``, particularly by using the module :mod:`FlowCal.gate`. Gating is the process of retaining events that satisfy some criteria, and discarding the ones that do not.

To start, navigate to the ``examples/FCFiles`` directory included with FlowCal, and open a ``python`` session therein. Then, import ``FlowCal`` as with any other python module.

>>> import FlowCal

Also, import ``numpy`` and ``pyplot`` from ``matplotlib``

>>> import numpy as np
>>> import matplotlib.pyplot as plt

Removing Saturated Events
-------------------------

We'll start by loading the data from file ``data_001.fcs`` into an ``FCSData`` object called ``s``. Then, transform channels ``FSC``, ``SSC``, and ``FL1`` into a.u.

>>> s = FlowCal.io.FCSData('data_001.fcs')
>>> s_t = FlowCal.transform.to_rfi(s, channels=['FSC', 'SSC', 'FL1'])

In the :doc:`plotting tutorial </python_tutorial/plot>` we looked at a density plot of the forward scatter/side scatter (``FSC``/``SSC``) channels and identified several clusters of events. This density plot is repeated below for convenience.

.. image:: https://www.dropbox.com/s/rq9id6rmp57hoe1/python_tutorial_plot_5.png?raw=1

From these clusters, the faint one in the middle corresponds to non-cellular debris, and the one above corresponds to cells. Two additional major groups on the left correspond to saturated events, with the lowest possible forward scatter value: 1 a.u..

Some flow cytometers will capture events outside of their range and assign them either the lowest or highest possible values of a channel, depending on which side of the range they fall on. We call these events "saturated". Including them in the analysis results, most of the time, in distorted distribution shapes and incorrect statistics. Therefore, it is generally advised to remove saturated events. To do so, ``FlowCal`` incorporates the function :func:`FlowCal.gate.high_low`. This function retains all the events in the specified channels between two specified values: a high and a low threshold. If these values are not specified, however, the function uses the saturating values.

>>> s_g1 = FlowCal.gate.high_low(s_t, channels=['FSC', 'SSC'])
>>> FlowCal.plot.density2d(s_g1, channels=['FSC', 'SSC'], xlog=True, ylog=True)
>>> plt.show()

.. image:: https://www.dropbox.com/s/pujabnm5qmsa4pc/python_tutorial_gate_1.png?raw=1

We successfully removed the event clusters on the left. We can go one step further and use :func:`FlowCal.gate.high_low` again to remove the cluster in the lower middle of the plot, which as we said before corresponds to debris.

>>> s_g2 = FlowCal.gate.high_low(s_g1, channels='SSC', low=220)
>>> FlowCal.plot.density2d(s_g2,
...                        channels=['FSC', 'SSC'],
...                        xlog=True,
...                        ylog=True,
...                        mode='scatter')
>>> plt.show()

.. image:: https://www.dropbox.com/s/yqc4kfmiysf2kmd/python_tutorial_gate_2.png?raw=1

This approach, however, requires one to estimate a low threshold value for every sample manually. In addition, we usually want events in the densest forward scatter/side scatter region, which requires a more complex shape than a pair of thresholds. We will now explore better ways to achieve this.

Ellipse Gate
------------

``FlowCal`` includes an ellipse-shaped gate, in which events are retained if they fall inside an ellipse with a specified center and dimensions. Let's try to obtain the densest region of the cell cluster.

>>> s_g3 = FlowCal.gate.ellipse(s_g1,
...                             channels=['FSC', 'SSC'],
...                             log=True,
...                             center=(1.9, 2.7),
...                             a=0.2,
...                             b=0.15,
...                             theta=15/180.*np.pi)
>>> FlowCal.plot.density2d(s_g3,
...                        channels=['FSC', 'SSC'],
...                        xlog=True,
...                        ylog=True,
...                        mode='scatter')
>>> plt.show()

.. image:: https://www.dropbox.com/s/triklae3jjth89g/python_tutorial_gate_3.png?raw=1

As shown above, the remaining events reside only inside an ellipse-shaped region. Note that we used the argument ``log``, which indicates that the gated region should look like an ellipse in a logarithmic plot. This also requires that the center and the major and minor axes (``a`` and ``b``) be specified in log space.

The disadvantage of this gate is that several parameters need to be specified, which make the resulting gate arbitrary. In addition, it is questionable whether we're actually capturing the densest part of the distribution. Using the mean or median as centers results in similar issues because the distribution is not symmetrical. The next gate solves these issues.

Density Gate
------------
:func:`FlowCal.gate.density2d` automatically identifies the region with the highest density of events in a two-dimensional diagram, and calculates how big it should be to capture a certain percentage of the total event count. One effect is that the number of user-defined parameters is reduced to one. Let's now try to separate cells from debris using this method.

>>> s_g4 = FlowCal.gate.density2d(s_g1,
...                               channels=['FSC', 'SSC'],
...                               gate_fraction=0.5)
>>> FlowCal.plot.density2d(s_g4, channels=['FSC', 'SSC'], log=True, mode='scatter')
>>> plt.show()

.. image:: https://www.dropbox.com/s/34079nzcgs4xxzv/python_tutorial_gate_4.png?raw=1

We can see that :func:`FlowCal.gate.density2d` automatically identified the region that contains cells, and defined a shape that more closely resembles what the ungated density map looks like. The parameter ``gating_fraction`` allows the user to control the fraction of events to retain, and it is the only parameter that the user is required to specify.

For more details on how :func:`FlowCal.gate.density2d` works, consult the section on :doc:`fundamentals of density gating</fundamentals/density_gate>`.

Plotting 2D Gates
-----------------

Finally, we will see a better way to visualize the result of applying a 2D gate. First, we will use density gating again, but this time we will do it a little differently.

>>> s_g5, m_g5, contour = FlowCal.gate.density2d(s_g1,
...                                              channels=['FSC', 'SSC'],
...                                              xlog=True,
...                                              ylog=True,
...                                              gate_fraction=0.5,
...                                              full_output=True)

The extra argument, ``full_output``, is available in every function in :mod:`FlowCal.gate`. It instructs a gating function to return some additional output arguments with information about the gating process. The second output argument is always a mask, a boolean array that indicates which events on the original FCSData object are being retained by the gate. 2-dimensional gating functions have a third output argument: a contour surrounding the gated region, which we will now use for plotting.

The function :func:`FlowCal.plot.density_and_hist` was introduced in the :doc:`plotting tutorial </python_tutorial/plot>` to produce plots of a single FCSData object. But it can also be used to plot the result of a gating step, showing the data before and after gating, and the gating contour. Let's use this ability to show the result of the density gating process.

>>> FlowCal.plot.density_and_hist(s_g1,
...                               gated_data=s_g5,
...                               gate_contour=contour,
...                               density_channels=['FSC', 'SSC'],
...                               density_params={'xlog':True,
...                                               'ylog':True,
...                                               'mode':'scatter'},
...                               hist_channels=['FL1'],
...                               hist_params={'log':True})
>>> plt.tight_layout()
>>> plt.show()

.. image:: https://www.dropbox.com/s/4hm191bfivdt2nt/python_tutorial_gate_5.png?raw=1

We can now observe the gating contour right on top of the ungated data, and see which events were kept and which ones were left out. In addition, we can visualize how gating affected the other channels. In particular, observe that bimodality in the ``FL1`` fluorescence channel disappeared with gating. This shows that the observed bimodality was produced by the difference in fluorescence between debris and cells, but that cells in this sample are unimodal.

.. note:: ``data_001.fcs`` was intentionally acquired with a low threshold value in ``SSC`` to show the capabilities of density gating. Normally, a lot of the debris can be eliminated by simply selecting a higher ``SSC`` threshold. However, density gating is still an excellent method to clean the data and eliminate all the debris that a simple threshold cannot filter. In our experience, this can still be a significant fraction of the total event count, especially if the cell culture has low density.