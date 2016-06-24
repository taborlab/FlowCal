Plotting Flow Cytometry Data
============================

This tutorial focuses on how to plot flow cytometry data using ``FlowCal``, particularly by using the module :mod:`FlowCal.plot`

To start, navigate to the ``examples`` directory included with FlowCal, and open a ``python`` session therein. Then, import ``FlowCal`` as with any other python module.

>>> import FlowCal

Also, import ``pyplot`` from ``matplotlib``

>>> import matplotlib.pyplot as plt

Histograms
----------

Let's load the data from file ``Data001.fcs`` into an ``FCSData`` object called ``s``, and tranform all channels to arbitrary units.

>>> s = FlowCal.io.FCSData('FCFiles/Data001.fcs')
>>> s = FlowCal.transform.to_rfi(s)

One is often interested in the fluorescence distribution across a population of cells. This is represented in a histogram. Since ``FCSData`` is a numpy array, one could use the standard ``hist`` function included in matplotlib. Alternatively, ``FlowCal`` includes its own histogram function specifically tailored to work with ``FCSData`` objects. For example, one can plot the contents of the ``FL1`` channel with a single call to :func:`FlowCal.plot.hist1d`.

>>> FlowCal.plot.hist1d(s, channel='FL1')
>>> plt.show()

.. image:: https://www.dropbox.com/s/lqz6blikfcmx2ub/python_tutorial_plot_hist1d_1.png?raw=1

:func:`FlowCal.plot.hist1d` behaves mostly like a regular matplotlib plotting function: it will plot in the current figure and axis. The axes labels are populated by default, but one can still use ``plt.xlabel`` and ``plt.ylabel`` to change them.

By default, :func:`FlowCal.plot.hist1d` uses something called *logicle* scaling for the x axis. This scaling allows visualization of high fluorescence values with logarithmic spacing, and low fluorescence values with a more linear spacing. In some modern flow cytometers, negative events may be present, and logicle scaling allows visualization of those as well. This can be changed to a more conventional linear or logarithmic scale by using the ``xscale`` argument. In addition, :func:`FlowCal.plot.hist1d` uses 256 uniformly spaced bins by default. We can override the default bins using the ``bins`` argument. Let's try using 1024 logarithmically-spaced bins.

>>> FlowCal.plot.hist1d(s, channel='FL1', xscale='log', bins=1024)
>>> plt.show()

.. image:: https://www.dropbox.com/s/nvlp43qz85th4al/python_tutorial_plot_hist1d_2.png?raw=1

Finally, :func:`FlowCal.plot.hist1d` can plot several FCSData objects at the same time. Let's now load 3 FCSData objects, transform all channels to a.u., and plot the ``FL1`` channel of all three with transparency.

>>> filenames = ['FCFiles/Data{:03d}.fcs'.format(i + 2) for i in range(3)]
>>> d = [FlowCal.io.FCSData(filename) for filename in filenames]
>>> d = [FlowCal.transform.to_rfi(di) for di in d]
>>> FlowCal.plot.hist1d(d, channel='FL1', alpha=0.7, bins=128)
>>> plt.legend(filenames, loc='upper left')
>>> plt.show()

.. image:: https://www.dropbox.com/s/3q6htlqrr28mw6h/python_tutorial_plot_hist1d_3.png?raw=1

Density Plots
-------------

It is also common to look at the forward scatter and side scatter values in a 2D histogram, scatter plot, or density diagram. From those, the user can extract size and shape information that would allow him to differentiate between cells and debris. ``FlowCal`` includes the function :func:`FlowCal.plot.density2d` for this purpose.

Let's look at the ``FSC`` and ``SSC`` channels in our sample ``s``.

>>> FlowCal.plot.density2d(s, channels=['FSC', 'SSC'])
>>> plt.show()

.. image:: https://www.dropbox.com/s/l041acoupnlb5os/python_tutorial_plot_density_2d_1.png?raw=1

The color indicates the number of events in the region, with red indicating a bigger number than yellow and blue, in that order, by default. Similarly to :func:`FlowCal.plot.hist1d`, :func:`FlowCal.plot.density2d` uses logicle scaling by default. In addition, :func:`FlowCal.plot.density2d` applies, by default, gaussian smoothing to the density plot.

:func:`FlowCal.plot.density2d` includes two visualization modes: ``mesh`` (seen above), and ``scatter``. The last one is good for distinguishing regions with few events.

>>> FlowCal.plot.density2d(s, channels=['FSC', 'SSC'], mode='scatter')
>>> plt.show()

.. image:: https://www.dropbox.com/s/j2fe7f7drib5nvs/python_tutorial_plot_density_2d_2.png?raw=1

The last plot shows three distinct populations. The one in the middle corresponds to cells, whereas the ones at the left and below correspond to non-biological debris. We will see how to "gate", or select only one population, in the :doc:`gating tutorial </python_tutorial/gate>`.

Combined Histogram and Density Plots
------------------------------------

FlowCal also includes "complex plot" functions, which produce their own figure and a set of axes, and use simple ``matplotlib`` or ``FlowCal`` plotting functions to populate them.

In particular, :func:`FlowCal.plot.density_and_hist` uses :func:`FlowCal.plot.hist1d` and :func:`FlowCal.plot.density2d` to produce a combined density plot/histogram that allow the user to quickly see information about one sample. For example, let's plot the ``FSC`` and ``SSC`` channels in a density plot, and the ``FL1`` channel in a histogram. In the following, ``density_params`` and ``hist_params`` are dictionaries that are directly passed to :func:`FlowCal.plot.hist1d` and :func:`FlowCal.plot.density2d` as keyword arguments.

>>> FlowCal.plot.density_and_hist(s,
...                               density_channels=['FSC', 'SSC'],
...                               density_params={'mode':'scatter'},
...                               hist_channels=['FL1'])
>>> plt.tight_layout()
>>> plt.show()

.. image:: https://www.dropbox.com/s/apb0ep5xp1idnht/python_tutorial_plot_density_and_hist_1.png?raw=1

:func:`FlowCal.plot.density_and_hist` can also plot data before and after applying gates. We will see this in the :doc:`gating tutorial </python_tutorial/gate>`.

Other Plotting Functions
------------------------
These are not the only functions in :mod:`FlowCal.plot`. For more information, consult the API reference.
