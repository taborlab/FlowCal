Density Gating
==============

Description
-----------

Density gating looks at two channels of flow cytometry data, and discards events that are clearly different from the main population in the sample. Density gating is applied mostly to the forward/side scatter channels in ``FlowCal``. When doing this, single microbeads or cells can be separated from aggregates and non-bead or non-biological debris, even when these events are a substantial fraction of the total count.

In the figure below, a sample was acquired with an intentionally low side-scatter threshold to allow a significant number of events corresponding to non-biological debris. Density gating was then applied to retain 50% of the events in the densest region. Because cells have a more uniform size than the observed debris, density gating retains mostly cells, which is reflected in the fact that FL1 fluorescence is bimodal before gating, but not after.

.. image:: https://www.dropbox.com/s/rz2cvv0vug4ws7g/fundamentals_density_1.png?raw=1

.. note:: The sample shown above was intentionally acquired with a low threshold value in ``SSC`` to show the capabilities of density gating. Normally, a lot of the debris can be eliminated by simply selecting a higher ``SSC`` threshold. However, density gating is still an excellent method to clean the data and eliminate all the debris that a simple threshold cannot filter. In our experience, this can still be a significant fraction of the total event count, especially if the cell culture has low density.

Algorithm
---------

Density gating is implemented in the function :func:`FlowCal.gate.density2d()`. In short, this function:

1. Determines the number of events to keep, based on the user specified gating fraction and the total number of events of the input sample.
2. Divides the 2D channel space into a rectangular grid, and counts the number of events falling within each bin of the grid. The number of counts per bin across all bins comprises a 2D histogram, which is a coarse approximation of the underlying probability density function.
3. Smoothes the histogram generated in Step 2 by applying a Gaussian Blur. Theoretically, the proper amount of smoothing results in a better estimate of the probability density function. Practically, smoothing eliminates isolated bins with high counts, most likely corresponding to noise, and smoothes the contour of the gated region.
4. Selects the bins with the greatest number of events in the smoothed histogram, starting with the highest and proceeding downward until the desired number of events to keep, calculated in step 1, is achieved.
5. Returns the gated event list.