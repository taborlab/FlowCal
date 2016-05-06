Outputs of the Excel UI
=======================

During processing of the calibration beads and cell samples, ``FlowCal`` creates two folders with images and an output Excel file in the same location as the :doc:`input Excel file<input_format>`. Here we describe these. In what follows, <ID> refers to the value specified in the ID column of the input Excel file.

.. _excel-ui-outputs-plots:

Plots
-----

1. The folder ``plot_beads`` contains plots of the individual steps of processing of the calibration particle samples:

    a. ``density_hist_<ID>.png``: A forward/side scatter 2D density diagram of the calibration particle sample, and a histogram for each relevant fluorescence channel.

    .. image:: https://www.dropbox.com/s/2o1oej5p4sn30x9/output_beads_density.png?raw=1

    b. ``clustering_<ID>.png``: A plot of the sub-populations identified during the clustering step, where the different sub-populations are shown in different colors. Depending on the number of channels used for clustering, this plot is a histogram (when using only one channel), a 2D scatter plot (when using two channels), or a 3D scatter plot with three 2D projections (when using three channels or more). If the populations have been identified incorrectly, changing the number of channels used for clustering or the density gate fraction can improve the results. These two parameters can be changed in the **Beads** sheet of the input Excel file.

    .. image:: https://www.dropbox.com/s/nzxtmwzto3ab1iv/output_beads_clustering.png?raw=1

    c. ``populations_<channel>_<ID>.png``: A histogram showing the identified microbead sub-populations in different colors, for each fluorescence channel in which a MEF standard curve is to be calculated. In addition, a vertical line is shown representing the median of each population, which is later used to calculate the standard curve. Sub-populations that were not used to generate the standard curve are shown in gray.

    .. image:: https://www.dropbox.com/s/bz14p0zfijvrqne/output_beads_populations.png?raw=1

    d. ``std_crv_<channel>_<ID>.png``: A plot of the fitted standard curve, for each channel in which MEF values were specified.

    .. image:: https://www.dropbox.com/s/slst4exz1sp0h9x/output_beads_sc.png?raw=1

2. The folder ``plot_samples`` contains plots of the experimental cell samples. Each experimental sample of name “ID” as specified in the Excel input sheet results in a file named ``<ID>.png``. This image contains a forward/side scatter 2D density diagram with the gated region indicated, and a histogram for each user-specified fluorescence channel.

.. image:: https://www.dropbox.com/s/ieew8wk88zcq91z/output_sample.png?raw=1

.. _excel-ui-outputs-excel:

Output Excel File
-----------------

The file ``<Name of the input Excel file>_output.xlsx`` contains calculated statistics for beads and samples. To produce this file, ``FlowCal`` copies the **Instruments**, **Beads**, and **Samples** sheets from the :doc:`input Excel<input_format>` file, unmodified, to the output file, and adds columns to the **Beads** and **Samples** sheet with statistics.

In both sheets, the number of events after gating and the acquisition time are reported for each sample. In addition, a column named **Analysis Notes** indicates the user about any errors that occurred during processing.

Statistics per beads file, per fluorescence channel include: the channel gain, the amplifier type, the equation of the beads fluorescence model used, and the values of the fitted parameters

.. image:: https://www.dropbox.com/s/uar7zo73crvu0je/output_spreadsheet_beads.png?raw=1

Statistics per cell sample, per fluorescence channel include: channel gain, mean, geometric mean, median, mode, arithmetic and geometric standard deviation, arithmetic and geometric coefficient of variation (CV), interquartile range (IQR), and robust coefficient of variation (RCV). Note that if an error has been found, the **Analysis Notes** field will be populated, and statistics and plots will not be reported.

.. image:: https://www.dropbox.com/s/xpowct9dig8acgh/output_spreadsheet_samples.png?raw=1

In addition, a **Histograms** tab is generated, with bin/counts pairs for each sample and relevant fluorescence channel in the specified units.

.. image:: https://www.dropbox.com/s/o62wj53w75b7jeh/output_spreadsheet_histograms.png?raw=1

One last tab named **About Analysis** is added with information about the corresponding input Excel file, the date and time of the run, and the FlowCal version used.

.. image:: https://www.dropbox.com/s/q8pu4s2he3jjwp2/output_spreadsheet_about.png?raw=1