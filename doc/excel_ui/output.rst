Outputs of the Excel UI
=======================

During processing of the calibration beads and cell samples, ``FlowCal`` creates two folders and an output Excel file in the same location as the input Excel file. Below, each output file is described, and an image illustrating the file contents is provided. <ID> refers to the value specified in the ID column of the Excel file.

1. The folder ``plot_beads`` contains plots of the individual steps of processing of the calibration particle samples:

    a. ``density_hist_<ID>.png``: A FSC/SSC 2D density diagram of the calibration particle sample, and a histogram for each relevant fluorescence channel.

    .. image:: /_static/output_beads_density.png

    b. ``clustering_<ID>.png``: A plot of the sub-populations identified during the clustering step, where the different sub-populations are shown in different colors. Depending on the number of channels used for clustering, this plot is a histogram (when using only one channel), a 2D scatter plot (when using two channels), or a 3D scatter plot with three 2D projections (when using three channels or more). If the populations have been identified incorrectly, changing the number of channels used for clustering, or the density gate fraction can improve the results. These two parameters can be changed in the **Beads** sheet of the input Excel file.

    .. image:: /_static/output_beads_clustering.png

    c. ``populations_<channel>_<ID>.png``: A histogram showing the identified microbead sub-populations in different colors, for each fluorescence channel in which a MEF standard curve is to be calculated. In addition, a vertical line is shown representing the median of each population, which is later used to calculate the standard curve. Sub-populations that were not used to generate the standard curve are shown in gray.

    .. image:: /_static/output_beads_populations.png

    d. ``std_crv_<channel>_<ID>.png``: A plot of the fitted standard curve, for each channel in which MEF values were specified.

    .. image:: /_static/output_beads_sc.png

2. The folder ``plot_samples`` contains plots of the experimental cell samples. Each experimental sample of name “ID” as specified in the Excel input sheet results in a file named ``<ID>.png``. This image contains a FSC/SSC 2D density diagram with the gated region indicated, and a histogram for each user-specified fluorescence channel.

.. image:: /_static/output_sample.png

3. The file ``<Name of the input Excel file>_output.xlsx``, contains calculated statistics for each sample, for each relevant fluorescence channel. To produce this file, ``FlowCal`` copies the **Instruments**, **Beads**, and **Samples** sheets from the input Excel file, unmodified, to the output file, and adds columns to the **Samples** sheet with statistics. Statistics per sample include: the number of events after gating and the acquisition time. Statistics per fluorescence channel include: channel gain, mean, geometric mean, median, mode, standard deviation, coefficient of variation (CV), interquartile range (IQR), and robust coefficient of variation (RCV).

.. image:: /_static/output_spreadsheet_samples.png

In addition, a **Histograms** tab is generated, with bin/counts pairs for each sample and relevant fluorescence channel, in the specified units.

.. image:: /_static/output_spreadsheet_histograms.png
