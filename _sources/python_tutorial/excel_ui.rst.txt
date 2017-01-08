Processing FCS Files with the Excel UI
======================================

This tutorial focuses on how to obtain processed flow cytometry data from ``FlowCal``'s Excel UI into python. This document assumes that the reader is familiar with ``FlowCal``'s Excel UI. For more information, please refer to the :doc:`Excel UI documentation </excel_ui/index>`.

To start, navigate to the ``examples`` directory included with FlowCal, and open a ``python`` session therein. Then, import ``FlowCal`` as with any other python module.

>>> import FlowCal

Introduction
------------

``FlowCal`` is a very flexible package that allows the user to perform different gating and transformation operations on flow cytometry data. As we saw in the :doc:`MEF tutorial </python_tutorial/mef>`, the process of transformation to MEF units also allows for a lot of customization. However, for most experiments the user might simply want to follow a procedure similar to this:

1. Open calibration beads files
2. Perform density gating in forward/side scatter to eliminate bead aggregates
3. Obtain standard curves for each fluorescence channel of interest
4. Open cell sample files
5. Perform density gating in forward/side scatter to eliminate aggregates and non-cellular debris.
6. Transform the fluorescence of cell samples to MEF using the standard curves obtained in step 3.

After this, what follows is highly dependent on the type of experiment. Some might be interested, for example, in the geometric mean fluorescence and standard deviation of cell samples as a function of some inducer. For these cases, the Excel UI allows to easily specify a set of FCS files that will be processed as described above, and generate a set of statistics for each fluorescence channel of interest. This is performed through a convenient input Excel file, which can also document other information about the experiment, such as inducer level of each sample.

However, some applications demand more complicated downstream processing, such as n-dimensional fluorescence analysis, which will inevitably require programming. In these cases, one can still use ``FlowCal``'s Excel UI to process files as above, and return transformed and gated ``FCSData`` objects for each specified FCS file to python, along with extra information contained in the input Excel file. This workflow combines the convenience of maintaining experimental information in an Excel file, the consistency of a standard FCS file processing pipeline, and the power of performing numerical analysis in python. We will now describe how to do this.

Processing Samples with the Excel UI
------------------------------------

For this tutorial, we will analyze all the data in the ``examples/FCFiles`` folder using the input Excel file, ``examples/experiment.xlsx``. This is the same file described in the :doc:`Excel UI documentation </excel_ui/index>`.

First, load the necessary tables from this file.

>>> input_file = 'experiment.xlsx'
>>> instruments_table = FlowCal.excel_ui.read_table(input_file,
...                                                 sheetname='Instruments',
...                                                 index_col='ID')
>>> beads_table = FlowCal.excel_ui.read_table(input_file,
...                                           sheetname='Beads',
...                                           index_col='ID')
>>> samples_table = FlowCal.excel_ui.read_table(input_file,
...                                             sheetname='Samples',
...                                             index_col='ID')

:func:`FlowCal.excel_ui.read_table` returns the contents of a sheet from an Excel file as a ``pandas`` ``DataFrame``. The file name is specified as the first argument, and the ``index_col`` argument specified which column to use as the ``DataFrame``'s index. For more information about ``DataFrames``, consult `pandas' documentation <http://pandas.pydata.org/pandas-docs/stable/dsintro.html>`_.

From there, one can obtain the file name and analysis options of each beads file, and call all the necessary ``FlowCal`` functions to perform density gating and standard curve calculation. Or one could let the Excel UI do all that with the following instruction:

>>> beads_samples, mef_transform_fxns = FlowCal.excel_ui.process_beads_table(
...     beads_table,
...     instruments_table,
...     verbose=True,
...     plot=True)

``FlowCal.excel_ui.process_beads_table`` uses the instruments table and the beads table to automatically open, density-gate, and transform the specified beads files, and generate MEF transformation functions as indicated by the Excel input file. The flags ``verbose`` and ``plot`` instruct the function to generate messages for each file being processed, and plots for each step of standard curve calculation, similar to what we saw in the :doc:`MEF tutorial </python_tutorial/mef>`. The ouput arguments are ``beads_samples``, a list of transformed and gated FCSData objects, and ``mef_transform_fxns``, a dictionary of MEF transformation functions, indexed by the ID of the beads files.

In a similar way, ``FlowCal``'s Excel UI can automatically density-gate and transform cell samples using a single instruction:

>>> samples = FlowCal.excel_ui.process_samples_table(
...     samples_table,
...     instruments_table,
...     mef_transform_fxns=mef_transform_fxns,
...     verbose=True,
...     plot=True)

``FlowCal.excel_ui.process_samples_table`` uses the instruments and samples tables to open, density-gate, and transform cell samples as specified, and return the processed data as a list of FCSData objects. If the input Excel file specifies that some samples should be transformed to MEF, ``FlowCal.excel_ui.process_samples_table`` also requires a dictionary with the respective MEF transformation functions (``mef_transform_fxns``), which was provided in the previous step by ``FlowCal.excel_ui.process_beads_table``.

**This is all the code required to obtain a set of processed cell samples**. From here, one can perform any desired analysis on ``samples``. Note that ``samples_table`` contains any other information in the input Excel file not directly used by ``FlowCal``, such as inducer concentration, incubation time, etc. This can be used to build an induction curve, fluorescence vs. final optical density (OD), etc.
