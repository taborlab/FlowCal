.. FlowCal documentation master file, created by Sebastian Castillo-Hair
   sphinx-quickstart on Mon Dec  7 10:49:16 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======
FlowCal
=======
``FlowCal`` is a library for processing, calibrating, and analyzing flow cytometry data in Python.
It features:

* Extraction of Flow Cytometry Standard (FCS) files into numpy array-like structures
* Traditional and non-standard gating, including automatic density-based two-dimensional gating.
* Transformation functions that allow conversion of data from raw FCS numbers to arbitrary fluorescence units (a.u.).
* Plotting, including generation of histograms, density plots and scatter plots.

Most importantly, ``FlowCal`` automatically analyzes calibration beads data in order to convert the fluorescence of cell samples into calibrated units, **Molecules of Equivalent Fluorophore (MEF)**. The most important advantages of using MEF are 1) Fluorescence can be reported independently of acquisition settings, and 2) Variation in data due to instrument shift is eliminated.

Finally, ``FlowCal`` includes a user-fiendly Excel User Interface to perform all of these operations automatically, without the need to write any code.


Table of Contents
=================

.. toctree::
   :maxdepth: 2

   getting_started/index.rst
   excel_ui/index.rst
   python_tutorial/index.rst
   reference/modules.rst
   theory/index.rst
   contribute/index.rst


.. Getting Started
.. ---------------

.. .. toctree::
..    :maxdepth: 1

..    Download FlowCal! <https://github.com/taborlab/FlowCal/archive/master.zip>
..    installation.rst
