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
* Traditional transformation functions, such as exponentiation.
* Analysis of calibration beads data, standard curve generation, and transformation to absolute units (Molecules of Equivalent Fluorophore, MEF).
* Plotting, including generation of histograms, density plots and scatter plots.
* A user-fiendly Excel UI to gate, transform, calibrate, plot, and generate statistics from a list of flow cytometry samples in a simple fashion.

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
