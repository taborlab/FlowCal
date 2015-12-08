===========================================
Flow Cytometry (``FlowCal``) Python Library
===========================================
``FlowCal`` is a library for processing and analyzing flow cytometry data.
It features:

* Data extraction from FCS files into numpy array-like structures
* Non-standard gating, including density-based two-dimensional gating.
* Automated analysis of calibration beads data, and standard curve generation.
* Data transformation, including exponentiation and conversion to absolute units (Molecules of Equivalent Fluorophore, MEF).
* Plotting, including generation of histograms, density plots and scatter plots.
* A user-fiendly Excel UI to automatically gate, transform, plot, and generate statistics from flow cytometry samples.

Installation
============

Pre-requisites
--------------
Python 2.7 is required. The following Python packages should also be present:

* ``numpy`` (1.9.2)
* ``scipy`` (0.15.1)
* ``matplotlib`` (1.4.3)
* ``palettable`` (2.1.1)
* ``scikit-learn`` (0.16.1)
* ``pandas`` (0.16.2)
* ``xlrd`` (0.9.3)
* ``XlsxWriter`` (0.7.3)

If ``FlowCal`` is installed through ``pip`` or ``setuptools``, prerequisites should be taken care of automatically.

OS-specific recommendations are given below.

Windows and MacOSX
~~~~~~~~~~~~~~~~~~
We recommend the Anaconda Python distribution from Continuum Analytics (https://store.continuum.io/cshop/anaconda/). Anaconda simplifies the installation of python and the required packages.

To install ``FlowCal``, download the repo and double click on ``Install FlowCal (Windows).bat``
or ``Install FlowCal (OSX)`` on Windows and Linux, respectively.

Ubuntu/Mint
~~~~~~~~~~~
Python is installed by default in Ubuntu and Linux Mint. We recommend managing packages through `pip` to get the appropriate versions.

To install ``FlowCal``, download the repo and run:

``python setup.py install``

You may need to request administrative permissions by using the ``sudo`` command.

The Excel UI
============
The Excel UI simplifies the analysis of flow cytometry samples to be gated and calibrated using calibration beads. The samples to be processed are specified in an input Excel file. Refer to the "Examples" section for more information on the format of this file.

To run the Excel UI, double click on ``Run FlowCal (Windows).bat`` or ``Run FlowCal (OSX)`` on Windows or OSX, respectively, and select the appropriate input Excel file.

Examples
========
The ``examples`` directory contains simple ready-to-run scripts that can be used as a starting point for flow cytometry data analysis. These files can be run as regular python scripts. The scripts included are:

* ``analysis_no_beads.py`` - This script will read data in the ``FCFiles`` folder, perform exponential transformation in all channels, and produce FSC/SSC scatter plots and fluorescence histograms in the ``plot_gated`` directory.
* ``analysis.py`` - This script will read data in the ``FCFiles`` folder, process the provided calibration beads data file, generate a calibration curve, process all the sample data files, transform fluorescence values to absolute (MEF) units, and produce plots.

In addition, ``experiment.xlsx`` is an example of an input file that specifies samples to be analyzed and calibrated by ``FlowCal``'s Excel UI.

Report Bugs
===========
The official way to report a bug is through the issue tracker on github (https://github.com/taborlab/FlowCal/issues). Try to be as explicit as possible when describing your issue. Ideally, a set of instructions to reproduce the error should be provided, together with the version of all the relevant packages you are using.

Request Features
================
Features can also be requested through the issue tracker on github. Try to be as descriptive as possible about the desired feature, and indicate whether you would be willing to write the code necessary to implement it.

Contributing
============
We can use some help! If you are interested in writing code for this project, either for adding some new feature or correcting a bug, please check the DEVELOPING.rst file.
