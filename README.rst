======================================
Flow Cytometry (``fc``) Python Library
======================================
``fc`` is a library for processing and analyzing flow cytometry data. It performs:

* Data extraction from FCS files into numpy array-like structures
* Non-standard gating, including density-based two-dimensional gating.
* Automated analysis of calibration beads data, and standard curve generation.
* Data transformation, including exponentiation and conversion to absolute units (Molecules of Equivalent Fluorophore, MEF).
* Plotting, including generation of histograms, density plots and scatter plots.
* Simple input/output interfacing with Microsoft Excel files.

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
* ``XlsxWriter`` (0.7.3)

If ``fc`` is installed through ``pip`` or ``setuptools``, prerequisites should be taken care of automatically.

OS-specific recommendations are given below.

Windows
~~~~~~~
We recommend the Anaconda Python distribution from Continuum Analytics (https://store.continuum.io/cshop/anaconda/). Anaconda simplifies the installation of python and the required packages.

Ubuntu/Mint
~~~~~~~~~~~
Python is installed by default in Ubuntu and Linux Mint. We recommend managing packages through `pip` to get the appropriate versions. 

``fc`` Installation
-------------------
To install ``fc``, download the repo and run:

``python setup.py install``

Linux users may need to request administrative permissions by using the ``sudo`` command.

Examples
========
The ``examples`` directory contains simple ready-to-run scripts that can be used as a starting point for flow cytometry data analysis. These files can be run as regular python scripts. The scripts included are:

* ``run_no_excel_no_beads.py`` - This script will read data in the ``FCFiles`` folder, perform exponential transformation in all channels, and produce FSC/SSC scatter plots and fluorescence histograms in the ``plot_gated`` directory.
* ``run_no_excel.py`` - This script will read data in the ``FCFiles`` folder, process the provided calibration beads data file, generate a calibration curve, process all the sample data files, transform fluorescence values to absolute (MEF) units, and produce plots.
* ``run_excel.py`` - This script is intended for people who wouldn't like to write code, or who would appreciate the convenience of a standard excel input/output file format. It does essentially the same as ``run_no_excel.py``; however, it uses an input excel file specifying the samples to read and details about the analysis to perform. After processing the data, the script writes the results of the analysis in another excel file, in a format convenient for further analysis. 

Report Bugs
===========
The official way to report a bug is through the issue tracker on github (https://github.com/castillohair/fc/issues). Try to be as explicit as possible when describing your issue. Ideally, a set of instructions to reproduce the error should be provided, together with the version of all the relevant packages you are using.

Request Features
================
Features can also be requested through the issue tracker on github. Try to be as descriptive as possible about the desired feature, and indicate whether you would be willing to write the code necessary to implement it.

Contributing
============
We can use some help! If you are interested in writing code for this project, either for adding some new feature or correcting a bug, please check the DEVELOPING.rst file.
