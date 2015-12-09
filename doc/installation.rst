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
