Installation
============

Installing ``FlowCal`` with Anaconda
------------------------------------
We recommend that Windows and OS X users download and install Anaconda, a Python distribution that already includes commonly used Python packages, many of which are used by ``FlowCal``. OSX already includes its own version of Python, but it does not include some Python tools that ``FlowCal`` requires. Therefore, Anaconda is recommended.

To install Anaconda, do the following:

1. Navigate to https://www.continuum.io/downloads. If you are using Windows, click on “Windows 64-bit Graphical Installer” or “Windows 32-bit Graphical Installer” under the “Python 2.7” column, depending on whether your computer is 32-bit or 64-bit. The 64-bit version is strongly recommended. Similarly, if you are using OS X, click on “Mac OS X 64-bit Graphical Installer” under the “Python 2.7” column. This will download the installer.

2. Double click the installer (.exe in Windows, .pkg in OS X) and follow the instructions on screen.

3. Download ``FlowCal`` from `here <https://github.com/taborlab/FlowCal/archive/master.zip>`_. A file called ``FlowCal-master.zip`` will be downloaded. Unzip this file.

4. Inside the unzipped folder, double click on ``Install FlowCal (Windows).bat`` or ``Install FlowCal (OSX)`` if you are using Windows or OS X, respectively. This will open a terminal window and install ``FlowCal``. The installation procedure may take a few minutes. When installation is finished, the terminal will show the message “Press Enter to finish...”. If the installation was successful, your terminal should look like the figure below. Press Enter to close the terminal window.

.. image:: _static/installation_completed.png

*Mac OS X only*: If the following error message appears after double clicking ``Install FlowCal (OSX)``: “’Install FlowCal (OSX)’ can’t be opened because it is from an unidentified developer.”, navigate to System Preferences -> Security and Privacy -> General, and click the “Open Anyways” button adjacent to the message stating “’Install FlowCal (OSX)’ was blocked from opening because it is not from an identified developer”. This will remove the security restriction from the program and allow it to run properly.

To see ``FlowCal`` in action, head to the :doc:`How to use the Excel UI</excel_ui/howto>` section. The ``FlowCal`` zip file includes an ``examples`` folder with files that you can use while following the tutorial.

Installing ``FlowCal`` in (barebones) Python
--------------------------------------------
Python 2.7 is required, along with ``pip`` and ``setuptools``. The following Python packages should also be present:

* ``numpy`` (>=1.8.2)
* ``scipy`` (>=0.14.0)
* ``matplotlib`` (>=1.3.1)
* ``palettable`` (>=2.1.1)
* ``scikit-learn`` (>=0.16.0)
* ``pandas`` (>=0.16.1)
* ``xlrd`` (>=0.9.2)
* ``XlsxWriter`` (>=0.5.2)

If you have ``pip``, a ``requirements.txt`` file is provided, such that the required packages can be installed by running::

	pip install -r requirements.txt

Linux and Mac OSX users may need to request administrative permissions by preceding this command with ``sudo``.

To install ``FlowCal``, run the following in ``FlowCal``'s root directory::

	python setup.py install

Again, some users may need to precede this command with ``sudo``.

*Ubuntu/Linux Mint*: ``FlowCal`` might need more recent versions of some python packages than the ones provided via ``apt``. To upgrade these, some non-python packages need to be installed in your system. On a freshly installed Linux Mint 17.3 system, the following packages need to be manually installed: 

* ``gcc``
* ``g++``
* ``gfortran``
* ``libblas-dev``
* ``liblapack-dev``
* ``libfreetype6-dev``
* ``python-dev``
* ``python-tk``
* ``python-pip``

All of these can be installed using::

	sudo apt-get install <package-name>

After this, you may upgrade the requirements using ``pip`` and then install ``FlowCal``.
