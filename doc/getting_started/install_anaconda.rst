Installing FlowCal with Anaconda
====================================

To install Anaconda and ``FlowCal``, do the following:

1. Navigate to https://www.continuum.io/downloads. If you are using Windows, click on “Windows 64-bit Graphical Installer” or “Windows 32-bit Graphical Installer” under the “Python 2.7” column, depending on whether your computer is 32-bit or 64-bit. The 64-bit version is strongly recommended. Similarly, if you are using OS X, click on “Mac OS X 64-bit Graphical Installer” under the “Python 2.7” column. This will download the installer.

.. note:: Make sure to download the installer for Python 2.7 and not 3.x. ``FlowCal`` is not currently compatible with Python 3.

2. Double click the installer (.exe in Windows, .pkg in OS X) and follow the instructions on screen.

.. note:: **Windows**: During installation, on the "Advanced Installation Options" screen, make sure to check both "Add Anaconda to my PATH environment variable" and "Register Anaconda as my default Python 2.7". Recent versions of Anaconda suggest to keep the first option unchecked. However, this option is necessary for the installation script on step 4 to work.

3. Download ``FlowCal`` from `here <https://github.com/taborlab/FlowCal/archive/master.zip>`_. A file called ``FlowCal-master.zip`` will be downloaded. Unzip this file.

4. Inside the unzipped folder, double click on ``Install FlowCal (Windows).bat`` or ``Install FlowCal (OSX)`` if you are using Windows or OS X, respectively. This will open a terminal window and install ``FlowCal``. The installation procedure may take a few minutes. When installation is finished, the terminal will show the message “Press Enter to finish...”. If the installation was successful, your terminal should look like the figure below. Press Enter to close the terminal window.

.. image:: https://www.dropbox.com/s/9ygziuk8r2r93kw/installation_completed.png?raw=1

.. note:: **Windows**: If the following message appears after double clicking ``Install FlowCal (Windows)``: “Windows protected your PC – Windows SmartScreen prevented an unrecognized app from starting...”, click on the “More info” link under the text, and then click on the “Run anyway” button. This will remove the security restriction from the program and allow it to run properly.

.. note:: **Mac OS X**: If the following error message appears after double clicking ``Install FlowCal (OSX)``: “’Install FlowCal (OSX)’ can’t be opened because it is from an unidentified developer.”, navigate to System Preferences -> Security and Privacy -> General, and click the “Open Anyways” button adjacent to the message stating “’Install FlowCal (OSX)’ was blocked from opening because it is not from an identified developer”. This will remove the security restriction from the program and allow it to run properly.

To see ``FlowCal`` in action, head to the :doc:`Excel UI</excel_ui/index>` section. The ``FlowCal`` zip file includes an ``examples`` folder with files that you can use while following the instructions.