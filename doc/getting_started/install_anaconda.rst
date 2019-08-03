Installing FlowCal with Anaconda
====================================

To install Anaconda and ``FlowCal``, do the following:

1. Navigate to https://www.anaconda.com/distribution/#download-section. Make sure that your operating system is selected (Windows, macOS, Linux). Click on the "Download" button below the "Python 3.7 version" message. This will download the installer.

.. note:: **Windows**: If your computer is a 32-bit PC, click on the message "32-Bit Graphical Installer" instead of the "Download" button. If you don't know whether yours is a 32 or 64 computer but you have purchased it in the last five years, it is probably a 64-bit computer and you can ignore this message.

.. note:: Python 2.7 is also supported. However, we recommend downloading the Python 3.7 version of Anaconda.

2. Double click the installer (.exe in Windows, .pkg in OS X) and follow the instructions on screen.

.. note:: **Windows**: During installation, on the "Advanced Installation Options" screen, make sure to check both "Add Anaconda to my PATH environment variable" and "Register Anaconda as my default Python". Recent versions of Anaconda suggest to keep the first option unchecked. However, this option is necessary for the installation script on step 4 to work.

3. Download ``FlowCal`` from `here <https://github.com/taborlab/FlowCal/archive/master.zip>`_. A file called ``FlowCal-master.zip`` will be downloaded. Unzip this file.

4. Inside the unzipped folder, double click on ``Install FlowCal (Windows).bat`` or ``Install FlowCal (OSX)`` if you are using Windows or OS X, respectively. This will open a terminal window and install ``FlowCal``. The installation procedure may take a few minutes. When installation is finished, the terminal will show the message “Press Enter to finish...”. If the installation was successful, your terminal should look like the figure below. Press Enter to close the terminal window.

.. image:: https://www.dropbox.com/s/9ygziuk8r2r93kw/installation_completed.png?raw=1

.. note:: **Windows**: If the following message appears after double clicking ``Install FlowCal (Windows)``: “Windows protected your PC – Windows SmartScreen prevented an unrecognized app from starting...”, click on the “More info” link under the text, and then click on the “Run anyway” button. This will remove the security restriction from the program and allow it to run properly.

.. note:: **Mac OS X**: If the following error message appears after double clicking ``Install FlowCal (OSX)``: “’Install FlowCal (OSX)’ can’t be opened because it is from an unidentified developer.”, navigate to System Preferences -> Security and Privacy -> General, and click the “Open Anyways” button adjacent to the message stating “’Install FlowCal (OSX)’ was blocked from opening because it is not from an identified developer”. This will remove the security restriction from the program and allow it to run properly.

To see ``FlowCal`` in action, head to the :doc:`Excel UI</excel_ui/index>` section. The ``FlowCal`` zip file includes an ``examples`` folder with files that you can use while following the instructions.