Command Line Interface (Advanced)
=================================

The Excel UI can be run from a command line interpreter with one of the following equivalent statements::

	flowcal [-h] [-i [INPUTPATH]] [-o [OUTPUTPATH]] [-v] [-p] [-H]
	FlowCal [-h] [-i [INPUTPATH]] [-o [OUTPUTPATH]] [-v] [-p] [-H]
	python -m FlowCal.excel_ui [-h] [-i [INPUTPATH]] [-o [OUTPUTPATH]] [-v] [-p] [-H]

Where the flags are::

    -h, --help            show this help message and exit
    -i [INPUTPATH], --inputpath [INPUTPATH]
                          input Excel file name. If not specified, show open
                          file window
    -o [OUTPUTPATH], --outputpath [OUTPUTPATH]
                          output Excel file name. If not specified, use
                          [INPUTPATH]_output
    -v, --verbose         print information about individual processing steps
    -p, --plot            generate and save density plots/histograms of beads
                          and samples
    -H, --histogram-sheet
                          generate sheet in output Excel file specifying
                          histogram bins

Running ``FlowCal``'s Excel UI without any flags will show the open file dialog to select an :doc:`input Excel file <input_format>`. Once a file is selected, FlowCal will generate an :doc:`output Excel file <outputs>`. In contrast to using ``Run FlowCal (OSX)`` or ``Run FlowCal (Windows).bat``, the statement above with no flags will not display any messages during processing or generate any plots. To display messages and generate plots, use::

	flowcal -v -p

``Run FlowCal (OSX)`` and ``Run FlowCal (Windows).bat`` use, in fact, the following equivalent statement::

	python -m FlowCal.excel_ui -v -p

.. note::
	In Mac OSX, a critical error may appear when trying to run the Excel UI from the command line. The error message is quite long, but one of the last lines reads similarly to this::

		libc++abi.dylib: terminating with uncaught exception of type NSException

	This is due to the ``macosx`` matplotlib backend conflicting with the ``TkInter`` library used to show the open file window. To solve this, you need to change matplotlib's backend to ``TkAgg``. A few ways to do so can be found `here <http://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`_.

	We recommend changing the matplotlib's backend temporarily by setting the ``MPLBACKEND`` environment variable. If you follow this method, you should run the following before calling ``FlowCal``::

		export MPLBACKEND="TkAgg"

	This is actually the solution implemented in ``Run FlowCal (OSX)``.

Using the command line arguments, one can create a batch script to process several Excel files at once, each pointing to a different set of FCS files. Such script would have the form::

	flowcal -i input_excel_file_1.xlsx -o output_excel_file_1.xlsx
	flowcal -i input_excel_file_2.xlsx -o output_excel_file_2.xlsx
	flowcal -i input_excel_file_3.xlsx -o output_excel_file_3.xlsx
	...
