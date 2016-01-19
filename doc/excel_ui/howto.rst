How to use FlowCal's Excel UI
=============================

``FlowCal``'s Excel UI allows for easy processing of data from a set of FCS files without having to write any code. Processing includes gating, calibration of fluorescence data from arbitrary units (a.u.) to Molecules of Equivalent Fluorophore (MEF), generation of plots, and calculation of different statistics (mean, median, C.V., etc). All the samples to be processed are specified in an Excel file, which is then read by ``FlowCal``. The calculated statistics in either a.u. or MEF are written to another Excel file, from which the user can perform any additional manipulations.

To use the ``FlowCal``’s Excel UI, follow these steps:

1. Make and save an Excel file indicating the FCS files to process. Click :doc:`here </excel_ui/format>` for more information on how to make a properly formatted input Excel file.
2. Launch ``FlowCal``’s Excel UI by double clicking on “Run FlowCal (Windows).bat” or “Run FlowCal (OSX)”.
3. A window will appear requesting an input Excel file. Locate the Excel file made in step 1 and click on “Open”.
4. ``FlowCal`` will start processing the indicated calibration beads and cell samples. A terminal window will appear indicating the progress of the analysis. When the analysis finishes, the message “Press Enter to finish...” will appear. Press Enter to close the terminal window.

To see a description of the files generated during the analysis by ``FlowCal``, click :doc:`here </excel_ui/output>`.