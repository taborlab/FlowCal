How to use FlowCal's Excel UI
=============================

To use the ``FlowCal``’s Excel UI, follow these steps:

1. Make and save an Excel file indicating the FCS files to process. Click :doc:`here <input_format>` for information on how to make a properly formatted input Excel file.
2. Launch ``FlowCal``’s Excel UI by double clicking on ``Run FlowCal (Windows).bat`` or ``Run FlowCal (OSX)``.
3. A window will appear requesting an input Excel file. Locate the Excel file made in step 1 and click on “Open”.
4. ``FlowCal`` will start :doc:`processing<analysis>` the indicated calibration beads and cell samples. A terminal window will appear indicating the progress of the analysis. 
5. When the analysis finishes, the message “Press Enter to finish...” will appear. Press Enter and close the terminal window. A set of :ref:`plots<excel-ui-outputs-plots>` and :ref:`an Excel file<excel-ui-outputs-excel>` with statistics will appear in the same directory in which the input Excel file was located.