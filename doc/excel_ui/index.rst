FlowCal's Excel UI
==================

``FlowCal``'s Excel UI allows for easy processing of data from a set of FCS files without having to write any code. Processing includes gating, calibration of fluorescence data from arbitrary units (a.u.) to Molecules of Equivalent Fluorophore (MEF), generation of plots, and calculation of different statistics (mean, median, C.V., etc). All the samples to be processed are specified in an Excel file, which is then read by ``FlowCal``. The calculated statistics in either a.u. or MEF are written to another Excel file, from which the user can perform any additional manipulations.

.. toctree::
   :maxdepth: 1

   how_to.rst
   input_format.rst
   outputs.rst
   cmd_interface.rst
