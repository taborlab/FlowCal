FlowCal's Excel UI
==================

``FlowCal``'s Excel UI allows for easy processing of flow cytometry data from a set of FCS files without having to write any code. The user simply writes an :doc:`Excel file <input_format>` listing the samples to be analyzed, along with some options. FlowCal then :doc:`processes<analysis>` those samples and produces :ref:`plots<excel-ui-outputs-plots>` and :ref:`statistics<excel-ui-outputs-excel>`, which can then be used in subsequent analyses. Calibration beads data can be included to report results in :doc:`calibrated MEF units</fundamentals/calibration>`.

.. toctree::
   :maxdepth: 1

   how_to.rst
   input_format.rst
   analysis.rst
   outputs.rst
   cmd_interface.rst
