Introduction
============

This folder contains flow cytometry data from a simple bacterial IPTG-induction experiment, and the necessary files that demonstrate how to analyze this data with FlowCal.


Experimental details
====================

This experiment used an E. coli strain with a high-copy plasmid containing sfGFP under control of the IPTG-inducible promoter Ptac. Five samples were started from this strain, with IPTG added to the following final concentrations: 0, 81, 161, 318, and 1000 micromolar. Samples were grown until late exponential, and fluorescence measurements were taken via flow cytometry. The five resulting flow cytometry standard (FCS) files are contained in the "FCFiles" folder, and are named "DataXXX.fcs". An additional sample containing calibration beads was also measured, and the corresponding FCS file is named "Beads006.fcs".


Details on the flow cytometer used
==================================

Data was acquired with a BD FACScan flow cytometer with a blue (488 nm, 30 mW) and a yellow (561 nm, 50 mW) laser. The following channels are relevant for these examples:

- FSC: Forward scatter channel
- SSC: Side scatter channel
- FL1: Fluorescence channel with a 510/21nm filter, commonly used for GFP.
- FL3: Fluorescence channel with a 650nm long-pass filter, commonly used for mCherry.

Data files are FCS version 3.0, with integer 10-bit data.


Examples included
=================

Excel UI
--------

This example shows how to process a set of cell sample and bead sample FCS files with the Excel UI, and produce a set of plots and an output Excel file with statistics.

To run, start FlowCal by running the "Run FlowCal" program. An "open file" dialog will appear. Navigate to this folder and select "experiment.xlsx".


Python API, without using calibration beads data
------------------------------------------------

This example shows how to process a set of cell sample FCS files entirely from Python with FlowCal.

To run, use the script "analyze_no_mef.py" as a regular python program.


Python API, using calibration beads data
----------------------------------------

This example shows how to process a set of cell sample and bead sample FCS files entirely from Python with FlowCal.

To run, use the script "analyze_mef.py" as a regular python program.


Python API, using calibration beads data and an input Excel file
----------------------------------------------------------------

This example shows how to process a set of cell sample and bead sample FCS files with the Excel UI, and obtain processed flow cytometry data for further analysis in Python.

To run, use the script "analyze_excel_ui.py" as a regular python program.
