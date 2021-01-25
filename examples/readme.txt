Introduction
============

This folder contains flow cytometry data from a simple bacterial 2,4‐diacetylphloroglucinol (DAPG) induction experiment and the necessary files that demonstrate how to analyze this data with FlowCal.


Experiment description
======================

Escherichia coli expressing the fluorescent protein superfolder GFP (sfGFP) from the DAPG-inducible promoter Pphlf were induced to 0, 2.33, 4.36, 8.16, 15.3, 28.6, 53.5, 100, 187, and 350 micromolar DAPG and grown shaking at 37C for ~6 hours prior to fluorophore maturation (wherein cells were incubated with the bacterial translation inhibitor chloramphenicol for one hour to allow fluorescent proteins to mature) and measurement of cellular fluorescence by flow cytometry. The ten resulting flow cytometry standard (FCS) files are located in the "FCFiles/" folder and are named "sampleXXX.fcs". An additional sample containing calibration beads was also measured; its corresponding FCS file is named "sample001.fcs".

A minimum control sample containing E. coli lacking sfGFP was measured along with its own beads sample on a separate day; the corresponding FCS files are located in the "FCFiles/min/" folder.

A maximum control sample containing E. coli expressing sfGFP from Pphlf but lacking the DAPG-sensitive PhlF transcriptional repressor was measured along with its own beads sample on a third day; the corresponding FCS files are located in the "FCFiles/max/" folder.

For more information, see this study: https://doi.org/10.15252/msb.20209618.


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

This example shows how to process cell-sample and bead-sample FCS files using the Excel UI to produce plots and an Excel file with statistics.

To run, start FlowCal by running the "Run FlowCal" program. An "open file" dialog will appear. Navigate to this folder and select "experiment.xlsx".


Python API, without using calibration beads data
------------------------------------------------

This example shows how to process cell-sample FCS files entirely from Python with FlowCal.

To run, use the script "analyze_no_mef.py" as a regular python program.


Python API, using calibration beads data
----------------------------------------

This example shows how to process cell-sample and bead-sample FCS files entirely from Python with FlowCal.

To run, use the script "analyze_mef.py" as a regular python program.


Python API, using calibration beads data and an input Excel file
----------------------------------------------------------------

This example shows how to process cell-sample and bead-sample FCS files using the Excel UI to obtain processed flow cytometry data for further analysis in Python.

To run, use the script "analyze_excel_ui.py" as a regular python program.
