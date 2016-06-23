#!/usr/bin/python
"""
FlowCal Python API example, using an input Excel UI file.

This script is divided in two parts. Part one uses FlowCal's Excel UI
to process calibration beads data and cell samples according to an input
Excel file. The exact operations performed are identical to when normally
using the Excel UI. However, instead of generating an output Excel file, a
list of objects representing gated and transformed flow cytometry samples
is obtained.

Part two exemplifies how to use the processed cell sample data with
FlowCal's plotting and statistics modules, in order to produce interesting
plots.

For details about the experiment, samples, and instrument used, please
consult readme.txt.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import FlowCal

if __name__ == "__main__":

    ###
    # Part 1: Obtaining processed flow cytometry data from an input Excel file
    ###

    # ``FlowCal.excel_ui.read_table() reads a table from an Excel file, and
    # returns the contents as a pandas dataframe.
    # We need tables describing the instruments used, the calibration beads
    # files, and the cell samples.
    instruments_table = FlowCal.excel_ui.read_table(filename='experiment.xlsx',
                                                    sheetname='Instruments',
                                                    index_col='ID')
    beads_table = FlowCal.excel_ui.read_table(filename='experiment.xlsx',
                                              sheetname='Beads',
                                              index_col='ID')
    samples_table = FlowCal.excel_ui.read_table(filename='experiment.xlsx',
                                                sheetname='Samples',
                                                index_col='ID')

    # Process beads samples
    # ``FlowCal.excel_ui.process_beads_table()`` reads a properly formatted
    # table describing calibration beads samples (``beads_table``), performs
    # density gating as specified in this table, and generates MEF
    # transformation functions that can later be applied to cell samples.
    # To do so, it requires a table describing the flow cytometer used
    # (``instruments_table``). Here, we also use verbose mode, and indicate that
    # plots describing individual steps should be generated in the folder
    # "plot_beads". The result is a list of ``FCSData`` objects representing
    # gated and transformed calibration beads samples (``beads_samples``), and
    # a dictionary containing MEF transformation functions
    # (``mef_transform_fxns``). This will be used later to process cell samples.
    beads_samples, mef_transform_fxns = FlowCal.excel_ui.process_beads_table(
        beads_table=beads_table,
        instruments_table=instruments_table,
        verbose=True,
        plot=True,
        plot_dir='plot_beads')

    # Process cell samples
    # ``FlowCal.excel_ui.process_samples_table()`` reads a properly formatted
    # table describing cell samples (``samples_table``), and performs density
    # gating and unit transformations as specified in this table. To do so, it
    # requires a table describing the flow cytometer used
    # (``instruments_table``) and a corresponding dictionary with MEF
    # transformation functions (``mef_transform_fxns``). Here, we also use
    # verbose mode, and indicate that plots of each sample should be generated
    # in the folder "plot_samples". The result is a list of ``FCSData`` objects
    # (``samples``) representing gated and transformed flow cytometry cell
    # samples.
    samples = FlowCal.excel_ui.process_samples_table(
        samples_table=samples_table,
        instruments_table=instruments_table,
        mef_transform_fxns=mef_transform_fxns,
        verbose=True,
        plot=True,
        plot_dir='plot_samples')

    ###
    # Part 2: Examples on how to use processed cell sample data
    ###

    # We will read IPTG concentrations for each sample from the Excel file.
    # ``samples_table`` contains all the data from sheet "Samples", including
    # data not directly used by ``FlowCal.excel_ui.process_samples_table()``.
    # ``samples_table`` is a pandas dataframe, with each column having the
    # same name as the corresponding header in the Excel file.
    # We multiply the IPTG concentration by 1000 to convert to micromolar.
    iptg = samples_table['IPTG (mM)']*1000

    # Histogram of all samples
    # Here, we plot the fluorescence histograms of all five samples in the same
    # figure, using ``FlowCal.plot.hist1d``. Note how this function can be used
    # in the context of accessory matplotlib functions to modify the axes
    # limits and labels and add a legend, among others.
    plt.figure(figsize=(6,3.5))
    FlowCal.plot.hist1d(samples,
                        channel='FL1',
                        histtype='step',
                        bins=128)
    plt.ylim([0, 2000])
    plt.xlabel('FL1  (Molecules of Equivalent Fluorescein, MEFL)')
    plt.legend(['{} $\mu M$ IPTG'.format(i) for i in iptg],
               loc='upper left',
               fontsize='small')
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=200)
    plt.close()

    # Here we illustrate how to obtain statistics from the fluorescence of each
    # sample, and how to use them in a plot.
    # The stats module contains functions to calculate different statistics
    # such as mean, median, and standard deviation. Here, we calculate the
    # geometric mean from channel FL1 of each sample, and plot them against the
    # corresponding IPTG concentrations.
    samples_fluorescence = [FlowCal.stats.gmean(s, channels='FL1')
                            for s in samples]
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(iptg,
             samples_fluorescence,
             marker='o',
             color=(0, 0.4, 0.7))
    plt.xlabel('IPTG Concentration ($\mu M$)')
    plt.ylabel('FL1 Fluorescence (MEFL)')
    plt.tight_layout()
    plt.savefig('dose_response.png', dpi=200)
    plt.close()

    print("\nDone.")