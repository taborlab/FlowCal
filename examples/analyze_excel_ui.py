#!/usr/bin/python
"""
FlowCal Python API example, using an input Excel UI file.

This script is divided in two parts. Part one uses FlowCal's Excel UI
to process calibration beads data and cell samples according to an input
Excel file. The exact operations performed are identical to when normally
using the Excel UI. However, instead of generating an output Excel file, an
OrderedDict of objects representing gated and transformed flow cytometry
samples is obtained.

Part two exemplifies how to use the processed cell sample data with
FlowCal's plotting and statistics modules to produce interesting plots.

For details about the experiment, samples, and instrument used, please
consult readme.txt.

"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import FlowCal

if __name__ == "__main__":

    ###
    # Part 1: Obtaining processed flow cytometry data from an input Excel file
    ###

    # ``FlowCal.excel_ui.read_table() reads a table from an Excel file and
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
    # (``instruments_table``). Here, we also use verbose mode and indicate that
    # plots describing individual steps should be generated in the folder
    # "plot_beads". The result is a dictionary of ``FCSData`` objects
    # representing gated and transformed calibration beads samples
    # (``beads_samples``) and a dictionary containing MEF transformation
    # functions (``mef_transform_fxns``). This will be used later to process
    # cell samples.
    beads_samples, mef_transform_fxns = FlowCal.excel_ui.process_beads_table(
        beads_table=beads_table,
        instruments_table=instruments_table,
        verbose=True,
        plot=True,
        plot_dir='plot_beads')

    # Process cell samples
    # ``FlowCal.excel_ui.process_samples_table()`` reads a properly formatted
    # table describing cell samples (``samples_table``) and performs density
    # gating and unit transformations as specified in this table. To do so, it
    # requires a table describing the flow cytometer used
    # (``instruments_table``) and a corresponding dictionary with MEF
    # transformation functions (``mef_transform_fxns``). Here, we also use
    # verbose mode and indicate that plots of each sample should be generated
    # in the folder "plot_samples". The result is an OrderedDict of
    # ``FCSData`` objects keyed on the sample ID (``samples``) representing
    # gated and transformed flow cytometry cell samples.
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

    # Each entry in the Excel table has a corresponding ID, which can be used
    # to reference information associated with or the sample loaded from a
    # row in the Excel file. Collect the IDs of the non-control samples (i.e.,
    # 'S0001', ..., 'S0010').
    sample_ids = ['S00{:02}'.format(n) for n in range(1,10+1)]

    # We will read DAPG concentrations from the Excel file. ``samples_table``
    # contains all the data from sheet "Samples", including data not directly
    # used by ``FlowCal.excel_ui.process_samples_table()``. ``samples_table``
    # is a pandas dataframe, with each column having the same name as the
    # corresponding header in the Excel file.
    dapg = samples_table.loc[sample_ids,'DAPG (uM)']

    # Plot 1: Histogram of all samples
    #
    # Here, we plot the fluorescence histograms of all ten samples in the same
    # figure using ``FlowCal.plot.hist1d``. Note how this function can be used
    # in the context of accessory matplotlib functions to modify the axes
    # limits and labels and to add a legend, among other things.

    # Color each histogram according to its DAPG concentration. Linearize the
    # color transitions using a logarithmic normalization to match the
    # logarithmic spacing of the DAPG concentrations. (Concentrations are also
    # augmented slightly to move the 0.0 concentration into the log
    # normalization range.)
    cmap = mpl.colormaps['gray_r']
    norm = mpl.colors.LogNorm(vmin=1e0, vmax=3500.)
    colors = [cmap(norm(dapg_i+4.)) for dapg_i in dapg]

    plt.figure(figsize=(6,3.5))
    FlowCal.plot.hist1d([samples[s_id] for s_id in sample_ids],
                        channel='FL1',
                        histtype='step',
                        bins=128,
                        edgecolor=colors)
    plt.ylim((0,2500))
    plt.xlim((0,5e4))
    plt.xlabel('FL1  (Molecules of Equivalent Fluorescein, MEFL)')
    plt.legend([r'{:.1f} $\mu M$ DAPG'.format(i) for i in dapg],
               loc='upper left',
               fontsize='small')
    plt.tight_layout()
    plt.savefig('histograms.png', dpi=200)
    plt.close()

    # Plot 2: Dose response curve
    #
    # Here, we illustrate how to obtain statistics from the fluorescence of
    # each sample and how to use them in a plot. The stats module contains
    # functions to calculate different statistics such as mean, median, and
    # standard deviation. In this example, we calculate the mean from channel
    # FL1 of each sample and plot them against the corresponding DAPG
    # concentrations.
    samples_fluorescence = [FlowCal.stats.mean(samples[s_id], channels='FL1')
                            for s_id in sample_ids]
    min_fluorescence = FlowCal.stats.mean(samples['min'], channels='FL1')
    max_fluorescence = FlowCal.stats.mean(samples['max'], channels='FL1')

    dapg_color = '#ffc400'  # common color used for DAPG-related plots

    plt.figure(figsize=(3,3))
    plt.plot(dapg,
             samples_fluorescence,
             marker='o',
             color=dapg_color)

    # Illustrate min and max bounds
    plt.axhline(min_fluorescence,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Min', x=2e2, y=1.6e2, ha='left', va='bottom', color='gray')
    plt.axhline(max_fluorescence,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Max', x=-0.7, y=5.2e3, ha='left', va='top', color='gray')

    plt.yscale('log')
    plt.ylim((5e1,1e4))
    plt.xscale('symlog')
    plt.xlim((-1e0, 1e3))
    plt.xlabel(r'DAPG Concentration ($\mu M$)')
    plt.ylabel('FL1 Fluorescence (MEFL)')
    plt.tight_layout()
    plt.savefig('dose_response.png', dpi=200)
    plt.close()

    # Plot 3: Dose response violin plot
    #
    # Here, we use a violin plot to show the fluorescence of (almost) all
    # cells as a function of DAPG. (The `upper_trim_fraction` and
    # `lower_trim_fraction` parameters eliminate the top and bottom 1% of
    # cells from each violin for aesthetic reasons. The summary statistic,
    # which is illustrated as a horizontal line atop each violin, is
    # calculated before cells are removed, though.) We set `yscale` to 'log'
    # because the cytometer used to collect this data produces positive
    # integer data (as opposed to floating-point data, which can sometimes be
    # negative), so the added complexity of a logicle y-scale (which is the
    # default) is not necessary.

    # FlowCal violin plots can also illustrate a mathematical model alongside
    # the violins. To take advantage of this feature, we first recapitulate a
    # model of the fluorescent protein (sfGFP) fluorescence produced by this
    # DAPG sensor as a function of DAPG (from this study:
    # https://doi.org/10.15252/msb.20209618). sfGFP fluorescence is cellular
    # fluorescence minus autofluorescence.
    def dapg_sensor_output(dapg_concentration):
        mn = 86.
        mx = 3147.
        K  = 20.
        n  = 3.57
        if dapg_concentration <= 0:
            return mn
        else:
            return mn + ((mx-mn)/(1+((K/dapg_concentration)**n)))

    # To model cellular fluorescence, which we are plotting with this violin
    # plot, we must add autofluorescence back to the sfGFP signal. For this
    # model, autofluorescence is the mean fluorescence of an E. coli strain
    # lacking sfGFP, which our min control is.
    autofluorescence = FlowCal.stats.mean(samples['min'], channels='FL1')
    def dapg_sensor_cellular_fluorescence(dapg_concentration):
        return dapg_sensor_output(dapg_concentration) + autofluorescence

    plt.figure(figsize=(4,3.5))
    FlowCal.plot.violin_dose_response(
        data=[samples[s_id] for s_id in sample_ids],
        channel='FL1',
        positions=dapg,
        min_data=samples['min'],
        max_data=samples['max'],
        model_fxn=dapg_sensor_cellular_fluorescence,
        violin_kwargs={'facecolor':dapg_color,
                       'edgecolor':'black'},
        violin_width_to_span_fraction=0.075,
        xscale='log',
        yscale='log',
        ylim=(1e1, 3e4),
        draw_model_kwargs={'color':'gray',
                           'linewidth':3,
                           'zorder':-1,
                           'solid_capstyle':'butt'})
    plt.xlabel(r'DAPG Concentration ($\mu M$)')
    plt.ylabel('FL1 Fluorescence (MEFL)')
    plt.tight_layout()
    plt.savefig('dose_response_violin.png', dpi=200)
    plt.close()

    print("\nDone.")
