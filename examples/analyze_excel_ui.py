#!/usr/bin/python
"""
FlowCal Python API example, using an input Excel UI file.

This script is divided in two parts. Part one uses FlowCal's Excel UI
to process calibration beads data and cell samples according to an input
Excel file. The exact operations performed are identical to when normally
using the Excel UI. However, instead of generating an output Excel file, an
OrderedDict of objects representing gated and transformed flow cytometry
samples is obtained. In addition, we perform multi-color compensation on all
samples using data from no-fluorophore and single-fluorophore control samples
(NFC and SFCs).

Part two exemplifies how to use the processed cell sample data with
FlowCal's plotting and statistics modules to produce interesting plots.

For details about the experiment, samples, and instrument used, please
consult readme.txt.

"""
import six
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

    # Perform multi-color compensation
    # ``FlowCal.compensate.get_transform_fxn()`` generates a transformation
    # function that performs multi-color compensation on a specified set of
    # channels in order to remove fluorophore bleedthrough.
    # This function requires data from single-fluorophore controls (SFCs), one
    # per channel to compensate, each from cells containing only one
    # fluorophore. This function can optionally use data from a no-fluorophore
    # control (NFC).
    compensation_fxn = FlowCal.compensate.get_transform_fxn(
        nfc_sample=samples['NFC'],
        sfc_samples=[samples['SFC1'], samples['SFC2']],
        comp_channels=['FL1', 'FL2'],
    )
    # Compensate all samples
    samples_compensated = {s_id: compensation_fxn(s, ['FL1', 'FL2'])
                           for s_id, s in six.iteritems(samples)}

    ###
    # Part 2: Examples on how to use processed cell sample data
    ###

    # Each entry in the Excel table has a corresponding ID, which can be used
    # to reference information associated with or the sample loaded from a
    # row in the Excel file. Collect the IDs of the non-control samples (i.e.,
    # 'S001', ..., 'S009').
    sample_ids = ['S0{:02}'.format(n) for n in range(1, 9+1)]

    # We will read aTc concentrations from the Excel file. ``samples_table``
    # contains all the data from sheet "Samples", including data not directly
    # used by ``FlowCal.excel_ui.process_samples_table()``. ``samples_table``
    # is a pandas dataframe, with each column having the same name as the
    # corresponding header in the Excel file.
    atc = samples_table.loc[sample_ids, 'aTc (ng/mL)']

    # Plot 1: Histogram of all samples
    #
    # Here, we plot the fluorescence histograms of all nine samples in the same
    # figure using ``FlowCal.plot.hist1d``. Note how this function can be used
    # in the context of accessory matplotlib functions to modify the axes
    # limits and labels and to add a legend, among other things.

    # Color each histogram according to its corresponding aTc concentration.
    # Use a perceptually uniform colormap (cividis), and transition among
    # colors using a logarithmic normalization, which comports with the
    # logarithmically spaced aTc concentrations.
    cmap = mpl.cm.get_cmap('cividis')
    norm = mpl.colors.LogNorm(vmin=0.5, vmax=20)
    colors = [cmap(norm(atc_i)) if atc_i > 0 else cmap(0.0)
              for atc_i in atc]

    plt.figure(figsize=(6, 5.5))
    plt.subplot(2, 1, 1)
    FlowCal.plot.hist1d([samples[s_id] for s_id in sample_ids],
                        channel='FL1',
                        histtype='step',
                        bins=128,
                        edgecolor=colors)
    plt.ylim((0,2500))
    plt.xlim((0,5e4))
    plt.xlabel('FL1  (Molecules of Equivalent Fluorescein, MEFL)')
    plt.legend(['{:.1f} ng/mL aTc'.format(i) for i in atc],
               loc='upper left',
               fontsize='small')

    plt.subplot(2, 1, 2)
    FlowCal.plot.hist1d([samples[s_id] for s_id in sample_ids],
                        channel='FL2',
                        histtype='step',
                        bins=128,
                        edgecolor=colors)
    plt.ylim((0,2500))
    plt.xlim((0,5e4))
    plt.xlabel('FL2  (Molecules of Equivalent PE, MEPE)')
    plt.legend(['{:.1f} ng/mL aTc'.format(i) for i in atc],
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
    # standard deviation. In this example, we calculate the mean from channels
    # FL1 and FL2 of each sample and plot them against the corresponding aTc
    # concentrations.
    samples_fl1 = [FlowCal.stats.mean(samples[s_id], channels='FL1')
                   for s_id in sample_ids]
    samples_fl2 = [FlowCal.stats.mean(samples[s_id], channels='FL2')
                   for s_id in sample_ids]
    # No fluorescence control (NFC) will give the minimum fluorescence level in
    # both channels. Single fluorescence controls (SFCs) containing sfGFP or
    # mCherry only will give the maximum levels in channels FL1 and FL2.
    min_fl1 = FlowCal.stats.mean(samples['NFC'], channels='FL1')
    max_fl1 = FlowCal.stats.mean(samples['SFC1'], channels='FL1')
    min_fl2 = FlowCal.stats.mean(samples['NFC'], channels='FL2')
    max_fl2 = FlowCal.stats.mean(samples['SFC2'], channels='FL2')

    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.plot(atc,
             samples_fl1,
             marker='o',
             color='tab:green')
    plt.axhline(min_fl1,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Min', x=3e1, y=2e2, ha='left', va='bottom', color='gray')
    plt.axhline(max_fl1,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Max', x=-0.8, y=5.2e3, ha='left', va='top', color='gray')
    plt.yscale('log')
    plt.ylim((5e1,1e4))
    plt.xscale('symlog')
    plt.xlim((-1e0, 1e2))
    plt.xlabel('aTc Concentration (ng/mL)')
    plt.ylabel('FL1 Fluorescence (MEFL)')
    
    plt.subplot(1, 2, 2)
    plt.plot(atc,
             samples_fl2,
             marker='o',
             color='tab:orange')
    plt.axhline(min_fl2,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Min', x=3e1, y=3.5e1, ha='left', va='bottom', color='gray')
    plt.axhline(max_fl2,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Max', x=-0.8, y=2.e3, ha='left', va='top', color='gray')
    plt.yscale('log')
    plt.ylim((1e1,5e3))
    plt.xscale('symlog')
    plt.xlim((-1e0, 1e2))
    plt.xlabel('aTc Concentration (ng/mL)')
    plt.ylabel('FL2 Fluorescence (MEPE)')

    plt.tight_layout()
    plt.savefig('dose_response.png', dpi=200)
    plt.close()

    # Plot 3: Dose response violin plot
    #
    # Here, we use a violin plot to show the fluorescence of (almost) all
    # cells as a function of aTc. (The `upper_trim_fraction` and
    # `lower_trim_fraction` parameters eliminate the top and bottom 1% of
    # cells from each violin for aesthetic reasons. The summary statistic,
    # which is illustrated as a horizontal line atop each violin, is
    # calculated before cells are removed, though.) We set `yscale` to 'log'
    # because the cytometer used to collect this data produces positive
    # integer data (as opposed to floating-point data, which can sometimes be
    # negative), so the added complexity of a logicle y-scale (which is the
    # default) is not necessary.
    plt.figure(figsize=(8, 3.5))

    plt.subplot(1, 2, 1)
    FlowCal.plot.violin_dose_response(
        data=[samples[s_id] for s_id in sample_ids],
        channel='FL1',
        positions=atc,
        min_data=samples['NFC'],
        max_data=samples['SFC1'],
        xlabel='aTc Concentration (ng/mL)',
        xscale='log',
        yscale='log',
        ylim=(1e1,1e4),
        violin_width=0.12,
        violin_kwargs={'facecolor': 'tab:green',
                       'edgecolor':'black'},
        draw_model_kwargs={'color':'gray',
                           'linewidth':3,
                           'zorder':-1,
                           'solid_capstyle':'butt'},
        )
    plt.ylabel('FL1 Fluorescence (MEFL)')

    plt.subplot(1, 2, 2)
    FlowCal.plot.violin_dose_response(
        data=[samples[s_id] for s_id in sample_ids],
        channel='FL2',
        positions=atc,
        min_data=samples['NFC'],
        max_data=samples['SFC2'],
        xlabel='aTc Concentration (ng/mL)',
        xscale='log',
        yscale='log',
        ylim=(1e0,1e4),
        violin_width=0.12,
        violin_kwargs={'facecolor': 'tab:orange',
                       'edgecolor':'black'},
        draw_model_kwargs={'color':'gray',
                           'linewidth':3,
                           'zorder':-1,
                           'solid_capstyle':'butt'},
        )
    plt.ylabel('FL2 Fluorescence (MEPE)')

    plt.tight_layout()
    plt.savefig('dose_response_violin.png', dpi=200)
    plt.close()

    # Plot 4: Dose response violin plot of compensated data
    #
    # Here, we redraw the previous violin plot but using compensated data.
    # y axis will now be plotted in ``logicle`` scale since histograms will
    # be centered around zero due to compensation.
    plt.figure(figsize=(8, 3.5))

    plt.subplot(1, 2, 1)
    FlowCal.plot.violin_dose_response(
        data=[samples_compensated[s_id] for s_id in sample_ids],
        channel='FL1',
        positions=atc,
        min_data=samples_compensated['NFC'],
        max_data=samples_compensated['SFC1'],
        xlabel='aTc Concentration (ng/mL)',
        xscale='log',
        yscale='logicle',
        ylim=(-3e2, 1e4),
        violin_width=0.12,
        violin_kwargs={'facecolor': 'tab:green',
                       'edgecolor':'black'},
        draw_model_kwargs={'color':'gray',
                           'linewidth':3,
                           'zorder':-1,
                           'solid_capstyle':'butt'},
        )
    plt.ylabel('FL1 Fluorescence (MEFL)')

    plt.subplot(1, 2, 2)
    FlowCal.plot.violin_dose_response(
        data=[samples_compensated[s_id] for s_id in sample_ids],
        channel='FL2',
        positions=atc,
        min_data=samples_compensated['NFC'],
        max_data=samples_compensated['SFC2'],
        xlabel='aTc Concentration (ng/mL)',
        xscale='log',
        yscale='logicle',
        ylim=(-1e2, 1e4),
        violin_width=0.12,
        violin_kwargs={'facecolor': 'tab:orange',
                       'edgecolor':'black'},
        draw_model_kwargs={'color':'gray',
                           'linewidth':3,
                           'zorder':-1,
                           'solid_capstyle':'butt'},
        )
    plt.ylabel('FL2 Fluorescence (MEPE)')

    plt.tight_layout()
    plt.savefig('dose_response_violin_compensated.png', dpi=200)
    plt.close()

    print("\nDone.")
