#!/usr/bin/python
"""
FlowCal Python API example, without using calibration beads data.

This script is divided in two parts. Part one processes data from ten cell
samples and generates plots of each one.

Part two exemplifies how to use the processed cell sample data with
FlowCal's plotting and statistics modules to produce interesting plots.

For details about the experiment, samples, and instrument used, please
consult readme.txt.

"""
import os
import os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import FlowCal

###
# Definition of constants
###

# Names of the FCS files containing data from cell samples
samples_filenames = ['FCFiles/sample006.fcs',
                     'FCFiles/sample007.fcs',
                     'FCFiles/sample008.fcs',
                     'FCFiles/sample009.fcs',
                     'FCFiles/sample010.fcs',
                     'FCFiles/sample011.fcs',
                     'FCFiles/sample012.fcs',
                     'FCFiles/sample013.fcs',
                     'FCFiles/sample014.fcs',
                     'FCFiles/sample015.fcs']

# DAPG concentration of each cell sample, in micromolar.
dapg = np.array([0, 2.33, 4.36, 8.16, 15.3, 28.6, 53.5, 100, 187, 350])

# Plots will be generated after gating and transforming cell samples. These
# will be stored in the following folder.
samples_plot_dir = 'plot_samples'

if __name__ == "__main__":

    # Check that plot directory exists, create if it does not.
    if not os.path.exists(samples_plot_dir):
        os.makedirs(samples_plot_dir)

    ###
    # Part 1: Processing cell sample data
    ###
    print("\nProcessing cell samples...")

    # We will use the list ``samples`` to store processed, transformed flow
    # cytometry data of each sample.
    samples = []

    # Iterate over cell sample filenames
    for sample_id, sample_filename in enumerate(samples_filenames):

        # Load flow cytometry data from the corresponding FCS file.
        # ``FlowCal.io.FCSData(filename)`` returns an object that represents
        # flow cytometry data loaded from file ``filename``.
        print("\nLoading file \"{}\"...".format(sample_filename))
        sample = FlowCal.io.FCSData(sample_filename)
        
        # Data loaded from an FCS file is in "Channel Units", the raw numbers
        # reported from the instrument's detectors. The FCS file also contains
        # information to convert these into Relative Fluorescence Intensity
        # (RFI) values, commonly referred to as arbitrary fluorescence units
        # (a.u.). The function ``FlowCal.transform.to_rfi()`` performs this
        # conversion.
        print("Performing data transformation...")
        sample = FlowCal.transform.to_rfi(sample)

        # Gating

        # Gating is the process of removing measurements of irrelevant
        # particles while retaining only the population of interest.
        print("Performing gating...")

        # ``FlowCal.gate.start_end()`` removes the first and last few events.
        # Transients in fluidics can make these events slightly different from
        # the rest. This may not be necessary in all instruments.
        sample_gated = FlowCal.gate.start_end(sample,
                                              num_start=250,
                                              num_end=100)

        # ``FlowCal.gate.high_low()`` removes events outside a range specified
        # by a ``low`` and a ``high`` value. If these are not specified (as
        # shown below), the function removes events outside the channel's range
        # of detection.
        # Detectors in a flow cytometer have a finite range of detection. If the
        # fluorescence of a particle is higher than the upper limit of this
        # range, the instrument will incorrectly record it with a value equal to
        # this limit. The same happens for fluorescence values lower than the
        # lower limit of detection. These saturated events should be removed,
        # otherwise statistics may be calculated incorrectly.
        # Note that this might not be necessary with newer instruments that
        # record data as floating-point numbers (and in fact it might eliminate
        # negative events). To see the data type stored in your FCS files, run
        # the following instruction: ``print sample_gated.data_type``.
        # We will remove saturated events in the forward/side scatter channels,
        # and in the fluorescence channel FL1.
        sample_gated = FlowCal.gate.high_low(sample_gated,
                                             channels=['FSC','SSC','FL1'])

        # ``FlowCal.gate.density2d()`` preserves only the densest population as
        # seen in a 2D density diagram of two channels. This helps remove
        # particle aggregations and other sparse populations that are not of
        # interest (i.e. debris).
        # We use the forward and side scatter channels and preserve 85% of the
        # events. Finally, setting ``full_output=True`` instructs the function
        # to return additional outputs in the form of a named tuple.
        # ``gate_contour`` is a curve surrounding the gated region, which we
        # will use for plotting later.
        density_gate_output = FlowCal.gate.density2d(
            data=sample_gated,
            channels=['FSC','SSC'],
            gate_fraction=0.85,
            full_output=True)
        sample_gated = density_gate_output.gated_data
        gate_contour = density_gate_output.contour

        # Plot forward/side scatter 2D density plot and 1D fluorescence
        # histograms
        print("Plotting density plot and histogram...")

        # Parameters for the forward/side scatter density plot
        density_params = {}
        # We use the "scatter" mode, in which individual particles will be
        # plotted individually as in a scatter plot, but with a color
        # proportional to the particle density around.
        density_params['mode'] = 'scatter'

        # Parameters for the fluorescence histograms
        hist_params = {}
        hist_params['xlabel'] = 'FL1 Fluorescence (a.u.)'

        # Plot filename
        # The figure can be saved in any format supported by matplotlib (svg,
        # jpg, etc.) by just changing the extension.
        plot_filename = '{}/density_hist_{}.png'.format(
            samples_plot_dir,
            'S{:03}'.format(sample_id + 1))

        # Plot and save
        # The function ``FlowCal.plot.density_and_hist()`` plots a combined
        # figure with a 2D density plot at the top and an arbitrary number of
        # 1D histograms below. In this case, we will plot the forward/side
        # scatter channels in the density plot and a histogram of the
        # fluorescence channel FL1 below.
        # Note that we are providing data both before (``sample``) and after
        # (``sample_gated``) gating. The 1D histogram will display the ungated
        # dataset with transparency and the gated dataset in front with a solid
        # color. In addition, we are providing ``gate_contour`` from the
        # density gating step, which will be displayed in the density diagram.
        # This will result in a convenient representation of the data both
        # before and after gating.
        FlowCal.plot.density_and_hist(
            sample,
            sample_gated,
            density_channels=['FSC','SSC'],
            hist_channels=['FL1'],
            gate_contour=gate_contour,
            density_params=density_params,
            hist_params=hist_params,
            savefig=plot_filename)

        # Save cell sample object
        samples.append(sample_gated)

    ###
    # Part 3: Examples on how to use processed cell sample data
    ###

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
    FlowCal.plot.hist1d(samples,
                        channel='FL1',
                        histtype='step',
                        bins=128,
                        edgecolor=colors)
    plt.ylim((0,2500))
    plt.xlim((0,5e3))
    plt.xlabel('FL1 Fluorescence (a.u.)')
    plt.legend([r'{} $\mu M$ DAPG'.format(i) for i in dapg],
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
    samples_fluorescence = [FlowCal.stats.mean(s, channels='FL1')
                            for s in samples]

    dapg_color = '#ffc400'  # common color used for DAPG-related plots

    plt.figure(figsize=(3,3))
    plt.plot(dapg,
             samples_fluorescence,
             marker='o',
             color=dapg_color)

    # Illustrate min and max bounds. Because some of our control samples were
    # measured at a different cytometer gain setting and we aren't using MEF
    # calibration here, we will use the 0uM and 350uM DAPG concentration
    # samples instead.
    plt.axhline(samples_fluorescence[0],
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Min', x=2e2, y=2.0e1, ha='left', va='bottom', color='gray')
    plt.axhline(samples_fluorescence[-1],
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Max', x=-0.7, y=2.1e2, ha='left', va='top', color='gray')

    plt.yscale('log')
    plt.ylim((5e0,5e2))
    plt.xscale('symlog')
    plt.xlim((-1e0, 1e3))
    plt.xlabel(r'DAPG Concentration ($\mu M$)')
    plt.ylabel('FL1 Fluorescence (a.u.)')
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
    # calculated before cells are removed, though.) We again use the 0uM and
    # 350uM DAPG concentration samples as the min and max data in lieu of
    # controls. We also set `yscale` to 'log' because the cytometer used to
    # collect this data produces positive integer data (as opposed to
    # floating-point data, which can sometimes be negative), so the added
    # complexity of a logicle y-scale (which is the default) is not necessary.
    plt.figure(figsize=(4,3.5))
    FlowCal.plot.violin_dose_response(
        data=samples,
        channel='FL1',
        positions=dapg,
        min_data=samples[0],
        max_data=samples[-1],
        violin_kwargs={'facecolor':dapg_color,
                       'edgecolor':'black'},
        violin_width_to_span_fraction=0.075,
        xscale='log',
        yscale='log',
        ylim=(1e0,2e3))
    plt.xlabel(r'DAPG Concentration ($\mu M$)')
    plt.ylabel('FL1 Fluorescence (a.u.)')
    plt.tight_layout()
    plt.savefig('dose_response_violin.png', dpi=200)
    plt.close()

    print("\nDone.")
