#!/usr/bin/python
"""
FlowCal Python API example, using calibration beads data.

This script is divided in three parts. Part one loads and processes
calibration beads data from FCS files in order to obtain transformation
functions that can later be used to convert cell fluorescence to units of
Molecules of Equivalent Fluorophore (MEF). At several points in this
process, plots are generated that give insight into the relevant steps.

Part two processes data from twelve cell samples and uses the MEF
transformation functions from part one to convert fluorescence of these
samples to MEF. Plots are also generated in this stage. In addition, we perform
multi-color compensation on all samples using data from no-fluorophore and
single-fluorophore control samples (NFC and SFCs).

Part three exemplifies how to use the processed cell sample data with
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

# Name of the FCS files containing calibration beads data. The no-fluorophore
# control (NFC) and single-fluorophore controls (SFC) were measured on separate
# days with their own beads samples, and one control was measured with a
# different cytometer gain setting. However, we can still compare these to our
# samples because we are calibrating all fluorescence measurements to MEF units.
beads_filename     = 'FCFiles/sample001.fcs'
nfc_beads_filename = 'FCFiles/controls/sample003.fcs'
sfc1_beads_filename = 'FCFiles/controls/sample002.fcs'
sfc2_beads_filename = 'FCFiles/controls/sample001.fcs'

# Names of the FCS files containing data from cell samples
samples_filenames = ['FCFiles/sample029.fcs',
                     'FCFiles/sample030.fcs',
                     'FCFiles/sample031.fcs',
                     'FCFiles/sample032.fcs',
                     'FCFiles/sample033.fcs',
                     'FCFiles/sample034.fcs',
                     'FCFiles/sample035.fcs',
                     'FCFiles/sample036.fcs',
                     'FCFiles/sample037.fcs']
nfc_sample_filename = 'FCFiles/controls/sample004.fcs'
sfc1_sample_filename = 'FCFiles/controls/sample010.fcs'
sfc2_sample_filename = 'FCFiles/controls/sample019.fcs'

# Fluorescence values of each bead subpopulation, in MEF.
# These values should be taken from the datasheet provided by the bead
# manufacturer. We take Molecules of Equivalent Fluorescein (MEFL) to calibrate
# the FL1 (GFP) channel.
mefl_values     = np.array([0, 792, 2079, 6588, 16471, 47497, 137049, None])
nfc_mefl_values = np.array([0, 792, 2079, 6588, 16471, 47497, 137049, None])
sfc1_mefl_values = np.array([0, 792, 2079, 6588, 16471, 47497, 137049, 271647])
sfc2_mefl_values = np.array([0, 792, 2079, 6588, 16471, 47497, 137049, None])

mepe_values     = np.array([0, 531, 1504, 4819, 12506, 36159, 109588, 250892])
nfc_mepe_values = np.array([0, 531, 1504, 4819, 12506, 36159, 109588, 250892])
sfc1_mepe_values = np.array([0, 531, 1504, 4819, 12506, 36159, 109588, 250892])
sfc2_mepe_values = np.array([0, 531, 1504, 4819, 12506, 36159, 109588, 250892])

# aTc concentration of each cell sample, in ng/mL.
atc = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 7.5, 20])

# Plots will be generated at various stages of analysis. The following are the
# names of the folders in which we will store these plots.
beads_plot_dir   = 'plot_beads'
samples_plot_dir = 'plot_samples'

if __name__ == "__main__":

    # Check that plot directories exist, create if they do not.
    if not os.path.exists(beads_plot_dir):
        os.makedirs(beads_plot_dir)
    if not os.path.exists(samples_plot_dir):
        os.makedirs(samples_plot_dir)

    ###
    # Part 1: Processing calibration beads data
    ###
    print("\nProcessing calibration beads...")

    # Load calibration beads data from the corresponding FCS file.
    # ``FlowCal.io.FCSData(filename)`` returns an object that represents flow
    # cytometry data loaded from file ``filename``.
    print("Loading file \"{}\"...".format(beads_filename))
    beads_sample     = FlowCal.io.FCSData(beads_filename)
    nfc_beads_sample = FlowCal.io.FCSData(nfc_beads_filename)
    sfc1_beads_sample = FlowCal.io.FCSData(sfc1_beads_filename)
    sfc2_beads_sample = FlowCal.io.FCSData(sfc2_beads_filename)

    # Data loaded from an FCS file is in "Channel Units", the raw numbers
    # reported from the instrument's detectors. The FCS file also contains
    # information to convert these into Relative Fluorescence Intensity (RFI)
    # values, commonly referred to as arbitrary fluorescence units (a.u.).
    # The function ``FlowCal.transform.to_rfi()`` performs this conversion.
    print("Performing data transformation...")
    beads_sample     = FlowCal.transform.to_rfi(beads_sample)
    nfc_beads_sample = FlowCal.transform.to_rfi(nfc_beads_sample)
    sfc1_beads_sample = FlowCal.transform.to_rfi(sfc1_beads_sample)
    sfc2_beads_sample = FlowCal.transform.to_rfi(sfc2_beads_sample)

    # Gating

    # Gating is the process of removing measurements of irrelevant particles,
    # while retaining only the population of interest.
    print("Performing gating...")

    # ``FlowCal.gate.start_end()`` removes the first and last few events.
    # Transients in fluidics can make these events slightly different from the
    # rest. This may not be necessary in all instruments.
    beads_sample_gated = FlowCal.gate.start_end(beads_sample,
                                                num_start=250,
                                                num_end=100)
    nfc_beads_sample_gated = FlowCal.gate.start_end(nfc_beads_sample,
                                                    num_start=250,
                                                    num_end=100)
    sfc1_beads_sample_gated = FlowCal.gate.start_end(sfc1_beads_sample,
                                                     num_start=250,
                                                     num_end=100)
    sfc2_beads_sample_gated = FlowCal.gate.start_end(sfc2_beads_sample,
                                                     num_start=250,
                                                     num_end=100)

    # ``FlowCal.gate.high_low()`` removes events outside a range specified by
    # a ``low`` and a ``high`` value. If these are not specified (as shown
    # below), the function removes events outside the channel's range of
    # detection.
    # Detectors in a flow cytometer have a finite range of detection. If the
    # fluorescence of a particle is higher than the upper limit of this range,
    # the instrument will incorrectly record it with a value equal to this
    # limit. The same happens for fluorescence values lower than the lower limit
    # of detection. These saturated events should be removed, otherwise
    # statistics may be calculated incorrectly.
    # Note that this might not be necessary with newer instruments that record
    # data as floating-point numbers (and in fact it might eliminate negative
    # events). To see the data type stored in your FCS files, run the following
    # instruction: ``print beads_sample_gated.data_type``.
    # For beads, we only remove saturated events on the forward/side scatter
    # channels.
    beads_sample_gated = FlowCal.gate.high_low(beads_sample_gated,
                                               channels=['FSC','SSC'])
    nfc_beads_sample_gated = FlowCal.gate.high_low(nfc_beads_sample_gated,
                                                   channels=['FSC','SSC'])
    sfc1_beads_sample_gated = FlowCal.gate.high_low(sfc1_beads_sample_gated,
                                                    channels=['FSC','SSC'])
    sfc2_beads_sample_gated = FlowCal.gate.high_low(sfc2_beads_sample_gated,
                                                    channels=['FSC','SSC'])

    # ``FlowCal.gate.density2d()`` preserves only the densest population as
    # seen in a 2D density diagram of two channels. This helps remove particle
    # aggregations and other sparse populations that are not of interest (i.e.
    # debris).
    # We use the forward and side scatter channels and preserve 85% of the
    # events. Since bead populations form a very narrow cluster in these
    # channels, we use a smoothing factor (``sigma``) lower than the default
    # (10). Finally, setting ``full_output=True`` instructs the function to
    # return additional outputs in the form of a named tuple. ``gate_contour``
    # is a curve surrounding the gated region, which we will use for plotting
    # later.
    density_gate_output = FlowCal.gate.density2d(
        data=beads_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.85,
        sigma=5.,
        full_output=True)
    beads_sample_gated = density_gate_output.gated_data
    gate_contour       = density_gate_output.contour

    density_gate_output = FlowCal.gate.density2d(
        data=nfc_beads_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.85,
        sigma=5.,
        full_output=True)
    nfc_beads_sample_gated = density_gate_output.gated_data
    nfc_gate_contour       = density_gate_output.contour

    density_gate_output = FlowCal.gate.density2d(
        data=sfc1_beads_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.85,
        sigma=5.,
        full_output=True)
    sfc1_beads_sample_gated = density_gate_output.gated_data
    sfc1_gate_contour       = density_gate_output.contour

    density_gate_output = FlowCal.gate.density2d(
        data=sfc2_beads_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.85,
        sigma=5.,
        full_output=True)
    sfc2_beads_sample_gated = density_gate_output.gated_data
    sfc2_gate_contour       = density_gate_output.contour

    # Plot forward/side scatter 2D density plot and 1D fluorescence histograms
    print("Plotting density plot and histogram...")

    # Parameters for the forward/side scatter density plot
    density_params = {}
    # We use the "scatter" mode, in which individual particles will be plotted
    # individually as in a scatter plot, but with a color proportional to the
    # particle density around.
    density_params['mode'] = 'scatter'
    # We use short axis limits and a low smoothing factor, as our calibration
    # beads form a very narrow cluster.
    density_params['xlim'] = [90, 1023]
    density_params['ylim'] = [90, 1023]
    density_params['sigma'] = 5.

    # Plot filename
    # The figure can be saved in any format supported by matplotlib (svg, jpg,
    # etc.) by just changing the extension.
    plot_filename     = '{}/density_hist_{}.png'.format(beads_plot_dir,
                                                        'beads')
    nfc_plot_filename = '{}/nfc_density_hist_{}.png'.format(beads_plot_dir,
                                                            'beads')
    sfc1_plot_filename = '{}/sfc1_density_hist_{}.png'.format(beads_plot_dir,
                                                            'beads')
    sfc2_plot_filename = '{}/sfc2_density_hist_{}.png'.format(beads_plot_dir,
                                                            'beads')

    # Plot and save
    # The function ``FlowCal.plot.density_and_hist()`` plots a combined figure
    # with a 2D density plot at the top and an arbitrary number of 1D
    # histograms below. In this case, we will plot the forward/side scatter
    # channels in the density plot and the fluorescence channels FL1 and FL2
    # below as two separate histograms.
    # Note that we are providing data both before (``beads_sample``) and after
    # (``beads_sample_gated``) gating. Each 1D histogram will display the
    # ungated dataset with transparency and the gated dataset in front with a
    # solid color. In addition, we are providing ``gate_contour`` from the
    # density gating step, which will be displayed in the density diagram. This
    # will result in a convenient representation of the data both before and
    # after gating.
    FlowCal.plot.density_and_hist(
        beads_sample,
        beads_sample_gated,
        density_channels=['FSC', 'SSC'],
        hist_channels=['FL1', 'FL2'],
        gate_contour=gate_contour,
        density_params=density_params,
        savefig=plot_filename)
    FlowCal.plot.density_and_hist(
        nfc_beads_sample,
        nfc_beads_sample_gated,
        density_channels=['FSC', 'SSC'],
        hist_channels=['FL1', 'FL2'],
        gate_contour=nfc_gate_contour,
        density_params=density_params,
        savefig=nfc_plot_filename)
    FlowCal.plot.density_and_hist(
        sfc1_beads_sample,
        sfc1_beads_sample_gated,
        density_channels=['FSC', 'SSC'],
        hist_channels=['FL1', 'FL2'],
        gate_contour=sfc1_gate_contour,
        density_params=density_params,
        savefig=sfc1_plot_filename)
    FlowCal.plot.density_and_hist(
        sfc2_beads_sample,
        sfc2_beads_sample_gated,
        density_channels=['FSC', 'SSC'],
        hist_channels=['FL1', 'FL2'],
        gate_contour=sfc2_gate_contour,
        density_params=density_params,
        savefig=sfc2_plot_filename)

    # Use beads data to obtain a MEF transformation function
    print("\nCalculating standard curve for channel FL1...")

    # ``FlowCal.mef.get_transform_fxn()`` generates a transformation function
    # that converts fluorescence from relative fluorescence units (RFI) to MEF.
    # This function uses bead data from ``beads_sample_gated``. We generate a
    # MEF transformation function for channels FL1 and FL3, with corresponding
    # MEF fluorescence values specified by the arrays ``mefl_values`` and
    # ``mepe_values``. By default, clustering (subpopulation recognition) will
    # be performed using information from both channels. We also enable the
    # ``verbose`` mode, which prints information of each step. Finally, we
    # instruct the function to generate plots of each step in the folder
    # specified in ``beads_plot_dir`` with the suffix "beads".
    mef_transform_fxn = FlowCal.mef.get_transform_fxn(
        beads_sample_gated,
        mef_channels=['FL1', 'FL2'],
        mef_values=[mefl_values, mepe_values],
        verbose=True,
        plot=True,
        plot_dir=beads_plot_dir,
        plot_filename='beads')
    nfc_mef_transform_fxn = FlowCal.mef.get_transform_fxn(
        nfc_beads_sample_gated,
        mef_channels=['FL1', 'FL2'],
        mef_values=[nfc_mefl_values, nfc_mepe_values],
        verbose=True,
        plot=True,
        plot_dir=beads_plot_dir,
        plot_filename='nfc_beads')
    sfc1_mef_transform_fxn = FlowCal.mef.get_transform_fxn(
        sfc1_beads_sample_gated,
        mef_channels=['FL1', 'FL2'],
        mef_values=[sfc1_mefl_values, sfc1_mepe_values],
        verbose=True,
        plot=True,
        plot_dir=beads_plot_dir,
        plot_filename='sfc1_beads')
    sfc2_mef_transform_fxn = FlowCal.mef.get_transform_fxn(
        sfc2_beads_sample_gated,
        mef_channels=['FL1', 'FL2'],
        mef_values=[sfc2_mefl_values, sfc2_mepe_values],
        verbose=True,
        plot=True,
        plot_dir=beads_plot_dir,
        plot_filename='sfc2_beads')

    ###
    # Part 2: Processing cell sample data
    ###
    print("\nProcessing cell samples...")

    # We will use the list ``samples`` to store processed, transformed flow
    # cytometry data of each sample.
    samples = []

    # Iterate over cell sample filenames
    for sample_id, sample_filename in enumerate(samples_filenames):

        # Load sample data from the corresponding FCS file.
        print("\nLoading file \"{}\"...".format(sample_filename))
        sample = FlowCal.io.FCSData(sample_filename)
        
        # Perform transformations
        print("Performing data transformation...")

        # Transform data to RFI, as done previously with beads.
        sample = FlowCal.transform.to_rfi(sample)

        # We now use the transformation function obtained from calibration beads
        # to transform FL1 data from RFI to MEF.
        sample = mef_transform_fxn(sample, channels=['FL1', 'FL2'])

        # Gating
        print("Performing gating...")

        # Remove the first and last few events, as done previously with beads.
        sample_gated = FlowCal.gate.start_end(sample,
                                              num_start=250,
                                              num_end=100)

        # Remove saturated events
        # We only do this for the forward/side scatter channels and for
        # fluorescence channel FL1.
        sample_gated = FlowCal.gate.high_low(sample_gated,
                                             channels=['FSC','SSC','FL1','FL2'])

        # Apply density gating on the forward/side scatter channels. Preserve
        # 85% of the events. Return also a contour around the gated region.
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
        density_params['mode'] = 'scatter'

        # Parameters for the fluorescence histograms
        hist_params = [{}, {}]
        hist_params[0]['xlabel'] = 'FL1 ' + \
            '(Molecules of Equivalent Fluorescein, MEFL)'
        hist_params[1]['xlabel'] = 'FL2 ' + \
            '(Molecules of Equivalent PE, MEPE)'

        # Plot filename
        # The figure can be saved in any format supported by matplotlib (svg,
        # jpg, etc.) by just changing the extension.
        plot_filename = '{}/density_hist_{}.png'.format(
            samples_plot_dir,
            'S{:03}'.format(sample_id + 1))

        # Plot and save
        # In this case, we will plot the forward/side scatter channels in
        # the density plot and a histogram of the fluorescence channel FL1
        # below.
        FlowCal.plot.density_and_hist(
            sample,
            sample_gated,
            density_channels=['FSC','SSC'],
            hist_channels=['FL1','FL2'],
            gate_contour=gate_contour,
            density_params=density_params,
            hist_params=hist_params,
            savefig=plot_filename)

        # Save cell sample object
        samples.append(sample_gated)

    # Now, process the nfc and sfc control samples
    print("\nProcessing control samples...")
    # Load, transform, and gate control samples
    nfc_sample = FlowCal.io.FCSData(nfc_sample_filename)
    sfc1_sample = FlowCal.io.FCSData(sfc1_sample_filename)
    sfc2_sample = FlowCal.io.FCSData(sfc2_sample_filename)

    nfc_sample = FlowCal.transform.to_rfi(nfc_sample)
    sfc1_sample = FlowCal.transform.to_rfi(sfc1_sample)
    sfc2_sample = FlowCal.transform.to_rfi(sfc2_sample)

    nfc_sample = nfc_mef_transform_fxn(nfc_sample, channels=['FL1','FL2'])
    sfc1_sample = sfc1_mef_transform_fxn(sfc1_sample, channels=['FL1','FL2'])
    sfc2_sample = sfc2_mef_transform_fxn(sfc2_sample, channels=['FL1','FL2'])

    nfc_sample_gated = FlowCal.gate.start_end(nfc_sample,
                                              num_start=250,
                                              num_end=100)
    sfc1_sample_gated = FlowCal.gate.start_end(sfc1_sample,
                                               num_start=250,
                                               num_end=100)
    sfc2_sample_gated = FlowCal.gate.start_end(sfc2_sample,
                                               num_start=250,
                                               num_end=100)

    nfc_sample_gated = FlowCal.gate.high_low(nfc_sample_gated,
                                             channels=['FSC','SSC','FL1','FL2'])
    sfc1_sample_gated = FlowCal.gate.high_low(sfc1_sample_gated,
                                              channels=['FSC','SSC','FL1','FL2'])
    sfc2_sample_gated = FlowCal.gate.high_low(sfc2_sample_gated,
                                              channels=['FSC','SSC','FL1','FL2'])

    density_gate_output = FlowCal.gate.density2d(
        data=nfc_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.85,
        full_output=True)
    nfc_sample_gated = density_gate_output.gated_data
    nfc_gate_contour = density_gate_output.contour

    density_gate_output = FlowCal.gate.density2d(
        data=sfc1_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.85,
        full_output=True)
    sfc1_sample_gated = density_gate_output.gated_data
    sfc1_gate_contour = density_gate_output.contour

    density_gate_output = FlowCal.gate.density2d(
        data=sfc2_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.85,
        full_output=True)
    sfc2_sample_gated = density_gate_output.gated_data
    sfc2_gate_contour = density_gate_output.contour

    # Plot and save
    nfc_plot_filename = '{}/density_hist_nfc.png'.format(samples_plot_dir)
    sfc1_plot_filename = '{}/density_hist_sfc1.png'.format(samples_plot_dir)
    sfc2_plot_filename = '{}/density_hist_sfc2.png'.format(samples_plot_dir)

    FlowCal.plot.density_and_hist(
        nfc_sample,
        nfc_sample_gated,
        density_channels=['FSC','SSC'],
        hist_channels=['FL1','FL2'],
        gate_contour=nfc_gate_contour,
        density_params=density_params,
        hist_params=hist_params,
        savefig=nfc_plot_filename)
    FlowCal.plot.density_and_hist(
        sfc1_sample,
        sfc1_sample_gated,
        density_channels=['FSC','SSC'],
        hist_channels=['FL1','FL2'],
        gate_contour=sfc1_gate_contour,
        density_params=density_params,
        hist_params=hist_params,
        savefig=sfc1_plot_filename)
    FlowCal.plot.density_and_hist(
        sfc2_sample,
        sfc2_sample_gated,
        density_channels=['FSC','SSC'],
        hist_channels=['FL1','FL2'],
        gate_contour=sfc2_gate_contour,
        density_params=density_params,
        hist_params=hist_params,
        savefig=sfc2_plot_filename)

    # Perform multi-color compensation
    # ``FlowCal.compensate.get_transform_fxn()`` generates a transformation
    # function that performs multi-color compensation on a specified set of
    # channels in order to remove fluorophore bleedthrough.
    # This function requires data from single-fluorophore controls (SFCs), one
    # per channel to compensate, each from cells containing only one
    # fluorophore. This function can optionally use data from a no-fluorophore
    # control (NFC).
    compensation_fxn = FlowCal.compensate.get_transform_fxn(
        nfc_sample=nfc_sample_gated,
        sfc_samples=[sfc1_sample_gated, sfc2_sample_gated],
        comp_channels=['FL1', 'FL2'],
    )
    # Compensate all samples
    samples_compensated = [compensation_fxn(s, ['FL1', 'FL2']) for s in samples]
    nfc_sample_compensated = compensation_fxn(nfc_sample_gated, ['FL1', 'FL2'])
    sfc1_sample_compensated = compensation_fxn(sfc1_sample_gated, ['FL1', 'FL2'])
    sfc2_sample_compensated = compensation_fxn(sfc2_sample_gated, ['FL1', 'FL2'])

    ###
    # Part 3: Examples on how to use processed cell sample data
    ###

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
    FlowCal.plot.hist1d(samples,
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
    FlowCal.plot.hist1d(samples,
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
    samples_fl1 = [FlowCal.stats.mean(s, channels='FL1') for s in samples]
    samples_fl2 = [FlowCal.stats.mean(s, channels='FL2') for s in samples]
    # No fluorescence control (NFC) will give the minimum fluorescence level in
    # both channels. Single fluorescence controls (SFCs) containing sfGFP or
    # mCherry only will give the maximum levels in channels FL1 and FL2.
    min_fl1 = FlowCal.stats.mean(nfc_sample_gated, channels='FL1')
    max_fl1 = FlowCal.stats.mean(sfc1_sample_gated, channels='FL1')
    min_fl2 = FlowCal.stats.mean(nfc_sample_gated, channels='FL2')
    max_fl2 = FlowCal.stats.mean(sfc2_sample_gated, channels='FL2')

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
        data=samples,
        channel='FL1',
        positions=atc,
        min_data=nfc_sample_gated,
        max_data=sfc1_sample_gated,
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
        data=samples,
        channel='FL2',
        positions=atc,
        min_data=nfc_sample_gated,
        max_data=sfc2_sample_gated,
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
    # Here, we repeat the previous violin plot but using compensated data.
    # y axis will now be plotted in ``logicle`` scale since histograms will
    # be centered around zero due to compensation.
    plt.figure(figsize=(8, 3.5))

    plt.subplot(1, 2, 1)
    FlowCal.plot.violin_dose_response(
        data=samples_compensated,
        channel='FL1',
        positions=atc,
        min_data=nfc_sample_compensated,
        max_data=sfc1_sample_compensated,
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
        data=samples_compensated,
        channel='FL2',
        positions=atc,
        min_data=nfc_sample_compensated,
        max_data=sfc2_sample_compensated,
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
