#!/usr/bin/python
"""
FlowCal Python API example, without using calibration beads data.

This script is divided in two parts. Part one processes data from nine
cell samples and generates plots of each one. In addition, multi-color
compensation is performed on all samples using data from no-fluorophore and
single-fluorophore control samples (NFC and SFCs).

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
samples_filenames = ['FCFiles/sample029.fcs',
                     'FCFiles/sample030.fcs',
                     'FCFiles/sample031.fcs',
                     'FCFiles/sample032.fcs',
                     'FCFiles/sample033.fcs',
                     'FCFiles/sample034.fcs',
                     'FCFiles/sample035.fcs',
                     'FCFiles/sample036.fcs',
                     'FCFiles/sample037.fcs']
nfc_sample_filename = 'FCFiles/controls/nfc/sample004.fcs'
sfc1_sample_filename = 'FCFiles/controls/sfc1/sample007.fcs'
sfc2_sample_filename = 'FCFiles/controls/sfc2/sample019.fcs'

# aTc concentration of each cell sample, in ng/mL.
atc = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 7.5, 20])

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
                                             channels=['FSC','SSC','FL1','FL2'])

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
        hist_params = [{}, {}]
        hist_params[0]['xlabel'] = 'FL1 Fluorescence (a.u.)'
        hist_params[1]['xlabel'] = 'FL2 Fluorescence (a.u.)'

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
    print("Loading file \"{}\"...".format(nfc_sample_filename))
    nfc_sample = FlowCal.io.FCSData(nfc_sample_filename)
    print("Loading file \"{}\"...".format(sfc1_sample_filename))
    sfc1_sample = FlowCal.io.FCSData(sfc1_sample_filename)
    print("Loading file \"{}\"...".format(sfc2_sample_filename))
    sfc2_sample = FlowCal.io.FCSData(sfc2_sample_filename)

    print("Performing data transformation...")
    nfc_sample = FlowCal.transform.to_rfi(nfc_sample)
    sfc1_sample = FlowCal.transform.to_rfi(sfc1_sample)
    sfc2_sample = FlowCal.transform.to_rfi(sfc2_sample)

    print("Performing gating...")
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

    print("Plotting density plot and histogram...")
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
    print("\nPerforming multi-color compensation...")
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
    # We now show how to generate plots using the processed flow cytometry
    # data we just obtained.
    print("\nGenerating plots...")

    # Plot 1: Histogram of all samples
    #
    # Here, we plot the fluorescence histograms of all ten samples in the same
    # figure using ``FlowCal.plot.hist1d``. Note how this function can be used
    # in the context of accessory matplotlib functions to modify the axes
    # limits and labels and to add a legend, among other things.

    # Color each histogram according to its corresponding aTc concentration.
    # Use a perceptually uniform colormap (cividis), and transition among
    # colors using a logarithmic normalization, which comports with the
    # logarithmically spaced aTc concentrations.
    cmap = mpl.cm.get_cmap('cividis')
    norm = mpl.colors.LogNorm(vmin=1e0, vmax=20)
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
    plt.xlim((0,5e3))
    plt.xlabel('FL1 Fluorescence (a.u.)')
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
    plt.xlim((0,5e3))
    plt.xlabel('FL2 Fluorescence (a.u.)')
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
    # standard deviation. In this example, we calculate the mean from channel
    # FL1 of each sample and plot them against the corresponding aTc
    # concentrations.

    # Because some of our control samples were measured at a different cytometer
    # gain setting and we aren't using MEF calibration here, we will use the 0
    # and 20 ng/mL aTc concentration samples instead.
    samples_fl1 = [FlowCal.stats.mean(s, channels='FL1') for s in samples]
    samples_fl2 = [FlowCal.stats.mean(s, channels='FL2') for s in samples]
    # No fluorescence control (NFC) will give the minimum fluorescence level in
    # both channels. Single fluorescence controls (SFCs) containing sfGFP or
    # mCherry only will give the maximum levels in channels FL1 and FL2.
    min_fl1 = FlowCal.stats.mean(nfc_sample_gated, channels='FL1')
    max_fl1 = FlowCal.stats.mean(sfc1_sample_gated, channels='FL1')
    min_fl2 = FlowCal.stats.mean(nfc_sample_gated, channels='FL2')
    max_fl2 = FlowCal.stats.mean(sfc2_sample_gated, channels='FL2')

    plt.figure(figsize=(6,3))

    plt.subplot(1, 2, 1)
    plt.plot(atc,
             samples_fl1,
             marker='o',
             color='tab:green')
    plt.axhline(min_fl1,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Min', x=3e1, y=1.4e1, ha='left', va='bottom', color='gray')
    plt.axhline(max_fl1,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Max', x=-0.8, y=3.3e2, ha='left', va='top', color='gray')
    plt.yscale('log')
    plt.ylim((5e0, 5e2))
    plt.xscale('symlog')
    plt.xlim((-1e0, 1e2))
    plt.xlabel('aTc Concentration (ng/mL)')
    plt.ylabel('FL1 Fluorescence (a.u.)')

    plt.subplot(1, 2, 2)
    plt.plot(atc,
             samples_fl2,
             marker='o',
             color='tab:orange')
    plt.axhline(min_fl2,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Min', x=3e1, y=0.9e1, ha='left', va='bottom', color='gray')
    plt.axhline(max_fl2,
                color='gray',
                linestyle='--',
                zorder=-1)
    plt.text(s='Max', x=-0.8, y=5e2, ha='left', va='top', color='gray')
    plt.yscale('log')
    plt.ylim((4e0, 1.5e3))
    plt.xscale('symlog')
    plt.xlim((-1e0, 1e2))
    plt.xlabel('aTc Concentration (ng/mL)')
    plt.ylabel('FL2 Fluorescence (a.u.)')

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
        violin_kwargs={'facecolor':'tab:green',
                       'edgecolor':'black'},
        violin_width_to_span_fraction=0.075,
        xscale='log',
        yscale='log',
        ylim=(1e0,1e3))
    plt.xlabel('aTc Concentration (ng/mL)')
    plt.ylabel('FL1 Fluorescence (a.u.)')
    
    plt.subplot(1, 2, 2)
    FlowCal.plot.violin_dose_response(
        data=samples,
        channel='FL2',
        positions=atc,
        min_data=nfc_sample_gated,
        max_data=sfc2_sample_gated,
        violin_kwargs={'facecolor':'tab:orange',
                       'edgecolor':'black'},
        violin_width_to_span_fraction=0.075,
        xscale='log',
        yscale='log',
        ylim=(1e0,2e3))
    plt.xlabel('aTc Concentration (ng/mL)')
    plt.ylabel('FL2 Fluorescence (a.u.)')

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
        ylim=(-3e1, 1e3),
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
        ylim=(-3e1, 2e3),
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
