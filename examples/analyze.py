#!/usr/bin/python
import os
import os.path

import numpy as np
import matplotlib.pyplot as plt

import FlowCal

# Names of the .fcs files to process
beads_filename = 'FCFiles/Beads006.fcs'
samples_filenames = ['FCFiles/Data001.fcs',
                     'FCFiles/Data002.fcs',
                     'FCFiles/Data003.fcs',
                     'FCFiles/Data004.fcs',
                     'FCFiles/Data005.fcs',
                     ]

# Directories to save plots
beads_plot_dir = 'plot_beads'
samples_plot_dir = 'plot_samples'

# MEF bead values
mefl_values = [0, 646, 1704, 4827, 15991, 47609, 135896, 273006]

# IPTG concentration of each sample in micromolar
iptg = np.array([0, 81, 161, 318, 1000])

if __name__ == "__main__":

    # Check that plot directories exist, create if they don't.
    if not os.path.exists(beads_plot_dir):
        os.makedirs(beads_plot_dir)
    if not os.path.exists(samples_plot_dir):
        os.makedirs(samples_plot_dir)

    ###
    # Process Beads Files
    ###
    print("\nProcessing calibration beads...")
    # Load beads data
    print("Loading file \"{}\"...".format(beads_filename))
    beads_sample = FlowCal.io.FCSData(beads_filename)

    # Transform all channels to Units of Relative Fluorescence Intensity (RFI),
    # otherwise known as arbitrary units (a.u.)
    print("Performing data transformation...")
    beads_sample = FlowCal.transform.to_rfi(beads_sample)

    # Gating
    print("Performing gating...")

    # Remove first and last events. Transients in fluidics can make the first
    # few and last events slightly different from the rest.
    beads_sample_gated = FlowCal.gate.start_end(beads_sample,
                                                num_start=250,
                                                num_end=100)

    # Remove saturating events in forward/side scatter.
    beads_sample_gated = FlowCal.gate.high_low(beads_sample_gated,
                                               channels=['FSC','SSC'])

    # Density gating
    beads_sample_gated, __, gate_contour = FlowCal.gate.density2d(
        data=beads_sample_gated,
        channels=['FSC','SSC'],
        gate_fraction=0.3,
        xscale='logicle',
        yscale='logicle',
        sigma=5.,
        full_output=True)

    # Plot forward/side scatter density plot and fluorescence histograms
    print("Plotting density plot and histogram...")
    # Parameters for the forward/side scatter density plot
    density_params = {}
    density_params['mode'] = 'scatter'
    density_params['xscale'] = 'logicle'
    density_params['yscale'] = 'logicle'
    density_params['xlim'] = [90, 1023]
    density_params['ylim'] = [90, 1023]
    density_params['sigma'] = 5.
    # Parameters for the fluorescence histograms
    hist_params = {'xscale': 'logicle'}
    # Figure name
    figname = '{}/density_hist_{}.png'.format(beads_plot_dir, 'beads')
    # Make figure
    FlowCal.plot.density_and_hist(
        beads_sample,
        beads_sample_gated,
        density_channels=['FSC','SSC'],
        hist_channels=['FL1', 'FL3'],
        gate_contour=gate_contour, 
        density_params=density_params,
        hist_params=hist_params,
        savefig=figname)

    # Obtain standard curve transformation
    print("\nCalculating standard curve for channel FL1...")
    mef_transform_fxn = FlowCal.mef.get_transform_fxn(
        beads_sample_gated,
        mef_values=mefl_values,
        mef_channels='FL1',
        clustering_channels=['FL1', 'FL3'],
        verbose=True,
        plot=True,
        plot_dir=beads_plot_dir,
        plot_filename='beads')

    ###
    # Process Cell Sample Files
    ###
    print("\nProcessing cell samples...")

    # Initialize samples list
    samples = []
    for sample_id, sample_filename in enumerate(samples_filenames):
        # Load cell sample file
        print("\nLoading file \"{}\"...".format(sample_filename))
        sample = FlowCal.io.FCSData(sample_filename)

        # Transform
        print("Performing data transformation...")

        # Transform all channels to Units of Relative Fluorescence Intensity
        # (RFI), otherwise known as arbitrary units (a.u.)
        sample = FlowCal.transform.to_rfi(sample)
        # Transform Fluorescence channel FL1 to MEF
        sample = mef_transform_fxn(sample, channels=['FL1'])

        # Gating
        print("Performing gating...")

        # Remove first and last events. Transients in fluidics can make the
        # first few and last events slightly different from the rest.
        sample_gated = FlowCal.gate.start_end(sample,
                                              num_start=250,
                                              num_end=100)

        # Remove saturating events in forward/side scatter and FL1.
        sample_gated = FlowCal.gate.high_low(sample_gated,
                                             channels=['FSC','SSC','FL1'])

        # Density gating
        sample_gated, __, gate_contour = FlowCal.gate.density2d(
            data=sample_gated,
            channels=['FSC','SSC'],
            gate_fraction=0.5,
            xscale='logicle',
            yscale='logicle',
            full_output=True)

        # Plot forward/side scatter density plot and fluorescence histograms
        print("Plotting density plot and histogram...")
        # Parameters for the forward/side scatter density plot
        density_params = {}
        density_params['mode'] = 'scatter'
        density_params['xscale'] = 'logicle'
        density_params['yscale'] = 'logicle'
        # Parameters for the fluorescence histograms
        hist_params = {}
        hist_params['xscale'] = 'logicle'
        hist_params['xlabel'] = 'FL1 ' + \
            '(Molecules of Equivalent Fluorescein, MEFL)'
        # Figure name
        figname = '{}/density_hist_{}.png'.format(
            samples_plot_dir,
            'S{:03}'.format(sample_id + 1))
        # Make figure
        FlowCal.plot.density_and_hist(
            sample,
            sample_gated,
            density_channels=['FSC','SSC'],
            hist_channels=['FL1'],
            gate_contour=gate_contour,
            density_params=density_params,
            hist_params=hist_params,
            savefig=figname)

        # Save cell sample object
        samples.append(sample_gated)

    ###
    # Plot combined histograms and dose-response curves
    ###

    # Histogram of all samples
    plt.figure(figsize=(6,3.5))
    FlowCal.plot.hist1d(
        samples,
        channel=['FL1'],
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

    # Dose response curve
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
