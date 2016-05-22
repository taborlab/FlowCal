#!/usr/bin/python
import os
import os.path

import numpy as np
import scipy
import matplotlib.pyplot as plt
import palettable

import FlowCal

# Directories
directory = 'FCFiles'
beads_plot_dir = 'plot_beads'
gated_plot_dir = 'plot_samples'

# Plot options
plot_gated = True

# Files
beads_file = 'data_006.fcs'
data_files = ['data_{:03d}.fcs'.format(i) for i in range(1,6)]

# Channels
sc_channels = ['FSC', 'SSC']
fl_channels = ['FL1', 'FL2', 'FL3']
# Colors for histograms
cm = palettable.colorbrewer.diverging.Spectral_8_r.mpl_colormap
hist_colors_list = [cm(i) for i in np.linspace(0,1,len(fl_channels))]
hist_colors = dict(zip(fl_channels, hist_colors_list))

# MEF channels
mef_channels = ['FL1']
# MEF type used per channel
mef_names = {'FL1': 'Molecules of Equivalent Fluorescein, MEFL',
            }
# MEF bead values
mef_values = {'FL1': [0, 792, 2079, 6588, 16471, 47497, 137049, 271647],
             }

if __name__ == "__main__":
    # Check that directories exists, create if they don't.
    if not os.path.exists(beads_plot_dir):
        os.makedirs(beads_plot_dir)
    if not os.path.exists(gated_plot_dir):
        os.makedirs(gated_plot_dir)

    # Load beads data
    print("\nProcessing beads...")
    beads_data = FlowCal.io.FCSData('{}/{}'.format(directory, beads_file))
    print("Beads file contains {} events.".format(beads_data.shape[0]))

    # Transform to Relative Fluorescence Units (rfi), commonly known as
    # arbitrary units (a.u.)
    beads_data = FlowCal.transform.to_rfi(beads_data,
                                          sc_channels + fl_channels)

    # Gating

    # Remove first and last events. Transients in fluidics can make the
    # first few and last events slightly different from the rest.
    beads_data = FlowCal.gate.start_end(beads_data, num_start=250, num_end=100)

    # Remove saturating events in forward/side scatter.
    beads_data = FlowCal.gate.high_low(beads_data, sc_channels)

    # Apply density gating
    print("\nRunning density gate on beads data...")
    gated_beads_data, __, gate_contour = FlowCal.gate.density2d(
        data=beads_data,
        channels=sc_channels,
        gate_fraction=0.3,
        full_output=True)

    # Plot forward/side scatter density plot and fluorescence histograms
    plt.figure(figsize = (6,4))
    FlowCal.plot.density_and_hist(
        beads_data,
        gated_beads_data, 
        density_channels=sc_channels,
        hist_channels=fl_channels,
        gate_contour=gate_contour, 
        density_params={'mode': 'scatter', 'xscale': 'log', 'yscale': 'log'},
        hist_params={'ylim': (0, 1000), 'xscale': 'log'},
        savefig='{}/density_hist_{}.png'.format(beads_plot_dir, beads_file))
    plt.close()

    # Obtain standard curve transformation
    print("\nCalculating standard curve...")
    peaks_mef = np.array([mef_values[chi] for chi in mef_channels])
    to_mef = FlowCal.mef.get_transform_fxn(
        gated_beads_data,
        peaks_mef,
        clustering_channels = fl_channels,
        mef_channels = mef_channels,
        verbose = True,
        plot = True,
        plot_dir = beads_plot_dir)


    # Load cell sample data
    print("\nLoading data...")
    data = []
    for df in data_files:
        di = FlowCal.io.FCSData('{}/{}'.format(directory, df))
        data.append(di)

        dv = di.detector_voltage('FL1')
        print("{} ({} events, FL1 voltage = {}).".format(
            str(di),
            di.shape[0],
            dv))

    # Transform to Relative Fluorescence Units (rfi), commonly known as
    # arbitrary units (a.u.)
    data_transf = [FlowCal.transform.to_rfi(di, sc_channels + mef_channels)
                   for di in data]

    # Transform specified fluorescence channels to MEF
    print("\nTransforming to MEF...")
    data_transf = [to_mef(di, mef_channels) for di in data_transf]

    # Gate

    # Remove first and last events. Transients in fluidics can make the
    # first few and last events slightly different from the rest.
    data_gated = [FlowCal.gate.start_end(di, num_start=250, num_end=100)
            for di in data_transf]

    # Remove saturating events in forward/side scatter and fluorescence
    # channels.
    data_gated = [FlowCal.gate.high_low(di, sc_channels + mef_channels)
            for di in data_gated]

    # Density gate
    print("\nRunning density gate on data files...")
    data_dg = []
    data_dg_contour = []
    for di in data_gated:
        print("{}...".format(str(di)))
        di_gated, __, gate_contour = FlowCal.gate.density2d(
            data=di,
            channels=sc_channels,
            xscale='log',
            yscale='log',
            gate_fraction=0.2,
            full_output=True)
        data_dg.append(di_gated)
        data_dg_contour.append(gate_contour)

    # Plot
    xlabels = [mef_names[chi] for chi in mef_channels]
    if plot_gated:
        print("\nPlotting density diagrams and histograms of data")
        for di, dig, dgc in zip(data_transf, data_dg, data_dg_contour):
            print("{}...".format(str(di)))
            # Create histogram parameters
            hist_params = []
            for chi in mef_channels:
                param = {}
                param['facecolor'] = hist_colors[chi]
                param['xlabel'] = mef_names[chi]
                param['xscale'] = 'log'
                hist_params.append(param)
            # Plot
            FlowCal.plot.density_and_hist(di,
                gated_data=dig,
                figsize=(7,7),
                density_channels=sc_channels,
                hist_channels=mef_channels,
                gate_contour=dgc, 
                density_params={'mode': 'scatter',
                                'xscale': 'log',
                                'yscale': 'log'},
                hist_params=hist_params,
                savefig='{}/{}.png'.format(gated_plot_dir, str(di)))
            plt.close()

    print("\nDone.")
