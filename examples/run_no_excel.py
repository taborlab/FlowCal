#!/usr/bin/python
import os
import os.path

import numpy
import scipy
from matplotlib import pyplot

import fc

# Directories
directory = 'FCFiles'
beads_plot_dir = 'plot_beads'
gated_plot_dir = 'plot_gated'

# Plot options
plot_gated = True

# Files
beads_file = 'data_006.fcs'
data_files = ['data_{:03d}.fcs'.format(i) for i in range(1,6)]

# Channels
sc_channels = ['FSC', 'SSC']
fl_channels = ['FL1', 'FL2', 'FL3']
# Colors for histograms
cm = fc.plot.load_colormap('spectral', 3)
hist_colors = dict(zip(fl_channels, cm[::-1]))

# MEF channels
mef_channels = ['FL1']
# MEF type used per channel
mef_names = {'FL1': 'Molecules of Equivalent Fluorescein, MEFL',
            }
# MEF bead values
mef_values = {'FL1': [numpy.nan, 792, 2079, 6588, 16471, 
                                                    47497, 137049, 271647],
             }

if __name__ == "__main__":
    # Check that directories exists, create if it doesn't.
    if not os.path.exists(beads_plot_dir):
        os.makedirs(beads_plot_dir)
    if not os.path.exists(gated_plot_dir):
        os.makedirs(gated_plot_dir)

    # Process beads data
    print "\nProcessing beads..."
    beads_data = fc.io.FCSData('{}/{}'.format(directory, beads_file))
    print "Beads file contains {} events.".format(beads_data.shape[0])
    # Trim
    beads_data = fc.gate.start_end(beads_data, num_start=250, num_end=100)
    beads_data = fc.gate.high_low(beads_data, sc_channels)
    # Density gate
    print "\nRunning density gate on beads data..."
    gated_beads_data, gate_contour = fc.gate.density2d(data = beads_data,
                                                    channels = sc_channels,
                                                    gate_fraction = 0.3)
    # Plot
    pyplot.figure(figsize = (6,4))
    fc.plot.density_and_hist(beads_data, gated_beads_data, 
        density_channels = sc_channels,
        hist_channels = fl_channels,
        gate_contour = gate_contour, 
        density_params = {'mode': 'scatter'}, 
        hist_params = {'ylim': (0, 1000), 'div': 4},
        savefig = '{}/density_hist_{}.png'.format(beads_plot_dir, beads_file))
    pyplot.close()

    # Obtain standard curve transformation
    print "\nCalculating standard curve..."
    peaks_mef = numpy.array([mef_values[chi] for chi in mef_channels])
    to_mef = fc.mef.get_transform_fxn(gated_beads_data, peaks_mef, 
                    cluster_method = 'gmm', 
                    cluster_channels = fl_channels,
                    mef_channels = mef_channels, verbose = True, 
                    plot = True, plot_dir = beads_plot_dir)


    # Process data files
    print "\nLoading data..."
    data = []
    for df in data_files:
        di = fc.io.FCSData('{}/{}'.format(directory, df))
        data.append(di)

        gain = di[:,'FL1'].channel_info[0]['pmt_voltage']
        print "{} ({} events, FL1 gain = {}).".format(str(di), 
            di.shape[0], gain)

    # Basic gating/trimming
    data = [fc.gate.start_end(di, num_start=250, num_end=100) for di in data]
    data = [fc.gate.high_low(di, sc_channels) for di in data]
    # Gating of fluorescence channels
    data = [fc.gate.high_low(di, mef_channels) for di in data]
    # Exponential transformation
    data_transf = [fc.transform.exponentiate(di, sc_channels) for di in data]

    # Transform to MEF
    print "\nTransforming to MEF..."
    data_transf = [to_mef(di, mef_channels) for di in data_transf]

    # Density gate
    print "\nRunning density gate on data files..."
    data_gated = []
    data_gated_contour = []
    for di in data_transf:
        print "{}...".format(str(di))
        di_gated, gate_contour = fc.gate.density2d(data = di,
                                            channels = sc_channels,
                                            gate_fraction = 0.2)
        data_gated.append(di_gated)
        data_gated_contour.append(gate_contour)

    # Plot
    xlabels = [mef_names[chi] for chi in mef_channels]
    if plot_gated:
        print "\nPlotting density diagrams and histograms of data"
        for di, dig, dgc in zip(data_transf, data_gated, data_gated_contour):
            print "{}...".format(str(di))
            # Create histogram parameters
            hist_params = []
            for chi in mef_channels:
                param = {}
                param['div'] = 4
                param['facecolor'] = hist_colors[chi]
                param['xlabel'] = mef_names[chi]
                param['log'] = True
                hist_params.append(param)
            # Plot
            fc.plot.density_and_hist(di, gated_data = dig, figsize = (7,7),
                density_channels = sc_channels,
                hist_channels = mef_channels, gate_contour = dgc, 
                density_params = {'mode': 'scatter', 'log': True}, 
                hist_params = hist_params,
                savefig = '{}/{}.png'.format(gated_plot_dir, str(di)))
            pyplot.close()

    # Generate bar plot
    print "\nGenerating bar plot..."

    labels = ['Sample {}'.format(i + 1) for i in range(len(data))]
    fc.plot.hist_and_bar(data_gated, channel = 'FL1', labels = labels,
        hist_params = {'log': True, 'div': 4,
                'xlabel': 'MEFL', 'ylim': (0, 400)},
        bar_params = {'ylabel': 'MEFL', 'ylim': (0, 40000)},
        bar_stats_func = numpy.median, savefig = 'hist_bar.png')

    print "\nDone."
