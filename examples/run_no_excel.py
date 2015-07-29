#!/usr/bin/python
import gc
import os
import os.path
import sys
sys.path.append('..')

import numpy
import scipy
from matplotlib import pyplot

import fc.io
import fc.gate
import fc.plot
import fc.transform
import fc.mef

# Directories
directory = 'FCFiles'
beads_plot_dir = 'plot_beads'
gated_plot_dir = 'plot_gated'

# Plot options
plot_gated = True

# Files
beads_file = 'data.006'
data_files = ['data.{:03d}'.format(i) for i in range(1,6)]

# MEF values
mefl = [numpy.nan, 792, 2079, 6588, 16471, 47497, 137049, 271647]

if __name__ == "__main__":
    # Check that directories exists, create if it doesn't.
    if not os.path.exists(beads_plot_dir):
        os.makedirs(beads_plot_dir)
    if not os.path.exists(gated_plot_dir):
        os.makedirs(gated_plot_dir)

    # Process beads data
    print "\nProcessing beads..."
    beads_data = fc.io.TaborLabFCSData('{}/{}'.format(directory, beads_file))
    print "Beads file contains {} events.".format(beads_data.shape[0])
    # Trim
    beads_data = fc.gate.start_end(beads_data, num_start=250, num_end=100)
    beads_data = fc.gate.high_low(beads_data, ['FSC-H', 'SSC-H'])
    # Density gate
    print "\nRunning density gate on beads data..."
    gated_beads_data, gate_contour = fc.gate.density2d(data = beads_data,
                                        gate_fraction = 0.3)
    # Plot
    pyplot.figure(figsize = (6,4))
    fc.plot.density_and_hist(beads_data, gated_beads_data, 
        density_channels = ['FSC-H', 'SSC-H'], 
        hist_channels = ['FL1-H', 'FL2-H', 'FL3-H'],
        gate_contour = gate_contour, 
        density_params = {'mode': 'scatter'}, 
        hist_params = {'ylim': (0, 1000), 'div': 4},
        savefig = '{}/density_hist_{}.png'.format(beads_plot_dir, beads_file))
    pyplot.close()
    gc.collect()

    # Obtain standard curve transformation
    print "\nCalculating standard curve..."
    peaks_mef = numpy.array(mefl)
    to_mef = fc.mef.get_transform_fxn(gated_beads_data, peaks_mef, 
                    cluster_method = 'gmm', 
                    cluster_channels = ['FL1-H', 'FL2-H', 'FL3-H'],
                    mef_channels = 'FL1-H', verbose = True, 
                    plot = True, plot_dir = beads_plot_dir)


    # Process data files
    print "\nLoading data..."
    data = []
    for df in data_files:
        di = fc.io.TaborLabFCSData('{}/{}'.format(directory, df))
        data.append(di)

        gain = di[:,'FL1-H'].channel_info[0]['pmt_voltage']
        print "{} ({} events, FL1-H gain = {}).".format(str(di), 
            di.shape[0], gain)

    # Basic gating/trimming
    ch_all = ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H']
    data = [fc.gate.start_end(di, num_start=250, num_end=100) for di in data]
    data = [fc.gate.high_low(di, ch_all) for di in data]
    # Exponential transformation
    data_transf = [fc.transform.exponentiate(di, ['FSC-H', 'SSC-H']) \
                                                                for di in data]

    # Transform to MEFL
    print "\nTransforming FL1-H to MEFL..."
    data_transf = [to_mef(di, 'FL1-H') for di in data_transf]

    # Density gate
    print "\nRunning density gate on data files..."
    data_gated = []
    data_gated_contour = []
    for di in data_transf:
        print "{}...".format(str(di))
        di_gated, gate_contour = fc.gate.density2d(data = di, 
                                            gate_fraction = 0.3)
        data_gated.append(di_gated)
        data_gated_contour.append(gate_contour)

    # Plot
    if plot_gated:
        print "\nPlotting density diagrams and histograms of data"
        for di, dig, dgc in zip(data_transf, data_gated, data_gated_contour):
            print "{}...".format(str(di))
            # Plot
            fc.plot.density_and_hist(di, gated_data = dig, figsize = (7,7),
                density_channels = ['FSC-H', 'SSC-H'], 
                hist_channels = ['FL1-H'], gate_contour = dgc, 
                density_params = {'mode': 'scatter', 'log': True}, 
                hist_params = {'div': 4, 'log': True, 'edgecolor': 'g',
                    'xlabel': 'Molecules of Equivalent Fluorescein (MEFL)'},
                savefig = '{}/{}.png'.format(gated_plot_dir, str(di)))
            pyplot.close()
            gc.collect()

    # Generate bar plot
    print "\nGenerating bar plots..."
    data_gfp = [numpy.median(di[:,'FL1-H']) for di in data_gated]
    xlabels = ['Sample {}'.format(i + 1) for i in range(len(data_gfp))]
    pyplot.figure(figsize = (7,3))
    fc.plot.bar(data_gfp, xlabels, ylabel = 'MEFL', ylim = (0, 40000),
        savefig = 'bar.png')
    