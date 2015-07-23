#!/usr/bin/python
import gc
import os
import os.path

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
gated_plot_dir = 'plot_gated'.format(directory)

# Plot options
plot_gated = True

# Files
data_files = ['data.{:03d}'.format(i) for i in range(1,6)]

if __name__ == "__main__":
    # Check that directories exists, create if it doesn't.
    if not os.path.exists(gated_plot_dir):
        os.makedirs(gated_plot_dir)

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
    data_transf = [fc.transform.exponentiate(di, ch_all) for di in data]

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
                hist_params = {'div': 4, 'log': True, 'edgecolor': 'g'},
                savefig = '{}/{}.png'.format(gated_plot_dir, str(di)))
            pyplot.close()
            gc.collect()

    # Generate bar plot
    data_gfp = [numpy.median(di[:,'FL1-H']) for di in data_gated]
    xlabels = ['Sample {}'.format(i + 1) for i in range(len(data_gfp))]
    pyplot.figure(figsize = (7,3))
    fc.plot.bar(data_gfp, xlabels, ylabel = 'GFP (A.U.)', ylim = (0, 600),
        savefig = 'bar.png')
    