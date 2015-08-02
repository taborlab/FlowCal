#!/usr/bin/python
import gc
import os
import os.path
import collections
from platform import system as platformSys
from subprocess import call

import numpy
import scipy
from matplotlib import pyplot

from Tkinter import Tk
from tkFileDialog import askopenfilename

import fc.io
import fc.excel_io
import fc.gate
import fc.plot
import fc.transform
import fc.mef
import fc.stats

def main():
    # Launch dialogue to select input file
    Tk().withdraw() # don't show main window
    # OSX ONLY: Call bash script to prevent file select window from sticking 
    # after use.
    if platformSys() == 'Darwin':
        call("defaults write org.python.python ApplePersistenceIgnoreState YES", 
            shell=True)
        input_form = askopenfilename(filetypes = [('Excel files', '*.xlsx')])
        call("defaults write org.python.python ApplePersistenceIgnoreState NO", 
            shell=True)
    else:
        input_form = askopenfilename(filetypes = [('Excel files', '*.xlsx')])
    if not input_form:
        print "Cancelled."
        return

    # Get base directory
    basedir, input_file = os.path.split(input_form)
    input_filename, __ = os.path.splitext(input_file)

    # Generate plotting directories
    beads_plot_dir = "{}/{}".format(basedir, 'plot_beads')
    gated_plot_dir = "{}/{}".format(basedir, 'plot_gated')
    if not os.path.exists(beads_plot_dir):
        os.makedirs(beads_plot_dir)
    if not os.path.exists(gated_plot_dir):
        os.makedirs(gated_plot_dir)

    # Generate path of output file
    output_form = "{}/{}".format(basedir, input_filename + '_output.xlsx')

    # Get beads files data from input form
    beads_info = fc.excel_io.import_rows(input_form, "beads")
    to_mef_all = {}

    print "\nProcessing beads..."
    for bi in beads_info:
        bid = bi['File Path']
        
        # Open file
        di = fc.io.TaborLabFCSData("{}/{}".format(basedir, bi['File Path']))
        print "{} ({} events).".format(str(di), di.shape[0])

        # Extract channels used for clustering
        if 'Clustering Channels' in bi:
            cluster_channels = bi['Clustering Channels'].split(',')
            cluster_channels = [cc.strip() for cc in cluster_channels]
        else:
            cluster_channels = ['FL1-H', 'FL2-H', 'FL3-H']

        # Trim
        di = fc.gate.start_end(di, num_start=250, num_end=100)
        di = fc.gate.high_low(di, ['FSC-H', 'SSC-H'])
        # Density gate
        print "Running density gate (fraction = {:.2f})..."\
            .format(float(bi['Gate Fraction']))
        gated_di, gate_contour = fc.gate.density2d(data = di,
                                gate_fraction = float(bi['Gate Fraction']))
        # Plot
        pyplot.figure(figsize = (6,4))
        fc.plot.density_and_hist(di, gated_di, 
            density_channels = ['FSC-H', 'SSC-H'], 
            hist_channels = cluster_channels,
            gate_contour = gate_contour, 
            density_params = {'mode': 'scatter'}, 
            hist_params = {'div': 4},
            savefig = '{}/density_hist_{}.png'.format(beads_plot_dir, di))

        # Process MEF values
        mef = []
        mef_channels = []
        for channel in ['FL1-H','FL2-H','FL3-H']:
            if channel+' Peaks' in bi:
                peaks = bi[channel+' Peaks'].split(',')
                peaks = [int(e) if e.strip().isdigit() else numpy.nan \
                    for e in peaks]
                mef.append(peaks)
                mef_channels.append(channel)
        mef = numpy.array(mef)

        # Obtain standard curve transformation
        print "\nCalculating standard curve..."
        to_mef = fc.mef.get_transform_fxn(gated_di, mef, 
                        cluster_method = bi['Clustering Method'], 
                        cluster_channels = cluster_channels,
                        mef_channels = mef_channels, verbose = True, 
                        plot = True, plot_dir = beads_plot_dir)

        # Save MEF transformation function
        to_mef_all[bid] = to_mef

    # Process data files
    print "\nLoading data..."
    # Get beads files data from input form
    cells_info = fc.excel_io.import_rows(input_form, "cells")

    # Load data files
    data = []
    for ci in cells_info:
        di = fc.io.TaborLabFCSData("{}/{}".format(basedir, ci['File Path']),
            ci)
        data.append(di)

        gain = di[:,'FL1-H'].channel_info[0]['pmt_voltage']
        print "{} ({} events, FL1-H gain = {}).".format(str(di), 
            di.shape[0], gain)

    # Parse transforms to conduct on data
    # transforms is an array of dictionaries.
    # Each dictionary contains the transformation type for each channel.
    transforms = []
    for di in data:
        channel_transf = collections.OrderedDict()
        for channel in ['FL1-H', 'FL2-H', 'FL3-H']:
            if channel + ' Transform' not in di.metadata:
                pass
            elif di.metadata[channel + ' Transform'] == '':
                pass
            elif di.metadata[channel + ' Transform'] == 'None':
                channel_transf[channel] = 'None'
            elif di.metadata[channel + ' Transform'] == 'Exponential':
                channel_transf[channel] = 'Exponential'
            elif di.metadata[channel + ' Transform'] == 'Mef':
                channel_transf[channel] = 'Mef'
            else:
                raise ValueError("{} not recognized for channel {}.".format(
                    di.metadata[channel + ' Transform'], channel))
        transforms.append(channel_transf)

    # Trim data
    print "\nTrimming data..."    
    data_trimmed = [fc.gate.start_end(di, num_start=250, num_end=100)\
                                                     for di in data]
    data_trimmed = [fc.gate.high_low(di, ['FSC-H', 'SSC-H'] + tf.keys())\
        for di, tf in zip(data_trimmed, transforms)]

    # Transform data
    print "\nTransforming data..."
    data_transf = []
    for di, tf in zip(data_trimmed, transforms):
        # Exponential transformation is applied by default to FSC and SSC
        dt = fc.transform.exponentiate(di, ['FSC-H', 'SSC-H'])
        # Print transformations used in fluorescence channels
        print str(di)+' ('+', '.join([k+': '+c for k, c in tf.iteritems()])+')'
        # Transform fluorescence channels
        for channel, transform in tf.iteritems():
            if transform == 'None':
                pass
            elif transform == 'Exponential':
                dt = fc.transform.exponentiate(dt, [channel])
            elif transform == 'Mef':
                to_mef = to_mef_all[dt.metadata['Beads File Path']]
                if channel not in to_mef.keywords['sc_channels']:
                    raise ValueError("Beads do not contain peaks for channel")
                dt = to_mef(dt, channel)
            else:
                print "Unexpected input for " + channel + ",",
        data_transf.append(dt)
        
    print '\nGating data...'
    data_gated = []
    data_gated_contour = []
    for di in data_transf:
        print "{} (gate fraction = {:.2f})...".format(str(di), 
                float(di.metadata['Gate Fraction']))
        di_gated, gate_contour = fc.gate.density2d(data = di, 
                        gate_fraction = float(di.metadata['Gate Fraction']))

        data_gated.append(di_gated)
        data_gated_contour.append(gate_contour)

    # Plot
    print "\nPlotting density diagrams and histograms of data"
    for di, dim, dgc in zip(data_transf, data_gated, data_gated_contour):
        print "{}...".format(str(di))

        # Plot
        fc.plot.density_and_hist(di, gated_data = dim, figsize = (7,7),
            density_channels = ['FSC-H', 'SSC-H'], 
            hist_channels = ['FL1-H'], gate_contour = dgc, 
            density_params = {'mode': 'scatter', 'log': True}, 
            hist_params = {'div': 4, 'log': True, 'edgecolor': 'g',
                'xlabel': 'Molecules of Equivalent Fluorescein (MEFL)'},
            savefig = '{}/{}.png'.format(gated_plot_dir, str(di)))
        pyplot.close()
        gc.collect()

    # Export to output excel file
    print "\nWriting output file..."
    # Calculate statistics
    # Figure out which channels have stats
    stat_channels = []
    for tf in transforms:
        for k, v in tf.iteritems():
            if k not in stat_channels:
                stat_channels.append(k)
    for diug, di, tc in zip(data_transf, data_gated, transforms):
        di.metadata['Ungated Counts'] = diug.shape[0]
        di.metadata['Gated Counts'] = di.shape[0]
        try:
          di.metadata['Acquisition Time (s)'] = di.acquisition_time
        except ValueError:
          pass
        
        for channel in stat_channels:
            if channel in tc.keys():
                di.metadata[channel + ' Gain'] = \
                                di[:,channel].channel_info[0]['pmt_voltage']
                di.metadata[channel + ' Mean'] = fc.stats.mean(di, channel)
                di.metadata[channel + ' Geom. Mean'] = \
                                fc.stats.gmean(di, channel)
                di.metadata[channel + ' Median'] = fc.stats.median(di, channel)
                di.metadata[channel + ' Mode'] = fc.stats.mode(di, channel)
                di.metadata[channel + ' Std'] = fc.stats.std(di, channel)
                di.metadata[channel + ' CV'] = fc.stats.CV(di, channel)
                di.metadata[channel + ' IQR'] = fc.stats.iqr(di, channel)
                di.metadata[channel + ' RCV'] = fc.stats.RCV(di, channel)
            else:
                di.metadata[channel + ' Gain'] = ''
                di.metadata[channel + ' Mean'] = ''
                di.metadata[channel + ' Geom. Mean'] = ''
                di.metadata[channel + ' Median'] = ''
                di.metadata[channel + ' Mode'] = ''
                di.metadata[channel + ' Std'] = ''
                di.metadata[channel + ' CV'] = ''
                di.metadata[channel + ' IQR'] = ''
                di.metadata[channel + ' RCV'] = ''
    # Convert data
    headers = data_gated[0].metadata.keys()
    content = []
    for di in data_gated:
        row = []
        for h in headers:
            row.append(di.metadata[h])
        content.append(row)
    # Save output excel file
    ws = [headers] + content
    fc.excel_io.export_workbook(output_form, {'cells': ws})

    print "\nDone."
    raw_input("Press Enter to exit...")

if __name__ == "__main__":
    main()