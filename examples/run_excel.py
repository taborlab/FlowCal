#!/usr/bin/python
import gc
import os
import os.path
import sys
from platform import system as platformSys
from subprocess import call
sys.path.append('..')

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
    # Launch dialog to select input file
    Tk().withdraw() # don't show main window
    # OSX ONLY: Call bash script to prevent file select window from sticking after use
    if platformSys() == 'Darwin':
        call("defaults write org.python.python ApplePersistenceIgnoreState YES", shell=True)
        input_form = askopenfilename(filetypes = [('Excel files', '*.xlsx')])
        call("defaults write org.python.python ApplePersistenceIgnoreState NO", shell=True)    
    else:
        input_form = askopenfilename(filetypes = [('Excel files', '*.xlsx')])
    if not input_form:
        print "Canelled."
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
    output_form = "{}/{}".format(basedir, input_filename + '_output.xls')

    # Get beads files data from input form
    beads_info = fc.excel_io.import_rows(input_form, "beads")
    to_mef_all = {}

    print "\nProcessing beads..."
    for bi in beads_info:
        # Open file
        di = fc.io.TaborLabFCSData("{}/{}".format(basedir, bi['File Path']))
        print "{} ({} events).".format(str(di), di.shape[0])

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
            hist_channels = ['FL1-H'],
            gate_contour = gate_contour, 
            density_params = {'mode': 'scatter'}, 
            hist_params = {'div': 4, 'edgecolor': 'g'},
            savefig = '{}/density_hist_{}.png'.format(beads_plot_dir, di))

        # Process MEF values
        mef = []
        mef_channels = []
        if 'MEFL' in bi:
            mefl = bi['MEFL'].split(',')
            mefl = [int(e) if e.strip().isdigit() else numpy.nan for e in mefl]
            mef.append(mefl)
            mef_channels.append('FL1-H')
        mef = numpy.array(mef)

        # Obtain standard curve transformation
        print "\nCalculating standard curve..."
        to_mef = fc.mef.get_transform_fxn(gated_di, mef, 
                        cluster_method = bi['Clustering Method'], 
                        cluster_channels = ['FL1-H', 'FL2-H', 'FL3-H'],
                        mef_channels = mef_channels, verbose = True, 
                        plot = True, plot_dir = beads_plot_dir)

        # Save MEF transformation function
        to_mef_all[bi['File Path']] = to_mef

    # Process data files
    print "\nLoading data..."
    # Get beads files data from input form
    cells_info = fc.excel_io.import_rows(input_form, "cells")

    # Load data files
    data = []
    for ci in cells_info:
        di = fc.io.TaborLabFCSData("{}/{}".format(basedir, ci['File Path']))
        data.append(di)

        gain = di[:,'FL1-H'].channel_info[0]['pmt_voltage']
        print "{} ({} events, FL1-H gain = {}).".format(str(di), 
            di.shape[0], gain)

    # Basic gating/trimming
    ch_all = ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H', 'FL3-H']
    data_trimmed = [fc.gate.start_end(di, num_start=250, num_end=100)\
                                                         for di in data]
    data_trimmed = [fc.gate.high_low(di, ch_all) for di in data_trimmed]
    # Exponential transformation
    data_transf = [fc.transform.exponentiate(di, ['FSC-H', 'SSC-H']) \
                                                        for di in data_trimmed]

    # Transform to MEF
    print "\nPerforming MEF transformation..."
    data_mef = []
    for ci, di in zip(cells_info, data_transf):
        print "{}...".format(str(di))
        to_mef = to_mef_all[ci['Beads File Path']]
        data_mef.append(to_mef(di, 'FL1-H'))

    # Density gate
    print "\nRunning density gate on data files..."
    data_gated = []
    data_gated_contour = []
    for ci, di in zip(cells_info, data_mef):
        print "{} (gate fraction = {:.2f})...".format(str(di), 
                float(ci['Gate Fraction']))
        di_gated, gate_contour = fc.gate.density2d(data = di, bins_log = True,
                                        gate_fraction = float(ci['Gate Fraction']))
        data_gated.append(di_gated)
        data_gated_contour.append(gate_contour)

    # Plot
    print "\nPlotting density diagrams and histograms of data"
    for di, dig, dgc in zip(data_mef, data_gated, data_gated_contour):
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

    # Export to output excel file
    print "\nWriting output file..."
    # Calculate statistics
    for ci, di, diug in zip(cells_info, data_gated, data):
        ci['Ungated Counts'] = diug.shape[0]
        ci['Gated Counts'] = di.shape[0]
        ci['Gated Counts/millisecond'] = fc.stats.rate(di,'Time')
        for channel in ['FL1-H']:
            ci[channel + ' Gain'] = di[:,channel].channel_info[0]['pmt_voltage']
            ci[channel + ' Mean'] = fc.stats.mean(di, channel)
            ci[channel + ' Geom. Mean'] = fc.stats.gmean(di, channel)
            ci[channel + ' Median'] = fc.stats.median(di, channel)
            ci[channel + ' Mode'] = fc.stats.mode(di, channel)
            ci[channel + ' Std'] = fc.stats.std(di, channel)
            ci[channel + ' CV'] = fc.stats.CV(di, channel)
            ci[channel + ' IQR'] = fc.stats.iqr(di, channel)
            ci[channel + ' RCV'] = fc.stats.RCV(di, channel)
    # Convert data
    headers = cells_info[0].keys()
    content = []
    for ci in cells_info:
        row = []
        for h in headers:
            row.append(ci[h])
        content.append(row)
    # Save output excel file
    ws = [headers] + content
    fc.excel_io.export_workbook(output_form, {'cells': ws})

    print "\nDone."
    raw_input("Press Enter to exit...")

if __name__ == "__main__":
    main()