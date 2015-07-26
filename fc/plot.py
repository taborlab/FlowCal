#!/usr/bin/python
#
# plot.py - Module containing plotting functions for flow cytometry data sets.
#
# Author: Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/6/2015
#
# Requires:
#   * numpy
#   * matplotlib
#   * scipy

import gc
import os
import csv

import numpy
import scipy.ndimage.filters
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def load_colormap(name, number):
    ''' Get colormap.

    If name specifies one of the provided colormaps, then open it.

    Colormap csvs have been extracted from http://colorbrewer2.org/.
    '''
    # Get path of module's directory
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    if name == 'spectral':
        # load raw csv data
        cm_raw = []
        with open(__location__ + '/spectral.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                cm_raw.append((float(row[0])/255,
                                float(row[1])/255, 
                                float(row[2])/255))
        # Get appropriate colormap for number of elements
        if number == 1:
            cm = [cm_raw[0]]
        elif number == 2:
            cm = [cm_raw[0], cm_raw[2]]
        elif number >= 3:
            start = numpy.sum(range(number)) - 3
            end = numpy.sum(range(number + 1)) - 3
            cm = cm_raw[start:end]
        return cm
    elif name == 'diverging':
        # load raw csv data
        cm_raw = []
        with open(__location__ + '/diverging.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                cm_raw.append((float(row[0])/255,
                                float(row[1])/255, 
                                float(row[2])/255))
        # Get appropriate colormap for number of elements
        if number <= 6:
            cm = cm_raw[:number]
        else:
            cm = [cm_raw[i%6] for i in range(number)]
        return cm
    else:
        raise ValueError("Colormap {} not recognized.".format(name))


##############################################################################
# SIMPLE PLOTS
##############################################################################
#
# The following functions produce simple plots independently of any other 
# function.
#
##############################################################################

def hist1d(data_list,
           channel = 0,
           log = False,
           div = 1,
           bins = None,
           legend = False,
           legend_loc = 'best',
           xlabel = None,
           ylabel = None,
           xlim = None,
           ylim = None,
           title = None,
           histtype = 'stepfilled',
           savefig = None,
           **kwargs):

    '''Plot 1D histogram of a list of data objects

    data_list  - a NxD FCSData object or numpy array, or a list of them.
    channel    - channel to use on the data objects.
    log        - whether the x axis should be in log units.
    div        - number to divide the default number of bins. Ignored if bins 
                argument is not None.
    bins       - bins argument to plt.hist.
    legend     - whether to include a legend.
    legend_loc - location of the legend to include.
    xlabel     - Label to use on the x axis
    ylabel     - Label to use on the y axis
    xlim       - Limits for the x axis
    ylim       - Limits for the y axis
    title      - Title for the plot
    histtype   - histogram type
    savefig    - if not None, it specifies the name of the file to save the 
                figure to.
    **kwargs   - passed directly to matploblib's hist. 'edgecolor', 
                'facecolor', 'linestyle', and 'label' can be specified as a 
                lists, with an element for each data object.
    '''    

    # Convert to list if necessary
    if not isinstance(data_list, list):
        data_list = [data_list]
        if 'edgecolor' in kwargs:
            kwargs['edgecolor'] = [kwargs['edgecolor']]
        if 'facecolor' in kwargs:
            kwargs['facecolor'] = [kwargs['facecolor']]
        if 'linestyle' in kwargs:
            kwargs['linestyle'] = [kwargs['linestyle']]
        if 'label' in kwargs:
            kwargs['label'] = [kwargs['label']]

    # Default colors
    if histtype == 'stepfilled' and 'facecolor' not in kwargs:
        kwargs['facecolor'] = load_colormap('spectral', len(data_list))
    elif histtype == 'step' and 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = load_colormap('spectral', len(data_list))

    # Iterate through data_list
    for i, data in enumerate(data_list):
        # Extract channel
        y = data[:, channel]
        # If bins are not specified, calculate bins from range
        if bins is None:
            r = y.channel_info[0]['range']
            if log == True:
                dr = (numpy.log10(r[1]) - numpy.log10(r[0]))/float(r[2] - 1)
                bins = numpy.logspace(numpy.log10(r[0]) - dr/2., 
                                        numpy.log10(r[1]) + dr/2., 
                                        (r[2]/div + 1))
            else:
                dr = (r[1] - r[0])/float(r[2] - 1)
                bins = numpy.linspace(r[0] - dr/2., 
                                        r[1] + dr/2., 
                                        (r[2]/div + 1))
        # Check for properties specified as lists.
        kwargsi = kwargs.copy()
        if 'edgecolor' in kwargsi:
            kwargsi['edgecolor'] = kwargsi['edgecolor'][i]
        if 'facecolor' in kwargsi:
            kwargsi['facecolor'] = kwargsi['facecolor'][i]
        if 'linestyle' in kwargsi:
            kwargsi['linestyle'] = kwargsi['linestyle'][i]
        if 'label' in kwargsi:
            kwargsi['label'] = kwargsi['label'][i]
        # Actually plot
        pyplot.hist(y, bins, histtype = histtype, **kwargsi)
        if log == True:
            pyplot.gca().set_xscale('log')

    # Final configuration
    if xlabel is None:
        pyplot.xlabel(data[:,channel].channel_info[0]['label'])
    else:
        pyplot.xlabel(xlabel)
    if ylabel is None:
        if 'normed' in kwargs:
            pyplot.ylabel('Probability')
        else:
            pyplot.ylabel('Counts')
    else:
        pyplot.ylabel(ylabel)
    if xlim is None:
        pyplot.xlim((bins[0], bins[-1]))
    else:
        pyplot.xlim(xlim)
    if ylim:
        pyplot.ylim(ylim)
    if title:
        pyplot.title(title)
    if legend:
        pyplot.legend(loc = legend_loc)

    # Save if necessary
    if savefig is not None:
        pyplot.tight_layout()
        pyplot.savefig(savefig, dpi = 300)
        pyplot.close()
        gc.collect()

def density2d(data, 
            channels = [0,1], 
            log = False, 
            div = 1, 
            bins = None, 
            smooth = True,
            sigma = 10.0,
            mode = 'mesh',
            colorbar = False,
            normed = False,
            xlabel = None,
            ylabel = None,
            title = None,
            savefig = None,
            **kwargs):
    '''Plot 2D density plot

    data        - a NxD FCSData object.
    channels    - channels to use in the density plot.
    log         - whether the x axis should be in log units.
    div         - number to divide the default number of bins. Ignored if bins 
                   argument is not None.
    bins        - bins to use for numpy.histogram2d.
    smooth      - Whether to apply gaussian smoothing to the histogram
    sigma       - Sigma parameter used for the gaussian smoothing.
    mode        - Plotting mode. Can be 'mesh' or 'scatter'.
    colorbar    - Plot colorbar
    normed      - Plot normed histogram (pmf)
    xlabel      - Label to use on the x axis
    ylabel      - Label to use on the y axis
    title       - Title for the plot.
    savefig     - if not None, it specifies the name of the file to save the 
                   figure to.
    kwargs      - passed directly to matplotlib's scatter or pcolormesh.
    '''

    # Extract channels to plot
    assert len(channels) == 2, 'Two channels need to be specified.'
    data_plot = data[:, channels]

    # Calculate bins if necessary
    if bins is None:
        rx = data_plot.channel_info[0]['range']
        ry = data_plot.channel_info[1]['range']
        if log == True:
            drx = (numpy.log10(rx[1]) - numpy.log10(rx[0]))/float(rx[2] - 1)
            dry = (numpy.log10(ry[1]) - numpy.log10(ry[0]))/float(ry[2] - 1)
            bins = numpy.array([numpy.logspace(numpy.log10(rx[0]), 
                                            numpy.log10(rx[1]) + drx, 
                                            (rx[2]/div + 1)),
                                numpy.logspace(numpy.log10(ry[0]), 
                                            numpy.log10(ry[1]) + dry, 
                                            (ry[2]/div + 1)),
                                ])
        else:
            drx = (rx[1] - rx[0])/float(rx[2] - 1)
            dry = (ry[1] - ry[0])/float(ry[2] - 1)
            bins = numpy.array([
                numpy.linspace(rx[0], rx[1] + drx, (rx[2]/div + 1)),
                numpy.linspace(ry[0], ry[1] + dry, (ry[2]/div + 1)),
                ])

    # Calculate histogram
    H, xedges, yedges = numpy.histogram2d(data_plot[:,0],
                                        data_plot[:,1],
                                        bins = bins)
    # H needs to be rotated and flipped
    H = numpy.rot90(H)
    H = numpy.flipud(H)

    # Normalize
    if normed:
        H = H/numpy.sum(H)

    # Smooth    
    if smooth:
        bH = scipy.ndimage.filters.gaussian_filter(
            H,
            sigma=sigma,
            order=0,
            mode='constant',
            cval=0.0)
    else:
        bH = H

    # Plotting mode
    if mode == 'scatter':
        Hind = numpy.ravel(H)
        xv, yv = numpy.meshgrid(xedges[:-1], yedges[:-1])
        x = numpy.ravel(xv)[Hind != 0]
        y = numpy.ravel(yv)[Hind != 0]
        z = numpy.ravel(bH)[Hind != 0]
        pyplot.scatter(x, y, s=1, edgecolor='none', c=z, **kwargs)
    elif mode == 'mesh':
        pyplot.pcolormesh(xedges, yedges, bH, **kwargs)
    else:
        raise ValueError("Mode {} not recognized.".format(mode))

    # Plot
    if colorbar:
        cbar = pyplot.colorbar()
        if normed:
            cbar.ax.set_ylabel('Probability')
        else:
            cbar.ax.set_ylabel('Counts')
    # Reset axis and log if necessary
    if log:
        pyplot.gca().set_xscale('log')
        pyplot.gca().set_yscale('log')
        a = list(pyplot.axis())
        a[0] = 10**(numpy.ceil(numpy.log10(xedges[0])))
        a[1] = 10**(numpy.ceil(numpy.log10(xedges[-1])))
        a[2] = 10**(numpy.ceil(numpy.log10(yedges[0])))
        a[3] = 10**(numpy.ceil(numpy.log10(yedges[-1])))
        pyplot.axis(a)
    else:
        a = list(pyplot.axis())
        a[0] = numpy.ceil(xedges[0])
        a[1] = numpy.ceil(xedges[-1])
        a[2] = numpy.ceil(yedges[0])
        a[3] = numpy.ceil(yedges[-1])
        pyplot.axis(a)
    # pyplot.grid(True)
    if xlabel:
        pyplot.xlabel(xlabel)
    else:
        pyplot.xlabel(data_plot.channel_info[0]['label'])
    if ylabel:
        pyplot.ylabel(ylabel)
    else:
        pyplot.ylabel(data_plot.channel_info[1]['label'])
    if title:
        pyplot.title(title)

    # Save if necessary
    if savefig is not None:
        pyplot.tight_layout()
        pyplot.savefig(savefig, dpi = 300)
        pyplot.close()
        gc.collect()

def scatter2d(data_list, 
                channels = [0,1], 
                savefig = None,
                **kwargs):

    '''Plot a 2D scatter plot of a list of data objects

    data_list  - a NxD FCSData object or numpy array, or a list of them.
    channels   - channels to use on the data objects.
    savefig    - if not None, it specifies the name of the file to save the 
                figure to.
    **kwargs   - passed directly to matploblib's plot functions. 'color' can be 
                specified as a list, with an element for each data object.
    '''    

    # Check appropriate number of channels
    assert len(channels) == 2, 'Two channels need to be specified.'

    # Convert to list if necessary
    if not isinstance(data_list, list):
        data_list = [data_list]
    if 'color' in kwargs:
        kwargs['color'] = [kwargs['color']]

    # Default colors
    if 'color' not in kwargs:
        kwargs['color'] = load_colormap('spectral', len(data_list))

    # Iterate through data_list
    for i, data in enumerate(data_list):
        data_plot = data[:, channels]
        kwargsi = kwargs.copy()
        if 'color' in kwargsi:
            kwargsi['color'] = kwargs['color'][i]
        # ch0 vs ch2
        pyplot.scatter(data_plot[:,0], data_plot[:,1],
            s = 5, alpha = 0.25, **kwargsi)

    # Extract info about channels
    if hasattr(data_plot, 'channel_info'):
        name_ch = [data_plot[:,i].channel_info[0]['label'] for i in [0,1]]
        gain_ch = [data_plot[:,i].channel_info[0]['pmt_voltage'] for i in [0,1]]
        range_ch = [data_plot[:,i].channel_info[0]['range'] for i in [0,1]]

        pyplot.xlabel('{} (gain = {})'.format(name_ch[0], gain_ch[0]))
        pyplot.ylabel('{} (gain = {})'.format(name_ch[1], gain_ch[1]))
        pyplot.xlim(range_ch[0][0], range_ch[0][1])
        pyplot.ylim(range_ch[1][0], range_ch[1][1])

    # Save if necessary
    if savefig is not None:
        pyplot.tight_layout()
        pyplot.savefig(savefig, dpi = 300)
        pyplot.close()
        gc.collect()


def scatter3d(data_list, 
                channels = [0,1,2], 
                savefig = None,
                **kwargs):

    '''Plot a 3D scatter plot and projections of a list of data objects

    data_list  - a NxD FCSData object or numpy array, or a list of them.
    channels   - channels to use on the data objects.
    savefig    - if not None, it specifies the name of the file to save the 
                figure to.
    **kwargs   - passed directly to matploblib's functions. 'color' can be 
                specified as a list, with an element for each data object.
    '''    

    # Check appropriate number of channels
    assert len(channels) == 3, 'Three channels need to be specified.'

    # Convert to list if necessary
    if not isinstance(data_list, list):
        data_list = [data_list]
    if 'color' in kwargs:
        kwargs['color'] = [kwargs['color']]

    # Default colors
    if 'color' not in kwargs:
        kwargs['color'] = load_colormap('spectral', len(data_list))

    # Initial setup
    ax_3d = pyplot.gcf().add_subplot(222, projection='3d')

    # Iterate through data_list
    for i, data in enumerate(data_list):
        data_plot = data[:, channels]
        kwargsi = kwargs.copy()
        if 'color' in kwargsi:
            kwargsi['color'] = kwargs['color'][i]
        # ch0 vs ch2
        pyplot.subplot(221)
        pyplot.scatter(data_plot[:,0], data_plot[:,2],
            s = 5, alpha = 0.25, **kwargsi)
        # ch0 vs ch1
        pyplot.subplot(223)
        pyplot.scatter(data_plot[:,0], data_plot[:,1],
            s = 5, alpha = 0.25, **kwargsi)
        # ch2 vs ch1
        pyplot.subplot(224)
        pyplot.scatter(data_plot[:,2], data_plot[:,1],
            s = 5, alpha = 0.25, **kwargsi)
        # 3d
        ax_3d.scatter(data_plot[:,0], data_plot[:,1], data_plot[:,2], 
            marker='o', alpha = 0.1, **kwargsi)

    # Extract info about channels
    if hasattr(data_plot, 'channel_info'):
        name_ch = [data_plot[:,i].channel_info[0]['label'] for i in [0,1,2]]
        gain_ch = [data_plot[:,i].channel_info[0]['pmt_voltage']\
                                                             for i in [0,1,2]]
        range_ch = [data_plot[:,i].channel_info[0]['range'] for i in [0,1,2]]

        # ch0 vs ch2
        pyplot.subplot(221)
        pyplot.ylabel('{} (gain = {})'.format(name_ch[2], gain_ch[2]))
        pyplot.xlim(range_ch[0][0], range_ch[0][1])
        pyplot.ylim(range_ch[2][0], range_ch[2][1])
        # ch0 vs ch1
        pyplot.subplot(223)
        pyplot.xlabel('{} (gain = {})'.format(name_ch[0], gain_ch[0]))
        pyplot.ylabel('{} (gain = {})'.format(name_ch[1], gain_ch[1]))
        pyplot.xlim(range_ch[0][0], range_ch[0][1])
        pyplot.ylim(range_ch[1][0], range_ch[1][1])
        # ch2 vs ch1
        pyplot.subplot(224)
        pyplot.xlabel('{} (gain = {})'.format(name_ch[2], gain_ch[2]))
        pyplot.xlim(range_ch[2][0], range_ch[2][1])
        pyplot.ylim(range_ch[1][0], range_ch[1][1])
        # 3d
        ax_3d.set_xlim(range_ch[0][0], range_ch[0][1])
        ax_3d.set_ylim(range_ch[1][0], range_ch[1][1])
        ax_3d.set_zlim(range_ch[2][0], range_ch[2][1])
        ax_3d.set_xlabel(name_ch[0])
        ax_3d.set_ylabel(name_ch[1])
        ax_3d.set_zlabel(name_ch[2])
        ax_3d.xaxis.set_ticklabels([])
        ax_3d.yaxis.set_ticklabels([])
        ax_3d.zaxis.set_ticklabels([])

    # Save if necessary
    if savefig is not None:
        pyplot.tight_layout()
        pyplot.savefig(savefig, dpi = 300)
        pyplot.close()
        gc.collect()

def mef_std_crv(peaks_ch, 
                peaks_mef,
                sc_beads,
                sc_abs,
                xlim = (0., 1023.),
                ylim = (1, 1e8),
                xlabel = None,
                ylabel = None,
                savefig = None,
                **kwargs):
    '''Plot the standard curves of a beads model.

    peaks_ch   - experimental values of peaks in channel space.
    peaks_mef  - theoretical MEF values of peaks
    sc_beads   - standard curve of the beads model.
    sc_abs     - standard curve in absolute MEF units.
    xlim       - limits on x axis
    ylim       - limits on y axis
    xlabel     - label for x axis
    ylabel     - label for y axis
    savefig    - if not None, it specifies the name of the file to save the 
                figure to.
    **kwargs   - passed directly to matploblib's plot.
    '''    

    # Get colors
    colors = load_colormap('diverging', 3)
    # Generate x data
    xdata = numpy.linspace(xlim[0],xlim[1],200)

    # Plot
    pyplot.plot(peaks_ch, peaks_mef, 'o', 
        label = 'Beads', color = colors[0])
    pyplot.plot(xdata, sc_beads(xdata), 
        label = 'Beads model', color = colors[1])
    pyplot.plot(xdata, sc_abs(xdata), 
        label = 'Standard curve', color = colors[2])
    pyplot.yscale('log')
    pyplot.xlim(xlim)
    pyplot.ylim(ylim)
    pyplot.grid(True)
    if xlabel:
        pyplot.xlabel(xlabel)
    if xlabel:
        pyplot.ylabel(ylabel)
    pyplot.legend(loc = 'best')
    
    # Save if necessary
    if savefig is not None:
        pyplot.tight_layout()
        pyplot.savefig(savefig, dpi = 300)
        pyplot.close()
        gc.collect()


##############################################################################
# COMPLEX PLOTS
##############################################################################
#
# The functions below produce plots by composing the results of the functions 
# defined above.
#
##############################################################################

def density_and_hist(data,
                    gated_data = None,
                    gate_contour = None,
                    density_channels = None,
                    density_params = {},
                    hist_channels = None,
                    hist_params = {},
                    figsize = None,
                    savefig = None,
                    ):
    '''Makes a combined density/histograms plot of a FCSData file.

    This function calls hist1d and density2d to plot a density diagram and a 
    number of histograms in the same plot using one single function call. 
    Setting density_channels to None will not produce a density diagram, 
    and setting hist_channels to None will not produce any histograms. Setting
    both to None will raise an error.

    If gated_data is True, this function will plot the histograms corresponding
    to gated_data on top of data's histograms, with some level of transparency
    on data. 

    Arguments:
    data                - FCSData object
    gated_data          - FCSData object
    gate_contour        - List of Nx2 curves, representing a gate, to be 
                            plotted in the density diagram.
    density_channels    - 2-element iterable with the channels to use for the 
                            density plot. Default: None (no density plot)
    density_params      - Dictionary with the kwargs to pass to the density2d 
                            function.
    hist_channels       - channels to use in each one of the histograms.
                            Default: None (no histograms)
    hist_params         - Dictionary with the kwargs to pass to the hist1d 
                            function.
    figsize             - Figure size. If None, calculate a default based on 
                            the number of subplots.
    savefig             - if not None, it specifies the name of the file to 
                            save the figure to.
    '''

    # Check number of plots
    if density_channels is None and hist_channels is None:
        raise ValueError("density_channels and hist_channels cannot be both \
            None.")
    # Change hist_channels to iterable if necessary
    if not hasattr(hist_channels, "__iter__"):
        hist_channels = [hist_channels]
    if isinstance(hist_params, dict):
        hist_params = [hist_params]*len(hist_channels)

    plot_density = not(density_channels is None)
    n_plots = plot_density + len(hist_channels)

    # Calculate plot size if necessary
    if figsize is None:
        height = 3.25*n_plots
        figsize = (6, height)

    # Create plot
    pyplot.figure(figsize = figsize)

    # Density plot
    if plot_density:
        pyplot.subplot(n_plots, 1, 1)
        # Plot density diagram
        density2d(data, channels = density_channels, **density_params)
        # Plot gate contour
        if gate_contour is not None:
            for g in gate_contour:
                pyplot.plot(g[:,0], g[:,1], color = 'r', linewidth = 1.5)
        # Add title
        if 'title' not in density_params:
            if gated_data is not None:
                ret = gated_data.shape[0]*100./data.shape[0]
                title = "{} ({:.1f}% retained)".format(str(data), ret)
            else:
                title = str(data)
            pyplot.title(title)


    # Colors
    if n_plots - 1 < 3:
        n_colors = 3
    else:
        n_colors = n_plots - 1
    colors = load_colormap('spectral', n_colors)
    colors = colors[::-1]
    # Histogram
    for i, hist_channel in enumerate(hist_channels):
        # Define subplot
        pyplot.subplot(n_plots, 1, plot_density + i + 1)
        # Default colors
        hist_params_i = hist_params[i].copy()
        if 'facecolor' not in hist_params_i:
            hist_params_i['facecolor'] = colors[i]
        # Plots
        if gated_data is not None:
            hist1d(data, channel = hist_channel, 
                alpha = 0.5, **hist_params_i)
            hist1d(gated_data, channel = hist_channel, 
                alpha = 1.0, **hist_params_i)
            pyplot.legend(['Ungated', 'Gated'])
        else:
            hist1d(data, channel = hist_channel, **hist_params_i)
    
    # Save if necessary
    if savefig is not None:
        pyplot.tight_layout()
        pyplot.savefig(savefig, dpi = 300)
        pyplot.close()
        gc.collect()
