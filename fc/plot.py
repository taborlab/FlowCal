#!/usr/bin/python
#
# plot.py - Module containing plotting functions for flow cytometry data sets.
#
# Author: Sebastian M. Castillo-Hair (smc9@rice.edu)
#         John T. Sexton (john.t.sexton@rice.edu)
# Date: 10/19/2015
#
# Requires:
#   * numpy
#   * matplotlib
#   * scipy

import numpy as np
import scipy.ndimage.filters
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

# Use default colors from palettable if available
try:
    import palettable
except ImportError, e:
    cmap_default = plt.get_cmap(matplotlib.rcParams['image.cmap'])
else:
    cmap_default = palettable.colorbrewer.diverging.Spectral_8_r.mpl_colormap
    matplotlib.rcParams['axes.color_cycle'] = palettable.colorbrewer\
        .qualitative.Paired_12.mpl_colors[1::2]

matplotlib.rcParams['savefig.dpi'] = 250

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
           legend_fontsize = 'medium',
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
        kwargs['facecolor'] = [cmap_default(i)\
                                for i in np.linspace(0, 1, len(data_list))]
    elif histtype == 'step' and 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = [cmap_default(i)\
                                for i in np.linspace(0, 1, len(data_list))]

    # Iterate through data_list
    for i, data in enumerate(data_list):
        # Extract channel
        y = data[:, channel]
        # If bins are not specified, get bins from FCSData object
        if bins is None and hasattr(y, 'channel_info'):
            # Get bin information
            r = y.channel_info[0]['range']
            bd = y.channel_info[0]['bin_edges']
            # Get bin scaled indices
            xd = np.linspace(0, 1, r[2] + 1)
            xs = np.linspace(0, 1, r[2]/div + 1)
            # Generate sub-sampled bins
            bins = np.interp(xs, xd, bd)

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
        plt.hist(y, bins, histtype = histtype, **kwargsi)
        if log == True:
            plt.gca().set_xscale('log')

    # Final configuration
    if xlabel is None:
        plt.xlabel(data[:,channel].channel_info[0]['label'])
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        if 'normed' in kwargs:
            plt.ylabel('Probability')
        else:
            plt.ylabel('Counts')
    else:
        plt.ylabel(ylabel)
    if xlim is None:
        plt.xlim((bins[0], bins[-1]))
    else:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    if legend:
        plt.legend(loc = legend_loc, prop={'size': legend_fontsize})

    # Save if necessary
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()

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

    # If bins are not specified, get bins from FCSData object
    if bins is None and hasattr(data_plot, 'channel_info'):
        # Get bin information
        rx = data_plot.channel_info[0]['range']
        bdx = data_plot.channel_info[0]['bin_edges']
        ry = data_plot.channel_info[1]['range']
        bdy = data_plot.channel_info[1]['bin_edges']
        # Get bin scaled indices
        xdx = np.linspace(0, 1, rx[2] + 1)
        xsx = np.linspace(0, 1, rx[2]/div + 1)
        xdy = np.linspace(0, 1, ry[2] + 1)
        xsy = np.linspace(0, 1, ry[2]/div + 1)
        # Generate sub-sampled bins
        bins = np.array([np.interp(xsx, xdx, bdx), 
                            np.interp(xsy, xdy, bdy)])

    # If colormap is not specified, use the default of this module
    if 'cmap' not in kwargs:
        kwargs['cmap'] = cmap_default

    # Calculate histogram
    H, xedges, yedges = np.histogram2d(data_plot[:,0],
                                        data_plot[:,1],
                                        bins = bins)

    # numpy histograms are organized such that the 1st dimension (eg. FSC) =
    # rows (1st index) and the 2nd dimension (eg. SSC) = columns (2nd index).
    # Visualized as is, this results in x-axis = SSC and y-axis = FSC, which
    # is not what we're used to. Transpose the histogram array to fix the
    # axes.
    H = H.T

    # Normalize
    if normed:
        H = H/np.sum(H)

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
        Hind = np.ravel(H)
        xv, yv = np.meshgrid(xedges[:-1], yedges[:-1])
        x = np.ravel(xv)[Hind != 0]
        y = np.ravel(yv)[Hind != 0]
        z = np.ravel(bH)[Hind != 0]
        plt.scatter(x, y, s=1, edgecolor='none', c=z, **kwargs)
    elif mode == 'mesh':
        plt.pcolormesh(xedges, yedges, bH, **kwargs)
    else:
        raise ValueError("Mode {} not recognized.".format(mode))

    # Plot
    if colorbar:
        cbar = plt.colorbar()
        if normed:
            cbar.ax.set_ylabel('Probability')
        else:
            cbar.ax.set_ylabel('Counts')
    # Reset axis and log if necessary
    if log:
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        a = list(plt.axis())
        a[0] = 10**(np.ceil(np.log10(xedges[0])))
        a[1] = 10**(np.ceil(np.log10(xedges[-1])))
        a[2] = 10**(np.ceil(np.log10(yedges[0])))
        a[3] = 10**(np.ceil(np.log10(yedges[-1])))
        plt.axis(a)
    else:
        a = list(plt.axis())
        a[0] = np.ceil(xedges[0])
        a[1] = np.ceil(xedges[-1])
        a[2] = np.ceil(yedges[0])
        a[3] = np.ceil(yedges[-1])
        plt.axis(a)
    # plt.grid(True)
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(data_plot.channel_info[0]['label'])
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(data_plot.channel_info[1]['label'])
    if title:
        plt.title(title)

    # Save if necessary
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()

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
        kwargs['color'] = [cmap_default(i)\
                                for i in np.linspace(0, 1, len(data_list))]

    # Iterate through data_list
    for i, data in enumerate(data_list):
        data_plot = data[:, channels]
        kwargsi = kwargs.copy()
        if 'color' in kwargsi:
            kwargsi['color'] = kwargs['color'][i]
        # ch0 vs ch2
        plt.scatter(data_plot[:,0], data_plot[:,1],
            s = 5, alpha = 0.25, **kwargsi)

    # Extract info about channels
    if hasattr(data_plot, 'channel_info'):
        name_ch = [data_plot[:,i].channel_info[0]['label'] for i in [0,1]]
        gain_ch = [data_plot[:,i].channel_info[0]['pmt_voltage'] for i in [0,1]]
        range_ch = [data_plot[:,i].channel_info[0]['range'] for i in [0,1]]

        plt.xlabel('{} (gain = {})'.format(name_ch[0], gain_ch[0]))
        plt.ylabel('{} (gain = {})'.format(name_ch[1], gain_ch[1]))
        plt.xlim(range_ch[0][0], range_ch[0][1])
        plt.ylim(range_ch[1][0], range_ch[1][1])

    # Save if necessary
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()


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
        kwargs['color'] = [cmap_default(i)\
                                for i in np.linspace(0, 1, len(data_list))]

    # Initial setup
    ax_3d = plt.gcf().add_subplot(222, projection='3d')

    # Iterate through data_list
    for i, data in enumerate(data_list):
        data_plot = data[:, channels]
        kwargsi = kwargs.copy()
        if 'color' in kwargsi:
            kwargsi['color'] = kwargs['color'][i]
        # ch0 vs ch2
        plt.subplot(221)
        plt.scatter(data_plot[:,0], data_plot[:,2],
            s = 5, alpha = 0.25, **kwargsi)
        # ch0 vs ch1
        plt.subplot(223)
        plt.scatter(data_plot[:,0], data_plot[:,1],
            s = 5, alpha = 0.25, **kwargsi)
        # ch2 vs ch1
        plt.subplot(224)
        plt.scatter(data_plot[:,2], data_plot[:,1],
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
        plt.subplot(221)
        plt.ylabel('{} (gain = {})'.format(name_ch[2], gain_ch[2]))
        plt.xlim(range_ch[0][0], range_ch[0][1])
        plt.ylim(range_ch[2][0], range_ch[2][1])
        # ch0 vs ch1
        plt.subplot(223)
        plt.xlabel('{} (gain = {})'.format(name_ch[0], gain_ch[0]))
        plt.ylabel('{} (gain = {})'.format(name_ch[1], gain_ch[1]))
        plt.xlim(range_ch[0][0], range_ch[0][1])
        plt.ylim(range_ch[1][0], range_ch[1][1])
        # ch2 vs ch1
        plt.subplot(224)
        plt.xlabel('{} (gain = {})'.format(name_ch[2], gain_ch[2]))
        plt.xlim(range_ch[2][0], range_ch[2][1])
        plt.ylim(range_ch[1][0], range_ch[1][1])
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
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()

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

    # Generate x data
    xdata = np.linspace(xlim[0],xlim[1],200)

    # Plot
    plt.plot(peaks_ch, peaks_mef, 'o', 
        label = 'Beads')
    plt.plot(xdata, sc_beads(xdata), 
        label = 'Beads model')
    plt.plot(xdata, sc_abs(xdata), 
        label = 'Standard curve')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    if xlabel:
        plt.xlabel(xlabel)
    if xlabel:
        plt.ylabel(ylabel)
    plt.legend(loc = 'best')
    
    # Save if necessary
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()

def bar(data, 
        labels,
        data_error = None,
        n_in_group = 1, 
        labels_in_group = [],
        legend_loc = 'best',
        legend_fontsize = 'medium',
        colors = None,
        bar_width = 0.75, 
        label_rotation = 0, 
        val_labels = True, 
        val_labels_fontsize = 'small',
        ylim = None,
        ylabel = None,
        savefig = None,
        **kwargs):
    ''' Draw a barplot.

    Individual bars can be grouped by specifying a number greater than one in
    n_in_group. Each group of n_in_group bars will share the same label and 
    will be plotted next to each other. Within a group, each bar can be 
    differentiated by color and legend label.

    data                - list or numpy array with the values to plot.
    labels              - labels for each bar or bar group.
    data_error          - size of the error bar to plot for each datapoint.
    n_in_group          - number of bars per group.
    labels_in_group     - labels within a group, used for a legend.
    legend_loc          - location of the legend.
    legend_fontsize     - font size used for the legend.
    colors              - list of colors of length == n_in_group.
    bar_width           - bar width.
    label_rotation      - angle to rotate the bar labels.
    val_labels          - if True, include labels above each bar with its
                            numberical value.
    val_labels_fontsize - font size of the labels above each bar.
    ylim                - limits on y axis
    ylabel              - label for y axis
    savefig             - if not None, it specifies the name of the file to 
                           save the figure to.
    **kwargs            - passed directly to matploblib's plot.
    '''

    # Default colors
    if colors is None:
        colors = matplotlib.rcParams['axes.color_cycle']

    # Calculate coordinates of x axis.
    x_coords = np.arange((len(data))/n_in_group)

    # Initialize plot
    ax = plt.gca()
    bars_group = []
    # Plot bars
    for i in range(n_in_group):
        x_coords_i = x_coords + i * bar_width / n_in_group
        if data_error is None:
            bars_group.append(plt.bar(x_coords_i, data[i::n_in_group], 
                bar_width / n_in_group, color = colors[i]))
        else:
            bars_group.append(plt.bar(x_coords_i, data[i::n_in_group], 
                bar_width / n_in_group, color = colors[i]), 
                yerr = data_error[i::n_in_group])
    ax.set_xticks(x_coords + bar_width / 2)

    # Rotate labels if necessary
    if label_rotation>0:
        ax.set_xticklabels(labels, rotation = label_rotation, ha='right')
    elif label_rotation < 0:        
        ax.set_xticklabels(labels, rotation = label_rotation, ha='left')
    else:
        ax.set_xticklabels(labels)

    # Set axes limits
    plt.xlim((bar_width - 1, x_coords[-1] + 1))
    if ylim:
        plt.ylim(ylim)

    # Set axes labels
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Add labels within group
    if labels_in_group:
        plt.legend(labels_in_group,
                   loc = legend_loc,
                   prop = {'size': legend_fontsize})

    # Add labels on top of bars
    if val_labels:
        dheight = plt.ylim()[1]*0.03
        fp = FontProperties(size = val_labels_fontsize)
        for bars in bars_group:
            for bar in bars:
                height = bar.get_height()
                if height > 1000:
                    text = '%3.0f'%height
                elif height > 100:
                    text = '%3.1f'%height
                else:
                    text = '%3.2f'%height
                ax.text(bar.get_x() + bar.get_width()/2., height + dheight, 
                    text, ha='center', va='bottom', fontproperties = fp)
    
    # Save if necessary
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()

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
    '''Makes a combined density/histograms plot of a FCSData object.

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
        height = 0.315 + 2.935*n_plots
        figsize = (6, height)

    # Create plot
    plt.figure(figsize = figsize)

    # Density plot
    if plot_density:
        plt.subplot(n_plots, 1, 1)
        # Plot density diagram
        density2d(data, channels = density_channels, **density_params)
        # Plot gate contour
        if gate_contour is not None:
            for g in gate_contour:
                plt.plot(g[:,0], g[:,1], color = 'k',
                    linewidth = 1.25)
        # Add title
        if 'title' not in density_params:
            if gated_data is not None:
                ret = gated_data.shape[0]*100./data.shape[0]
                title = "{} ({:.1f}% retained)".format(str(data), ret)
            else:
                title = str(data)
            plt.title(title)

    # Colors
    n_colors = n_plots - 1
    colors = [cmap_default(i) for i in np.linspace(0, 1, n_colors)]
    # Histogram
    for i, hist_channel in enumerate(hist_channels):
        # Define subplot
        plt.subplot(n_plots, 1, plot_density + i + 1)
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
            plt.legend(['Ungated', 'Gated'], loc = 'best', fontsize = 'medium')
        else:
            hist1d(data, channel = hist_channel, **hist_params_i)
    
    # Save if necessary
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()


def hist_and_bar(data_list,
                channel,
                labels,
                bar_stats_func = np.mean,
                hist_params = {},
                bar_params = {},
                figsize = None,
                savefig = None,
                ):
    '''Makes a combined histogram/bar plot of a set of FCSData objects.

    This function calls hist1d and bar to plot a histogram and a bar plot of
    several FCSData objects on a specified channel, using the same function
    call. The number plotted in the bar plot is calculated from the events list
    using the function specified in bar_stats_func.

    Parameters can be passed directly to hist1d and bar using hist_params and
    bar_params. n_in_group in bar_params is read by this function. If
    n_in_group is greater than one, the histogram's default colors and
    linestyles are modified in the following way: One color is used for a
    group, and members of a group a differentiated by linestyle.


    Arguments:
    data_list       - A list of FCSData objects.
    channel         - Channel to use.
    labels          - Labels to assign to each sample or group of samples.
    bar_stats_func  - Function to use to obtain a single number to plot in the
                        bar plot.
    hist_params     - Dictionary with the kwargs to pass to the hist1d
                        function.
    bar_params      - Dictionary with the kwargs to pass to the bar function.
    figsize         - Figure size. If None, calculate a default based on the
                        number of subplots.
    savefig         - if not None, it specifies the name of the file to save
                        the figure to.
    '''

    # Copy hist_params and bar_params, due to possible modifications
    hist_params = hist_params.copy()
    bar_params = bar_params.copy()

    # Extract number of groups to plot
    if 'n_in_group' in bar_params:
        n_in_group = bar_params['n_in_group']
    else:
        n_in_group = 1

    # Check appropriate length of labels array
    assert len(data_list)/n_in_group == len(labels), \
        "len(labels) should be the same as len(data_list)/n_in_group."

    # Calculate plot size if necessary
    if figsize is None:
        width = len(labels)*1.5
        figsize = (width, 6.2)

    # Create plot
    plt.figure(figsize = figsize)

    # Plot histogram

    # Default histogram type
    if 'histtype' not in hist_params:
        hist_params['histtype'] = 'step'
    histtype = hist_params['histtype']

    # Generate default colors
    hist_def_colors_1 = [cmap_default(i)\
                                for i in np.linspace(0, 1, len(labels))]
    hist_def_colors = []
    for hi in hist_def_colors_1:
        for j in range(n_in_group):
            hist_def_colors.append(hi)
    # Assign default colors if necessary
    if histtype == 'stepfilled' and 'facecolor' not in hist_params:
        hist_params['facecolor'] = hist_def_colors
    elif histtype == 'step' and 'edgecolor' not in hist_params:
        hist_params['edgecolor'] = hist_def_colors

    # Generate default linestyles
    hist_def_linestyles_1 = ['solid', 'dashed', 'dashdot', 'dotted']
    hist_def_linestyles = []
    for i in range(len(labels)):
        for j in range(n_in_group):
            hist_def_linestyles.append(hist_def_linestyles_1[j])
    # Assign default linestyles if necessary
    if 'linestyle' not in hist_params:
        hist_params['linestyle'] = hist_def_linestyles

    # Default legend
    if 'legend' not in hist_params:
        hist_params['legend'] = True
    if hist_params['legend'] and 'label' not in hist_params:
        hist_labels = []
        for i in range(len(labels)):
            for j in range(n_in_group):
                if 'labels_in_group' in bar_params:
                    hist_labels.append('{} ({})'.format(labels[i],
                        bar_params['labels_in_group'][j]))
                else:
                    hist_labels.append(labels[i])

        hist_params['label'] = hist_labels

    # Actually plot histogram
    plt.subplot(2, 1, 1)
    hist1d(data_list, channel = channel, **hist_params)

    # Bar plot
    # Calculate quantities to plot
    bar_data = [bar_stats_func(di[:, channel]) for di in data_list]

    # Actually plot
    plt.subplot(2, 1, 2)
    bar(bar_data, labels, **bar_params)

    # Save if necessary
    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
