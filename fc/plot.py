"""
Functions for visualizing flow cytometry data.

"""

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
    """
    Plot one 1D histogram from one or more flow cytometry data sets.

    This function does not create a new figure or axis, so it can be called
    directly to plot in a previously created axis if desired. If `savefig`
    is not specified, the plot is maintained in the current axis when the
    function returns. This allows for further modifications to the axis by
    direct calls to, for example, ``plt.xlabel``, ``plt.title``, etc.
    However, if `savefig` is specified, the figure is closed after being
    saved. In this case, parameters `xlabel`, `ylabel`, `xlim`, `ylim`,
    `title`, and the legend-related parameters of this function are the
    only way to modify the axis.

    Parameters
    ----------
    data_list : FCSData or numpy array or list of FCSData or numpy array
        Flow cytometry data to plot.
    channel : int or str, optional
        Channel from where to take the events to plot. If ndim == 1,
        channel is ignored. String channel specifications are only
        supported for data types which support string-based indexing
        (e.g. FCSData).
    log : bool, optional
        Flag specifying whether the x axis should be in log scale.
    div : int or float, optional
        Downscaling factor for the default number of bins. If `bins` is not
        specified, the default set of bins extracted from an element in
        `data_list` contains ``n`` bins, and ``div != 1``, `hist1d` will
        actually use ``n/div`` bins that cover the same range as the
        default bins. `div` is ignored if `bins` is specified.
    bins : array_like, optional
        bins argument to pass to plt.hist. If not specified, attempts to
        extract bins from data object.
    legend : bool, optional
        Flag specifying whether to include a legend. If `legend` is True,
        the legend labels will be taken from ``kwargs['label']``.
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, str, optional
        Histogram type. Directly passed to ``plt.hist``.
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    legend_loc : str, optional
        Location of the legend.
    xlabel : str, optional
        Label to use on the x axis. If None, attempts to extract channel
        name from last data object.
    ylabel : str, optional
        Label to use on the y axis. If None and ``kwargs['normed']==True``,
        use 'Probability'. If None and ``kwargs['normed']==False``, use
        'Counts'.
    xlim : tuple, optional
        Limits for the x axis. If not specified and `bins` exists, use
        the lowest and highest values of `bins`.
    ylim : tuple, optional
        Limits for the y axis.
    title : str, optional
        Plot title.
        `edgecolor`, `facecolor`, `linestyle`, and `label` can be specified
        as lists, with an element for each data object.
    kwargs : dict, optional
        Additional parameters passed directly to matploblib's ``hist``.
        ``facecolor``, ``edgecolor``, ``linestyle``, and ``label`` can be
        specified as a list, with an element for each object in
        `data_list`. If ``histtype=='stepfilled'`` and no ``facecolor`` is
        specified, default values for ``facecolor`` are taken from the
        default colormap. If ``histtype=='step'`` and no ``edgecolor``
        is specified in `kwargs`, default values for ``edgecolor`` are
        taken from the default colormap.

    Notes
    -----
    `hist1d` calls matplotlib's ``hist`` function for each object in
    `data_list`. `hist_type`, the type of histogram to draw, is directly
    passed to ``plt.hist``. Additional keyword arguments provided to
    `hist1d` are passed directly to ``plt.hist``.

    """
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
        if data.ndim > 1:
            y = data[:, channel]
        else:
            y = data
        # If bins are not specified, try to get bins from data object
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
        if bins is not None:
            n, edges, patches = plt.hist(y, bins, histtype=histtype, **kwargsi)
        else:
            n, edges, patches = plt.hist(y, histtype=histtype, **kwargsi)
        if log == True:
            plt.gca().set_xscale('log')

    ###
    # Final configuration
    ###

    # x and y labels
    if xlabel is not None:
        # Highest priority is user-provided label
        plt.xlabel(xlabel)
    elif hasattr(y, 'channel_info'):
        # Attempt to use channel name
        plt.xlabel(y.channel_info[0]['label'])

    if ylabel is not None:
        # Highest priority is user-provided label
        plt.ylabel(ylabel)
    elif 'normed' in kwargs:
        plt.ylabel('Probability')
    else:
        # Default is "Counts"
        plt.ylabel('Counts')

    # x and y limits
    if xlim is not None:
        # Highest priority is user-provided limits
        plt.xlim(xlim)
    elif bins is not None:
        # Use bins if specified
        plt.xlim((edges[0], edges[-1]))

    if ylim is not None:
        plt.ylim(ylim)

    # title and legend
    if title is not None:
        plt.title(title)

    if legend is not None:
        plt.legend(loc=legend_loc, prop={'size': legend_fontsize})

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
    """
    Plot a 2D density plot from two channels of a flow cytometry data set.

    `density2d` has two plotting modes which are selected using the `mode`
    argument. With ``mode=='mesh'``, this function plots the data as a true
    2D histogram, in which a plane is divided into bins and the color of
    each bin is directly related to the number of elements therein. With
    ``mode=='scatter'``, this function also calculates a 2D histogram,
    but it plots a 2D scatter plot in which each dot corresponds to a bin,
    colored according to the number elements therein. The most important
    difference is that the ``scatter`` mode does not color regions
    corresponding to empty bins. This allows for easy identification of
    regions with low number of events. For both modes, the calculated
    histogram can be smoothed using a Gaussian kernel by specifying
    ``smooth=True``. The width of the kernel is, in this case, given by
    `sigma`.

    This function does not create a new figure or axis, so it can be called
    directly to plot in a previously created axis if desired. If `savefig`
    is not specified, the plot is maintained in the current axis when the
    function returns. This allows for further modifications to the axis by
    direct calls to, for example, ``plt.xlabel``, ``plt.title``, etc.
    However, if `savefig` is specified, the figure is closed after being
    saved. In this case, parameters `xlabel`, `ylabel`, `xlim`, `ylim`,
    `title`, and the legend-related parameters of this function are the
    only way to modify the axis.

    Parameters
    ----------
    data : FCSData or numpy array
        Flow cytometry data to plot.
    channels : list of int, list of str, optional
        Two channels to use for the plot.
    log : bool, optional
        Flag specifying whether the axes should be in log scale.
    div : int or float, optional
        Downscaling factor for the default number of bins. If `bins` is not
        specified, the default set of bins extracted from `data` contains
        ``n*m`` bins, and ``div != 1``, `density2d` will actually use
        ``n/div * m/div`` bins that cover the same range as the default
        bins. `div` is ignored if `bins` is specified.
    bins : array_like, optional
        bins argument to pass to plt.hist. If not specified, attempts to 
        extract bins from `data`.
    smooth : bool, optional
        Flag indicating whether to apply Gaussian smoothing to the
        histogram.
    mode : {'mesh', 'scatter'}, str, optional
        Plotting mode. 'mesh' produces a 2D-histogram whereas 'scatter'
        produces a scatterplot colored by histogram bin value.
    colorbar : bool, optional
        Flag indicating whether to add a colorbar to the plot.
    normed : bool, optional
        Flag indicating whether to plot a normed histogram (probability
        mass function instead of a counts-based histogram).
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    sigma : float, optional
        The sigma parameter for the Gaussian kernel to use when smoothing.
    xlabel : str, optional
        Label to use on the x axis. If None, attempts to extract channel
        name from `data`.
    ylabel : str, optional
        Label to use on the y axis. If None, attempts to extract channel
        name from `data`.
    title : str, optional
        Plot title.
    kwargs : dict, optional
        Additional parameters passed directly to the underlying matplotlib
        functions: ``plt.scatter`` if ``mode==scatter``, and
        ``plt.pcolormesh`` if ``mode==mesh``.

    """
    # Extract channels to plot
    assert len(channels) == 2, 'Two channels need to be specified.'
    data_plot = data[:, channels]

    # If bins are not specified, try to get bins from data object
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
    if bins is not None:
        H,xe,ye = np.histogram2d(data_plot[:,0], data_plot[:,1], bins=bins)
    else:
        H,xe,ye = np.histogram2d(data_plot[:,0], data_plot[:,1])

    # Smooth    
    if smooth:
        sH = scipy.ndimage.filters.gaussian_filter(
            H,
            sigma=sigma,
            order=0,
            mode='constant',
            cval=0.0)
    else:
        sH = None

    # Normalize
    if normed:
        H = H / np.sum(H)
        sH = sH / np.sum(sH) if sH is not None else None

    ###
    # Plot
    ###

    # numpy histograms are organized such that the 1st dimension (eg. FSC) =
    # rows (1st index) and the 2nd dimension (eg. SSC) = columns (2nd index).
    # Visualized as is, this results in x-axis = SSC and y-axis = FSC, which
    # is not what we're used to. Transpose the histogram array to fix the
    # axes.
    H = H.T
    sH = sH.T if sH is not None else None

    if mode == 'scatter':
        Hind = np.ravel(H)
        xc = (xe[:-1] + xe[1:]) / 2.0   # x-axis bin centers
        yc = (ye[:-1] + ye[1:]) / 2.0   # y-axis bin centers
        xv, yv = np.meshgrid(xc, yc)
        x = np.ravel(xv)[Hind != 0]
        y = np.ravel(yv)[Hind != 0]
        z = np.ravel(H if sH is None else sH)[Hind != 0]
        plt.scatter(x, y, s=1, edgecolor='none', c=z, **kwargs)
    elif mode == 'mesh':
        plt.pcolormesh(xe, ye, H if sH is None else sH, **kwargs)
    else:
        raise ValueError("mode {} not recognized".format(mode))

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
        a[0] = 10**(np.ceil(np.log10(xe[0])))
        a[1] = 10**(np.ceil(np.log10(xe[-1])))
        a[2] = 10**(np.ceil(np.log10(ye[0])))
        a[3] = 10**(np.ceil(np.log10(ye[-1])))
        plt.axis(a)
    else:
        a = list(plt.axis())
        a[0] = np.ceil(xe[0])
        a[1] = np.ceil(xe[-1])
        a[2] = np.ceil(ye[0])
        a[3] = np.ceil(ye[-1])
        plt.axis(a)

    # x and y labels
    if xlabel is not None:
        # Highest priority is user-provided label
        plt.xlabel(xlabel)
    elif hasattr(data_plot, 'channel_info'):
        # Attempt to use channel name
        plt.xlabel(data_plot.channel_info[0]['label'])

    if ylabel is not None:
        # Highest priority is user-provided label
        plt.ylabel(ylabel)
    elif hasattr(data_plot, 'channel_info'):
        # Attempt to use channel name
        plt.ylabel(data_plot.channel_info[1]['label'])

    # title
    if title is not None:
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
    """
    Plot one 2D scatter plot from one or more FCSData objects.

    The name of the specified channels and the detector gain are used for
    the axes labels.

    This function does not create a new figure or axis, so it can be called
    directly to plot in a previously created axis if desired. If `savefig`
    is not specified, the plot is maintained in the current axis when the
    function returns. This allows for further modifications to the axis by
    direct calls to, for example, ``plt.xlabel``, ``plt.title``, etc.
    However, if `savefig` is specified, the figure is closed after being
    saved. In this case, the default values for ``xlabel`` and ``ylabel``
    will be used.

    Parameters
    ----------
    data_list : FCSData object, or list of FCSData objects
        Flow cytometry data to plot.
    channels : list of int, list of str
        Two channels to use for the plot.
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    kwargs : dict, optional
        Additional parameters passed directly to matploblib's ``scatter``.
        `color` can be specified as a list, with an element for each data
        object. If the keyword argument `color` is not provided, elements
        from `data_list` are plotted with colors taken from the default
        colormap.

    Notes
    -----
    `scatter2d` calls matplotlib's ``scatter`` function for each object in
    data_list. Additional keyword arguments provided to `scatter2d` are
    passed directly to ``plt.scatter``.

    """
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
    """
    Plot one 3D scatter plot from one or more FCSData objects.

    `scatter3d` creates a 3D scatter plot and three 2D projected scatter
    plots in four different axes for each FCSData object in `data_list`,
    in the same figure. The name of the specified channels and the detector
    gain are used for the axes labels.

    This function does not create a new figure, so it can be called
    directly to plot in a previously created figure if desired. However,
    it creates four axes using ``plt.subplot``. If `savefig` is not
    specified, the plot is maintained in the current figure when the
    function returns. If `savefig` is specified, the figure is closed
    after being saved.

    Parameters
    ----------
    data_list : FCSData object, or list of FCSData objects
        Flow cytometry data to plot.
    channels : list of int, list of str
        Three channels to use for the plot.
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    kwargs : dict, optional
        Additional parameters passed directly to matploblib's ``scatter``.
        `color` can be specified as a list, with an element for each data
        object. If the keyword argument `color` is not provided, elements
        from `data_list` are plotted with colors taken from the default
        colormap.

    Notes
    -----
    `scatter3d` uses matplotlib's ``scatter``, with the 3D scatter plot
    using a 3D projection. Additional keyword arguments provided to
    `scatter3d` are passed directly to ``scatter``.

    """
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
    """
    Plot a standard curve with fluorescence of calibration beads.

    This function does not create a new figure or axis, so it can be called
    directly to plot in a previously created axis if desired. If `savefig`
    is not specified, the plot is maintained in the current axis when the
    function returns. This allows for further modifications to the axis by
    direct calls to, for example, ``plt.xlabel``, ``plt.title``, etc.
    However, if `savefig` is specified, the figure is closed after being
    saved. In this case, parameters `xlabel`, `ylabel`, `xlim`, and `ylim`
    are the only way to modify the axis.

    Parameters
    ----------
    peaks_ch : array_like
        Fluorescence of the calibration beads' subpopulations, in channel
        numbers.
    peaks_mef : array_like
        Fluorescence of the calibration beads' subpopulations, in MEF
        units.
    sc_beads : function
        The calibration beads fluorescence model.
    sc_abs : function
        The standard curve (transformation functionfrom channel number to
        MEF units).
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    xlim : tuple, optional
        Limits for the x axis.
    ylim : tuple, optional
        Limits for the y axis.
    xlabel : str, optional
        Label to use on the x axis.
    ylabel : str, optional
        Label to use on the y axis.

    """
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
    """
    Draw a barplot.

    This function does not create a new figure or axis, so it can be called
    directly to plot in a previously created axis if desired. If `savefig`
    is not specified, the plot is maintained in the current axis when the
    function returns. This allows for further modifications to the axis by
    direct calls to, for example, ``plt.ylabel``, ``plt.title``, etc.
    However, if `savefig` is specified, the figure is closed after being
    saved. In this case, parameters `ylabel`, `ylim`, and the
    legend-related parameters of this function are the only way to modify
    the axis.

    Parameters
    ----------
    data : array_like
        Values to plot.
    labels : list of str
        Labels for each bar or bar group, to be displayed on the x axis.
    n_in_group : int, optional
        Number of bars per group. Each group of `n_in_group` bars will
        share the same label and will be plotted next to each other.
    labels_in_group : list of str, optional
        Labels for bars within a group, to be used in the legend.
        `labels_in_group` should be of length `n_in_group`. If
        `labels_in_group` is not specified, a legend will not be plotted.
    colors : list, optional
        Colors to be used for each bar within a group. `colors` should be
        of length `n_in_group`.
    val_labels : bool, optional
        Flag indicating whether to include labels above each bar with its
        numerical value.
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    data_error : array_like, optional
        Size of the error bar to plot for each bar.
    legend_loc : int or str, optional
        Location of the legend. Check ``plt.legend`` for possible values.
    legend_fontsize : int or float or str, optional
        Font size used for the legend.
    bar_width : float, optional
        Bar width.
    label_rotation : float, optional
        Angle to rotate the x-axis labels.
    val_labels_fontsize : int or float or str, optional
        Font size of the labels with numberical values above each bar.
    ylim : tuple, optional, optional
        Limits for the y axis.
    ylabel : str, optional
        Label to use on the y axis.

    """
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
    """
    Make a combined density/histogram plot of a FCSData object.

    This function calls `hist1d` and `density2d` to plot a density diagram
    and a number of histograms in different subplots of the same plot using
    one single function call. Setting `density_channels` to None will not
    produce a density diagram, and setting `hist_channels` to None will not
    produce any histograms. Setting both to None will raise an error.
    Additional parameters can be provided to `density2d` and `hist1d` by
    using `density_params` and `hist_params`.

    If `gated_data` is provided, this function will plot the histograms
    corresponding to `gated_data` on top of `data`'s histograms, with some
    transparency on `data`. In addition, a legend will be added with the
    labels 'Ungated' and 'Gated'. If `gate_contour` is provided and it
    contains a valid list of 2D curves, these will be plotted on top of the
    density plot.

    This function creates a new figure and a set of axes. If `savefig` is
    not specified, the plot is maintained in the newly created figure when
    the function returns. However, if `savefig` is specified, the figure
    is closed after being saved.

    Parameters
    ----------
    data : FCSData object
        Flow cytometry data object to plot.
    gated_data : FCSData object, optional
        Flow cytometry data object. If `gated_data` is specified, the
        histograms of `data` are plotted with an alpha value of 0.5, and
        the histograms of `gated_data` are plotted on top of those with
        an alpha value of 1.0.
    gate_contour : list, optional
        List of Nx2 curves, representing a gate contour to be plotted in
        the density diagram.
    density_channels : list
        Two channels to use for the density plot. If `density_channels` is
        None, do not plot a density plot.
    density_params : dict, optional
        Parameters to pass to `density2d`.
    hist_channels : list
        Channels to use for each histogram. If `hist_channels` is None,
        do not plot histograms.
    hist_params : list, optional
        List of dictionaries with the parameters to pass to each call of
        `hist1d`.
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    figsize : tuple, optional
        Figure size. If None, calculate a default based on the number of
        subplots.

    Raises
    ------
    ValueError
        If both `density_channels` and `hist_channels` are None.

    """
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
    """
    Make a combined histogram/bar plot for one channel from a list of
    FCSData objects.

    This function calls `hist1d` and `bar` to plot a histogram and a bar
    plot of several FCSData objects on a specified channel, using a single
    function call. The number plotted in the bar plot is calculated from
    the events list using the function specified in `bar_stats_func`.
    Additional parameters can be provided to `hist1d` and `bar` by
    using `hist_params` and `bar_params`.

    'n_in_group' in `bar_params` is read by `hist_and_bar`. If
    `n_in_group` is greater than one, the histogram's default colors and
    linestyles are modified in the following way: One color is used for all
    data elements in a group, and members of a group are differentiated by
    linestyle.

    This function creates a new figure and a set of axes. If `savefig` is
    not specified, the plot is maintained in the newly created figure when
    the function returns. However, if `savefig` is specified, the figure
    is closed after being saved.

    Parameters
    ----------
    data_list : List of FCSData objects
        Flow cytometry data to plot.
    channel : int or str
        Channel to use for the plot.
    labels : list of str
        Labels to assign to each sample or group of samples.
    bar_stats_func : function
        Function to use to obtain a single number for each element in
        `data_list` to plot in the bar plot.
    hist_params : dict, optional
        Parameters to pass to `hist1d`.
    bar_params : dict, optional
        Parameters to pass to `bar`.
    savefig : str, optional
        The name of the file to save the figure to. If None, do not save.

    Other parameters
    ----------------
    figsize : tuple, optional
        Figure size. If None, calculate a default based on the number of
        subplots.

    """
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
