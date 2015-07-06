#!/usr/bin/python
#
# plot.py - Module containing plotting functions for flow cytometry data sets.
#
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/6/2015
#
# Requires:
#   * numpy
#   * matplotlib
#   * scipy

import os
import csv
import numpy
from matplotlib import pyplot
import scipy.ndimage.filters

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

    else:
        raise ValueError("Colormap {} not recognized.".format(name))

def hist1d(data_list,
           channel = 0,
           log = False,
           div = 1,
           bins = None,
           legend = False,
           legend_loc = None,
           xlabel = None,
           savefig = None,
           histtype = 'stepfilled',
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
    histtype   - histogram type
    **kwargs   - passed directly to matploblib's hist. 'edgecolor', 
                'facecolor', 'linestyle', and 'label' can be specified as a 
                lists, with an element for each data object.
    '''    

    # Convert to list if it's not already
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
    if 'facecolor' not in kwargs:
        kwargs['facecolor'] = load_colormap('spectral', len(data_list))

    # Iterate through data_list
    for i, data in enumerate(data_list):
        # Extract channel
        y = data[:, channel]
        # If bins are not specified, calculate bins from range
        if bins is None:
            r = y.channel_info[0]['range']
            if log == True:
                bins = numpy.logspace(numpy.log10(r[0]), 
                                        numpy.log10(r[1]), 
                                        (r[2] + 1)/div)
            else:
                bins = numpy.linspace(r[0], r[1] + 1, (r[2] + 1)/div)
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
        pyplot.xlabel(data.channel_info[0]['label'])
    else:
        pyplot.xlabel(xlabel)
    pyplot.xlim((bins[0], bins[-1]))
    if 'normed' in kwargs:
        pyplot.ylabel('Probability')
    else:
        pyplot.ylabel('Counts')
    if legend:
        pyplot.legend(loc = legend_loc)

    # Save if necessary
    if savefig is not None:
        pyplot.savefig(savefig, dpi = 300)


# def hist2d(data,
#            bins=np.arange(1025)-0.5,
#            axes_limits=[0,1023,0,1023],
#            xlabel='FSC',
#            ylabel='SSC',
#            title=None,
#            colorbar=True,
#            gate=None,
#            ax=None):
#     '''Plot 2D histogram.

#     data        - NxD numpy array (only first 2 dimensions [columns] are used)
#     bins        - bins argument to np.histogram2d (default=np.arange(1025)-0.5)
#     axes_limits - axis boundaries
#     xlabel      - string to label x-axis
#     ylabel      - string to label y-axis
#     title       - string to label plot
#     colorbar    - show colorbar
#     gate        - Mx2 numpy array or list of Mx2 numpy arrays specifying red
#                   line(s) on plot
#     ax          - matplotlib axis object'''
    
#     if len(data.shape) < 2:
#         raise ValueError('must specify at least 2 dimensions')
    
#     # Make 2D histogram
#     H,xe,ye = np.histogram2d(data[:,0], data[:,1], bins=bins)

#     # Plot results
#     if ax is None:
#         fig = plt.figure()
#         cur_ax = fig.add_subplot(1,1,1)
#     else:
#         cur_ax = ax

#     # numpy histograms are organized such that the 1st dimension (eg. FSC) =
#     # rows (1st index) and the 2nd dimension (eg. SSC) = columns (2nd index).
#     # Visualized as is, this results in x-axis = SSC and y-axis = FSC with the
#     # origin at the top left corner, which is not what we're used to. Transpose
#     # the histogram array to fix the axes and set origin to 'lower' to have
#     # (0,0) at the bottom left corner instead of the top left corner.
#     img = cur_ax.imshow(H.T,origin='lower',interpolation='none')

#     if colorbar:
#         plt.colorbar(img, ax=cur_ax, label='Counts')

#     if not (gate is None):
#         if isinstance(gate, list):
#             for cntr in gate:
#                 cur_ax.plot(cntr[:,0], cntr[:,1], 'r')
#         else:
#             cur_ax.plot(gate[:,0], gate[:,1], 'r')

#     if not (axes_limits is None):
#         cur_ax.axis(axes_limits)
    
#     if not (xlabel is None):
#         cur_ax.set_xlabel(xlabel)
    
#     if not (ylabel is None):
#         cur_ax.set_ylabel(ylabel)

#     if not (title is None):
#         cur_ax.set_title(str(title))

#     if ax is None:
#         plt.show()

# def density2d(data,
#               bins=np.arange(1025)-0.5,
#               sigma=10.0,
#               axes_limits=[0,1023,0,1023],
#               xlabel='FSC',
#               ylabel='SSC',
#               title=None,
#               colorbar=True,
#               gate=None,
#               ax=None):
#     '''Plot 2D histogram which has been blurred with a 2D Gaussian kernel and
#     normalized to a valid probability mass function.

#     data        - NxD numpy array (only first 2 dimensions [columns] are used)
#     bins        - bins argument to np.histogram2d (default=np.arange(1025)-0.5)
#     sigma       - standard deviation of Gaussian kernel
#     axes_limits - axis boundaries
#     xlabel      - string to label x-axis
#     ylabel      - string to label y-axis
#     title       - string to label plot
#     colorbar    - show colorbar
#     gate        - Mx2 numpy array or list of Mx2 numpy arrays specifying red
#                   line(s) on plot
#     ax          - matplotlib axis object'''

#     if len(data.shape) < 2:
#         raise ValueError('must specify at least 2 dimensions')
        
#     # Make 2D histogram
#     H,xe,ye = np.histogram2d(data[:,0], data[:,1], bins=bins)

#     # Blur 2D histogram
#     bH = scipy.ndimage.filters.gaussian_filter(
#         H,
#         sigma=sigma,
#         order=0,
#         mode='constant',
#         cval=0.0,
#         truncate=6.0)

#     # Normalize filtered histogram to make it a valid probability mass function
#     D = bH / np.sum(bH)

#     # Plot results
#     if ax is None:
#         fig = plt.figure()
#         cur_ax = fig.add_subplot(1,1,1)
#     else:
#         cur_ax = ax

#     # numpy histograms are organized such that the 1st dimension (eg. FSC) =
#     # rows (1st index) and the 2nd dimension (eg. SSC) = columns (2nd index).
#     # Visualized as is, this results in x-axis = SSC and y-axis = FSC with the
#     # origin at the top left corner, which is not what we're used to. Transpose
#     # the density array to fix the axes and set origin to 'lower' to have (0,0)
#     # at the bottom left corner instead of the top left corner.
#     img = cur_ax.imshow(D.T,origin='lower',interpolation='none')

#     if colorbar:
#         plt.colorbar(img, ax=cur_ax, label='Probability')

#     if not (gate is None):
#         if isinstance(gate, list):
#             for cntr in gate:
#                 cur_ax.plot(cntr[:,0], cntr[:,1], 'r')
#         else:
#             cur_ax.plot(gate[:,0], gate[:,1], 'r')

#     if not (axes_limits is None):
#         cur_ax.axis(axes_limits)
    
#     if not (xlabel is None):
#         cur_ax.set_xlabel(xlabel)
    
#     if not (ylabel is None):
#         cur_ax.set_ylabel(ylabel)

#     if not (title is None):
#         cur_ax.set_title(str(title))

#     if ax is None:
#         plt.show()