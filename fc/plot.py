#!/usr/bin/python
#
# plot.py - Module containing plotting functions for flow cytometry data sets.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 6/28/2015
#
# Requires:
#   * numpy
#   * matplotlib
#   * scipy

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters

# Mapping of string to RGB tuple for some pre-selected colors
colors = {
    'lr':(242.0/255, 220.0/255, 219.0/255),     # light red
    'dr':(192.0/255, 080.0/255, 077.0/255),     # dark red
    'lg':(235.0/255, 241.0/255, 222.0/255),     # light green
    'dg':(155.0/255, 187.0/255, 089.0/255),     # dark green
    'lb':(198.0/255, 217.0/255, 241.0/255),     # light blue
    'db':(031.0/255, 073.0/255, 125.0/255)      # dark blue
}

def hist2d(data,
           bins=np.arange(1025)-0.5,
           axes_limits=[0,1023,0,1023],
           xlabel='FSC',
           ylabel='SSC',
           title=None,
           colorbar=True,
           gate=None,
           ax=None):
    '''Plot 2D histogram.

    data        - NxD numpy array (only first 2 dimensions [columns] are used)
    bins        - bins argument to np.histogram2d (default=np.arange(1025)-0.5)
    axes_limits - axis boundaries
    xlabel      - string to label x-axis
    ylabel      - string to label y-axis
    title       - string to label plot
    colorbar    - show colorbar
    gate        - Mx2 numpy array or list of Mx2 numpy arrays specifying red
                  line(s) on plot
    ax          - matplotlib axis object'''
    
    if len(data.shape) < 2:
        raise ValueError('must specify at least 2 dimensions')
    
    # Make 2D histogram
    H,xe,ye = np.histogram2d(data[:,0], data[:,1], bins=bins)

    # Plot results
    if ax is None:
        fig = plt.figure()
        cur_ax = fig.add_subplot(1,1,1)
    else:
        cur_ax = ax

    # numpy histograms are organized such that the 1st dimension (eg. FSC) =
    # rows (1st index) and the 2nd dimension (eg. SSC) = columns (2nd index).
    # Visualized as is, this results in x-axis = SSC and y-axis = FSC with the
    # origin at the top left corner, which is not what we're used to. Transpose
    # the histogram array to fix the axes and set origin to 'lower' to have
    # (0,0) at the bottom left corner instead of the top left corner.
    img = cur_ax.imshow(H.T,origin='lower',interpolation='none')

    if colorbar:
        plt.colorbar(img, ax=cur_ax, label='Counts')

    if not (gate is None):
        if isinstance(gate, list):
            for cntr in gate:
                cur_ax.plot(cntr[:,0], cntr[:,1], 'r')
        else:
            cur_ax.plot(gate[:,0], gate[:,1], 'r')

    if not (axes_limits is None):
        cur_ax.axis(axes_limits)
    
    if not (xlabel is None):
        cur_ax.set_xlabel(xlabel)
    
    if not (ylabel is None):
        cur_ax.set_ylabel(ylabel)

    if not (title is None):
        cur_ax.set_title(str(title))

    if ax is None:
        plt.show()

def density2d(data,
              bins=np.arange(1025)-0.5,
              sigma=10.0,
              axes_limits=[0,1023,0,1023],
              xlabel='FSC',
              ylabel='SSC',
              title=None,
              colorbar=True,
              gate=None,
              ax=None):
    '''Plot 2D histogram which has been blurred with a 2D Gaussian kernel and
    normalized to a valid probability mass function.

    data        - NxD numpy array (only first 2 dimensions [columns] are used)
    bins        - bins argument to np.histogram2d (default=np.arange(1025)-0.5)
    sigma       - standard deviation of Gaussian kernel
    axes_limits - axis boundaries
    xlabel      - string to label x-axis
    ylabel      - string to label y-axis
    title       - string to label plot
    colorbar    - show colorbar
    gate        - Mx2 numpy array or list of Mx2 numpy arrays specifying red
                  line(s) on plot
    ax          - matplotlib axis object'''

    if len(data.shape) < 2:
        raise ValueError('must specify at least 2 dimensions')
        
    # Make 2D histogram
    H,xe,ye = np.histogram2d(data[:,0], data[:,1], bins=bins)

    # Blur 2D histogram
    bH = scipy.ndimage.filters.gaussian_filter(
        H,
        sigma=sigma,
        order=0,
        mode='constant',
        cval=0.0,
        truncate=6.0)

    # Normalize filtered histogram to make it a valid probability mass function
    D = bH / np.sum(bH)

    # Plot results
    if ax is None:
        fig = plt.figure()
        cur_ax = fig.add_subplot(1,1,1)
    else:
        cur_ax = ax

    # numpy histograms are organized such that the 1st dimension (eg. FSC) =
    # rows (1st index) and the 2nd dimension (eg. SSC) = columns (2nd index).
    # Visualized as is, this results in x-axis = SSC and y-axis = FSC with the
    # origin at the top left corner, which is not what we're used to. Transpose
    # the density array to fix the axes and set origin to 'lower' to have (0,0)
    # at the bottom left corner instead of the top left corner.
    img = cur_ax.imshow(D.T,origin='lower',interpolation='none')

    if colorbar:
        plt.colorbar(img, ax=cur_ax, label='Probability')

    if not (gate is None):
        if isinstance(gate, list):
            for cntr in gate:
                cur_ax.plot(cntr[:,0], cntr[:,1], 'r')
        else:
            cur_ax.plot(gate[:,0], gate[:,1], 'r')

    if not (axes_limits is None):
        cur_ax.axis(axes_limits)
    
    if not (xlabel is None):
        cur_ax.set_xlabel(xlabel)
    
    if not (ylabel is None):
        cur_ax.set_ylabel(ylabel)

    if not (title is None):
        cur_ax.set_title(str(title))

    if ax is None:
        plt.show()

def hist1d(data,
           bins=np.arange(1025)-0.5,
           normed=False,
           xlim=[0,1023],
           ylim=None,
           edge_color='db',
           face_color='lb',
           title=None,
           xlabel=None,
           ax=None,
           **kwargs):
    '''Plot 1D histogram.

    data       - Nx1 or NxD numpy array (only first dimension [column] is used)
    bins       - bins argument to plt.hist (default=np.arange(1025)-0.5)
    normed     - normalize histogram to make probability density
    xlim       - x-axis limits
    ylim       - y-axis limits
    edge_color - color of histogram edge. Can either be a string indicating a
                 pre-selected color or a matplotlib color spec (default='lb'
                 [light blue])
    face_color - color of histogram face. Can either be a string indicating a
                 pre-selected color or a matplotlib color spec (default='db'
                 [dark blue])
    title      - string to label plot
    xlabel     - string to label x-axis
    ax         - matplotlib axis object'''
    
    
    if len(data.shape) == 1:    # 1D array
        d = data
    elif len(data.shape) == 2:  # 2D array
        d = data[:,0]
    else:
        raise ValueError('data must be either Nx1 or NxD numpy array')
    
    if ax is None:
        fig = plt.figure()
        cur_ax = fig.add_subplot(1,1,1)
    else:
        cur_ax = ax

    try:
        ec = colors[edge_color]
    except KeyError:
        ec = edge_color

    try:
        fc = colors[face_color]
    except KeyError:
        fc = face_color

    n,bins,patches = cur_ax.hist(
        d,
        bins=bins,
        normed=normed,
        histtype='stepfilled',
        color=fc,
        edgecolor=ec,
        **kwargs)

    if not (xlim is None):
        cur_ax.set_xlim(xlim)

    if not (ylim is None):
        cur_ax.set_ylim(ylim)

    if normed:
        cur_ax.set_ylabel('Probability')
    else:
        cur_ax.set_ylabel('Counts')

    if not (title is None):
        cur_ax.set_title(str(title))

    if not (xlabel is None):
        cur_ax.set_xlabel(str(xlabel))

    if ax is None:
        plt.show()
