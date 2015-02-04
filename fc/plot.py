#!/usr/bin/python
#
# plot.py - Module containing plotting functions for flow cytometry data sets.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 2/4/2015
#
# Requires:
#   * numpy
#   * matplotlib

import numpy as np
import matplotlib.pyplot as plt

# Mapping of string to RGB tuple for some pre-selected colors
colors = {
    'lr':(242.0/255, 220.0/255, 219.0/255),     # light red
    'dr':(192.0/255, 080.0/255, 077.0/255),     # dark red
    'lg':(235.0/255, 241.0/255, 222.0/255),     # light green
    'dg':(155.0/255, 187.0/255, 089.0/255),     # dark green
    'lb':(198.0/255, 217.0/255, 241.0/255),     # light blue
    'db':(031.0/255, 073.0/255, 125.0/255)      # dark blue
}

def fsc_ssc_counts(data,
            axes_limits=[0,1023,0,1023],
            title=None,
            colorbar=True,
            gate=None,
            ax=None):
    '''Plot 2D histogram of FSC v SSC counts.

    data        - NxD numpy array (row=event), 1st column=FSC, 2nd column=SSC
    axes_limits - axis boundaries (default=[0,1023,0,1023])
    title       - string to label plot (default=None)
    colorbar    - show colorbar (default=True)
    gate        - Mx2 numpy array, specifies red line on plot (default=None)
    ax          - matplotlib axis object (default=None)

    returns     - 2D histogram, numpy array of counts'''
    
    # make 2D histogram of FSC v SSC
    e = np.arange(1025)-0.5      # bin edges (centered over 0 - 1023)
    H,xe,ye = np.histogram2d(data[:,0], data[:,1], bins=e)
    H = H.T     # numpy transposes axes to be consistent with histogramnd

    # plot results
    if ax is None:
        fig = plt.figure()
        cur_ax = fig.add_subplot(1,1,1)
    else:
        cur_ax = ax

    img = cur_ax.imshow(H,origin='lower')

    if colorbar:
        plt.colorbar(img, ax=cur_ax)

    if not (gate is None):
        cur_ax.plot(gate[:,0], gate[:,1], 'r')

    cur_ax.axis(axes_limits)
    cur_ax.set_xlabel('FSC')
    cur_ax.set_ylabel('SSC')

    if not (title is None):
        cur_ax.set_title(str(title))

    if ax is None:
        plt.show()

    return H

def fsc_ssc_density(data,
            axes_limits=[0,1023,0,1023],
            title=None,
            colorbar=True,
            gate=None,
            ax=None):
    '''Plot FSC v SSC density.

    data        - NxD numpy array (row=event), 1st column=FSC, 2nd column=SSC
    axes_limits - axis boundaries (default=[0,1023,0,1023])
    title       - string to label plot (default=None)
    colorbar    - show colorbar (default=True)
    gate        - Mx2 numpy array, specifies red line on plot (default=None)
    ax          - matplotlib axis object (default=None)

    returns     - 2D histogram, numpy array of counts'''

    pass    

def hist(data,
         xlim=[0,1023],
         ylim=None,
         edge_color='db',
         face_color='lb',
         title=None,
         xlabel=None,
         ax=None):
    '''Plot 1D histogram of specified numpy array

    data       - Nx1 numpy array (row=event)
    xlim       - x-axis limits (default=[0,1023])
    ylim       - y-axis limits (default=None)
    edge_color - color of histogram edge. Can either be a string indicating a
                 pre-selected color or a matplotlib color spec (default='lb'
                 [light blue])
    face_color - color of histogram face. Can either be a string indicating a
                 pre-selected color or a matplotlib color spec (default='db'
                 [dark blue])
    title      - string to label plot (default=None)
    xlabel     - string to label x-axis (default=None)
    ax         - matplotlib axis object (default=None)

    returns - 1D histogram, numpy array of counts'''

    if len(data.shape) > 1:
        raise ValueError('more than 1 dimension specified')
    
    # plot results
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

    e = np.arange(1025)-0.5      # bin edges (centered over 0 - 1023)
    n,bins,patches = cur_ax.hist(
        data,
        bins=e,
        histtype='stepfilled',
        color=fc,
        edgecolor=ec)

    if not (xlim is None):
        cur_ax.set_xlim(xlim)

    if not (ylim is None):
        cur_ax.set_ylim(ylim)

    cur_ax.set_ylabel('Counts')

    if not (title is None):
        cur_ax.set_title(str(title))

    if not (xlabel is None):
        cur_ax.set_xlabel(str(xlabel))

    if ax is None:
        plt.show()

    return n
