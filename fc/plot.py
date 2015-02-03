#!/usr/bin/python
#
# plot.py - Module containing plotting functions for flow cytometry data sets.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 2/2/2015
#
# Requires:
#   * numpy
#   * matplotlib

import numpy as np
import matplotlib.pyplot as plt

def fsc_ssc(data,
            axes_limits=[0,1023,0,1023],
            title=None,
            colorbar=True,
            gate=None,
            ax=None):
    '''Plot FSC v SSC.

    data        - NxD numpy array (row=event), 1st column=FSC, 2nd column=SSC
    axes_limits - axis boundaries (default=[0,1023,0,1023])
    title       - string to label plot (default=None)
    colorbar    - show colorbar (default=True)
    gate        - Mx2 numpy array, specifies red line on plot (default=None)
    ax          - matplotlib axis object (default=None)'''
    
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
