#!/usr/bin/python
#
# transform.py - Module containing flow cytometry transformations.
#
# All transformations should be of the following form:
#
#     data_out = transform(data_in, parameters)
#
# where DATA_IN and DATA_OUT are NxD numpy arrays describing N cytometry events
# observing D data dimensions, and PARAMETERS are transformation specific
# parameters.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 6/30/2015
#
# Requires:
#   * numpy
#   * scipy

import numpy as np
import scipy.ndimage.filters

def exponentiate(data):
    '''Exponentiate data using the following transformation:
        y = 10^(x/256)
    This transformation spaces out the data across 4 logs from 1 to 10,000.
    This is also the transformation typically applied by flow cytometer
    software.

    data    - NxD numpy array (row=event)

    returns - NxD numpy array'''
    
    return np.power(10, (data.astype(float)/256.0))

def _find_hist_peaks(data, labels, labels_all = None, 
        min_val = 0, max_val = 1023):
    '''
    Find histogram peaks from a dataset.

    This function assumes that clustering of the data has been previously 
    performed, and the labels for the different clusters are given in the 
    arguments. The algorithm then proceeds as follows:
        1. For each one of the C clusters, calculate the histogram for the 
            dataset.
        2. Use a 1D Gaussian filter to smooth out the histogram. The sigma 
            value for the Gaussian filter is chosen based on the standard 
            deviation of the fit Gaussian from the GMM for that paritcular 
            segment.
        3. Identify peak as the maximum value of the Gaussian-blurred histogram.

    data        - Nx1 numpy array with the 1D data from where peaks should be 
                  identified. Data values are assumed to be integer.
    labels      - Nx1 numpy array with the cluster labels of each data sample.
    labels_all  - Cx1 numpy array with all the individual cluster labels.
    min_val     - minimum possible value in data.
    max_val     - maximum possible value in data.

    returns     - Cx1 numpy array with the values of the identified peaks.
                - Cx(max_val - min_val + 1) numpy array with a smoothed 
                  histogram for each cluster.
    '''

    # Check if individual labels have been provided, otherwise calculate
    if labels_all is None:
        labels_all = list(set(labels))

    # Calculate bin edges and centers
    bin_edges = np.arange(min_val, max_val + 2) - 0.5
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_centers = np.arange(min_val, max_val + 1)

    # Identify peaks for each cluster
    peaks = np.zeros(len(labels_all))
    hists_smooth = np.zeros([len(labels_all), len(bin_centers)])
    for i, li in enumerate(labels_all):
        # Extract data that belongs to this cluster
        data_li = data[labels == li]
        # Calculate sample mean and standard deviation
        # mu_li = np.mean(data_li)
        sigma_li = np.std(data_li)
        # Calculate histogram
        hist, __ = np.histogram(data_li, bin_edges)
        # Do Gaussian blur on histogram
        # We have found empirically that using one half of the distribution's 
        # standard deviation results in a nice fit.
        hist_smooth = scipy.ndimage.filters.gaussian_filter1d(hist, sigma_li/2.)
        # Extract peak
        i_max = np.argmax(hist_smooth)
        peak = bin_centers[i_max]
        # Pack values
        peaks[i] = peak
        hists_smooth[i,:] = hist_smooth

    return peaks, hists_smooth

def _fit_mef_standards_curve(known_peaks, fit_peaks, method='log line+auto'):
    '''description'''
    
    if method == 'log line':
        pass
    elif method == 'exp':
        pass
    elif method == 'exp+auto':
        pass
    elif method == 'log line+auto':
        pass
    else: 
        raise ValueError('Unrecognized method')
    
    def standards_curve(data):
        '''Transform data from Channels to Molecules of Equivalent
        Fluorophore.'''
        
        return data
    
    return standards_curve

def to_mef(data, beads, known_mef, peaks=None, sc=None):
    '''description'''
    
    if sc is None:
        if peaks is None:
            peaks = _find_hist_peaks(beads)
            
        sc = _fit_mef_standards_curve(known_mef, peaks)
            
    return sc(data)

def blank(data, white_cells):
    pass

def compensate(data):
    pass
