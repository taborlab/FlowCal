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
# Date: 6/29/2015
#
# Requires:
#   * numpy

import numpy as np

def exponentiate(data):
    '''Exponentiate data using the following transformation:
        y = 10^(x/256)
    This transformation spaces out the data across 4 logs from 1 to 10,000.
    This is also the transformation typically applied by flow cytometer
    software.

    data    - NxD numpy array (row=event)

    returns - NxD numpy array'''
    
    return np.power(10, (data.astype(float)/256.0))

def _find_hist_peaks(data, num_peaks):
    '''description'''

    # do some kind of classificaiton. kmeans? gmm? dbscan?
    
    # serialize on classification
    labels = None

    # calculate sample mean and standard deviation
    mu = None
    sigma = None

    # do Gaussian blur

    # extract peaks

    return np.array([])

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
