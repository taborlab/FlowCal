#!/usr/bin/python
#
# gate.py - Module containing flow cytometry gate functions.
#
# All gate functions should be of the following form:
#
#     mask = gate(data, parameters)
#
# where DATA is a NxD numpy array describing N cytometry events observing D
# data dimensions, PARAMETERS are gate specific parameters, and MASK is a
# Boolean numpy array of length N indicating which events were gated out
# (False) and which events were kept (True) such that DATA[MASK,:] represents
# the gated data set.
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 2/1/2015
#
# Requires:
#   * numpy

import numpy as np

def high_low(data, high=(2**10)-1, low=0):
    '''Gate out high and low values.

    data    - NxD numpy array (row=event)
    high    - high value to discard (default=1023)
    low     - low value to discard (default=0)

    returns - Boolean numpy array of length N'''
    
    return ~np.any((data==high)|(data==low),axis=1)

def extrema(data, extrema=[(2**10)-1, 0]):
    '''Gate out list of extreme values.

    data    - NxD numpy array (row=event)
    extrema - list of values to discard (default=[1023,0])

    returns - Boolean numpy array of length N'''
    
    mask = np.zeros(shape=data.shape,dtype=bool)
    for e in extrema:
        mask |= data==e
    return ~mask.any(axis=1)

def circular_median(data, gate_fraction=0.65):
    '''Gate out all events but those with (FSC,SSC) values closest to the 2D
    (FSC,SSC) median.

    data          - NxD numpy array (row=event), 1st column=FSC, 2nd column=SSC
    gate_fraction - fraction of data points to keep (default=0.65)

    returns       - Boolean numpy array of length N'''

    # Determine number of points to keep
    n = np.ceil(gate_fraction*float(data.shape[0]))

    # Calculate distance to median point
    m = np.median(data[:,0:2],0)
    d = np.linalg.norm()

def whitening(data):
    raise NotImplementedError()

def density(data, sigma, gate_fraction):
    raise NotImplementedError()
