#!/usr/bin/python
#
# gate.py - Module containing flow cytometry gate functions.
#
# All gate functions should be of one of the following forms:
#
#     mask = gate(data, parameters)
#     mask, contour = gate(data, parameters)
#
# where DATA is a NxD numpy array describing N cytometry events observing D
# data dimensions, PARAMETERS are gate specific parameters, MASK is a
# Boolean numpy array of length N indicating which events were gated out
# (False) and which events were kept (True) such that DATA[MASK,:] represents
# the gated data set, and CONTOUR is an optional 2D numpy array of x-y
# coordinates tracing out a line which represents the gate (useful for
# plotting).
#
# Author: John T. Sexton (john.t.sexton@rice.edu)
# Date: 2/2/2015
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

    returns       - Boolean numpy array of length N, 2D numpy array of x-y
                    coordinates of gate contour'''

    if data.shape[0] < 2:
        raise ValueError('data must have more than 1 event')

    # Determine number of points to keep
    n = int(np.ceil(gate_fraction*float(data.shape[0])))

    # Calculate distance to median point
    m = np.median(data[:,0:2],0)
    d = np.sqrt(np.sum(np.square(m-data[:,0:2]),1))

    # Select closest points
    idx = sorted(xrange(d.shape[0]), key=lambda k: d[k])
    mask = np.zeros(shape=data.shape[0],dtype=bool)
    mask[idx[:n]] = True

    # Last point defines boundary of circle which can serve as gate contour
    theta = np.arange(0,2*np.pi,2*np.pi/100)
    r = d[idx[n-1]]
    x = [m[0] + (r*np.cos(t)) for t in theta]
    y = [m[1] + (r*np.sin(t)) for t in theta]

    # Close the circle
    x.append(x[0])
    y.append(y[0])

    return mask, np.array([x,y]).T

def whitening(data):
    raise NotImplementedError()

def density(data, sigma, gate_fraction):
    raise NotImplementedError()
