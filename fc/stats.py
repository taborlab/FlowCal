#!/usr/bin/python
#
# stats.py - Module containing stats functions to be applied to FCSData 
# objects.
#
# Authors: Brian Landry (brian.landry@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/13/2015
#
# Requires:
#   * numpy
#   * scipy

import numpy
import scipy.stats

def mean(data, channel):
    ''' Calculate the mean.

    Attributes:
    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return numpy.mean(data[:,channel])

def gmean(data, channel):
    ''' Calculate the geometric mean.

    Attributes:
    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return scipy.stats.gmean(data[:,channel])

def median(data, channel):
    ''' Calculate the median.

    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return numpy.median(data[:,channel])

def mode(data, channel):
    ''' Calculate the mode.

    Attributes:
    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return numpy.argmax(numpy.bincount(data[:,channel].astype('int32')))

def std(data, channel):
    ''' Calculate the standard deviation.

    Attributes:
    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return numpy.std(data[:,channel])

def CV(data, channel):
    ''' Calculate the Coefficient of Variation.

    Attributes:
    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return numpy.std(data[:,channel])/numpy.mean(data[:,channel])

def iqr(data, channel):
    ''' Calculate the Interquartile Range.

    Attributes:
    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    q75, q25 = numpy.percentile(data[:,channel], [75 ,25])
    return q75 - q25

def RCV(data, channel):
    ''' Calculate the Robust Coefficient of Variation

    Attributes:
    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    q75, q25 = numpy.percentile(data[:,channel], [75 ,25])
    return (q75 - q25)/numpy.median(data[:,channel])

def rate(data, channel):
    ''' Calculate the flow rate of events.

    data    - NxD FCSData object or numpy array
    channel - Channel in which to calculate the statistic
    '''
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return float(len(data[:,channel]))/(data[-1,channel]-data[0,channel])