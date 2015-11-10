"""
Functions to calculate statistics from the events in a FCSData object.
"""

import numpy as np
import scipy.stats

def mean(data, channel):
    """Calculate the mean of the events on a specified channel of a
    FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    float
        The mean of the events in the specified channel of `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return np.mean(data[:,channel])

def gmean(data, channel):
    """Calculate the geometric mean of the events on a specified channel
    of a FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    float
        The geometric mean of the events in the specified channel of
        `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return scipy.stats.gmean(data[:,channel])

def median(data, channel):
    """Calculate the median of the events on a specified channel of a
    FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    int or float
        The median of the events in the specified channel of `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return np.median(data[:,channel])

def mode(data, channel):
    """Calculate the mode of the events on a specified channel of a
    FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    int or float
        The mode of the events in the specified channel of `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return np.argmax(np.bincount(data[:,channel].astype('int32')))

def std(data, channel):
    """Calculate the standard deviation of the events on a specified
    channel of a FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    float
        The standard deviation of the events in the specified channel of
        `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return np.std(data[:,channel])

def CV(data, channel):
    """Calculate the Coefficient of Variation of the events on a specified
    channel of a FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    float
        The Coefficient of Variation of the events in the specified channel
        of `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    return np.std(data[:,channel])/np.mean(data[:,channel])

def iqr(data, channel):
    """Calculate the Interquartile Range of the events on a specified
    channel of a FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    int or float
        The Interquartile Range of the events in the specified channel of
        `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    q75, q25 = np.percentile(data[:,channel], [75 ,25])
    return q75 - q25

def RCV(data, channel):
    """Calculate the Robust Coefficient of Variation of the events on a
    specified channel of a FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channel : int or str
        Channel on which to calculate the statistic

    Returns
    -------
    float
        The Robust Coefficient of Variation of the events in the specified
        channel of `data`.

    """
    if hasattr(channel, '__iter__'):
        raise ValueError("Channel should be a scalar.")

    q75, q25 = np.percentile(data[:,channel], [75 ,25])
    return (q75 - q25)/np.median(data[:,channel])
