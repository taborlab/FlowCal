"""
Functions to calculate statistics from the events in a FCSData object.

"""

import numpy as np
import scipy.stats

def mean(data, channels=None):
    """
    Calculate the mean of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The mean of the events in the specified channels of `data`.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    return np.mean(data_stats, axis=0)

def gmean(data, channels=None):
    """
    Calculate the geometric mean of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The geometric mean of the events in the specified channels of
        `data`.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    return scipy.stats.gmean(data_stats, axis=0)

def median(data, channels=None):
    """
    Calculate the median of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The median of the events in the specified channels of `data`.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    return np.median(data_stats, axis=0)

def mode(data, channels=None):
    """
    Calculate the mode of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The mode of the events in the specified channels of `data`.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    # scipy.stats.mode returns two outputs, the first of which is an array
    # containing the modal values. This array has the same number of
    # dimensions as the input, and with only one element in the first
    # dimension. We extract this fist element to make it match the other
    # functions in this module.
    return scipy.stats.mode(data_stats, axis=0)[0][0]

def std(data, channels=None):
    """
    Calculate the standard deviation of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The standard deviation of the events in the specified channels of
        `data`.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    return np.std(data_stats, axis=0)

def cv(data, channels=None):
    """
    Calculate the Coeff. of Variation of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The Coefficient of Variation of the events in the specified
        channels of `data`.

    Notes
    -----
    The Coefficient of Variation (CV) of a dataset is defined as the
    standard deviation divided by the mean of such dataset.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    return np.std(data_stats, axis=0) / np.mean(data_stats, axis=0)

def gstd(data, channels=None):
    """
    Calculate the geometric std. dev. of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The geometric standard deviation of the events in the specified
        channels of `data`.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    return np.exp(np.std(np.log(data_stats), axis=0))

def gcv(data, channels=None):
    """
    Calculate the geometric CV of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The geometric coefficient of variation of the events in the
        specified channels of `data`.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    return np.sqrt(np.exp(np.std(np.log(data_stats), axis=0)**2) - 1)

def iqr(data, channels=None):
    """
    Calculate the Interquartile Range of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The Interquartile Range of the events in the specified channels of
        `data`.

    Notes
    -----
    The Interquartile Range (IQR) of a dataset is defined as the interval
    between the 25% and the 75% percentiles of such dataset.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    q75, q25 = np.percentile(data_stats, [75 ,25], axis=0)
    return q75 - q25

def rcv(data, channels=None):
    """
    Calculate the RCV of the events in an FCSData object.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int or str or list of int or list of str, optional
        Channels on which to calculate the statistic. If None, use all
        channels.

    Returns
    -------
    float or numpy array
        The Robust Coefficient of Variation of the events in the specified
        channels of `data`.

    Notes
    -----
    The Robust Coefficient of Variation (RCV) of a dataset is defined as
    the Interquartile Range (IQR) divided by the median of such dataset.

    """
    # Slice data to take statistics from
    if channels is None:
        data_stats = data
    else:
        data_stats = data[:, channels]

    # Calculate and return statistic
    q75, q25 = np.percentile(data_stats, [75 ,25], axis=0)
    return (q75 - q25)/np.median(data_stats, axis=0)
