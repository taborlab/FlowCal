"""
Functions for transforming flow cytometry data

All transformations are of the following form:

    data_t = transform(data, channels, params):

where `data` and `data_t` are NxD FCSData objects or numpy arrays,
representing N events with D channels, `channels` indicate the channels in
which to apply the transformation, and `params` are transformation-specific
parameters. Each transformation function can apply its own restrictions or
default on `channels`.

If `data` is an FCSData object, `transform` should rescale ``data.domain``
and ``data.hist_bin_edges`` if necessary.

"""

import numpy as np

def transform(data, channels, transform_fxn, def_channels = None):
    """
    Apply some transformation function to flow cytometry data.

    This function is a template transformation function, intended to be
    used by other specific transformation functions. It performs basic
    checks on `channels` and `data`. It then applies `transform_fxn` to the
    specified channels. Finally, it rescales  ``data.domain`` and
    ``data.hist_bin_edges`` if necessary.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int, str, list of int, list of str, optional
        Channels on which to perform the transformation. If `channels` is
        None, use def_channels.
    transform_fxn : function
        Function that performs the actual transformation.
    def_channels : int, str, list of int, list of str, optional
        Default set of channels in which to perform the transformation.
        If `def_channels` is None, use all channels.

    Returns
    -------
    data_t : FCSData or numpy array
        NxD transformed flow cytometry data.

    """
    # Copy data array
    data_t = data.copy().astype(np.float64)

    # Default
    if channels is None:
        if def_channels is None:
            channels = range(data_t.shape[1])
        else:
            channels = def_channels

    # Convert channels to iterable
    if not hasattr(channels, '__iter__'):
        channels = [channels]

    # Apply transformation
    data_t[:,channels] = transform_fxn(data_t[:,channels])

    # Apply transformation to ``data.domain`` and ``data.hist_bin_edges``
    if hasattr(data_t, '_domain'):
        for channel in channels:
            # Transform channel name to index if necessary
            channel_idx = data_t._name_to_index(channel)
            if data_t._domain[channel_idx] is not None:
                data_t._domain[channel_idx] = \
                    transform_fxn(data_t._domain[channel_idx])
    if hasattr(data_t, '_hist_bin_edges'):
        for channel in channels:
            # Transform channel name to index if necessary
            channel_idx = data_t._name_to_index(channel)
            if data_t._hist_bin_edges[channel_idx] is not None:
                data_t._hist_bin_edges[channel_idx] = \
                    transform_fxn(data_t._hist_bin_edges[channel_idx])

    return data_t


def exponentiate(data, channels = None):
    """
    Exponentiate flow cytometry data.

    This function applies the following transformation:

        y = 10^(x/256)

    This spaces out 10-bit data across 4 logs from 1 to 1e4, which rescales
    the fluorecence units to those observed in most flow cytometry
    aquisition software, if a log amplifier has been used.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int, str, list of int, list of str, optional
        Channels on which to perform the transformation. If `channels` is
        None, perform transformation in all channels.

    Returns
    -------
    FCSData or numpy array
        NxD transformed flow cytometry data.

    """
    # Transform and return
    def transform_fxn(x): return 10**(x/256.0)
    return transform(data, channels, transform_fxn)


def to_mef(data, channels, sc_list, sc_channels = None):
    """
    Transform flow cytometry data using a standard curve function.

    This function accepts a list of standard curves (`sc_list`) and a list
    of channels to which those standard curves should be applied
    (`sc_channels`). `to_mef` automatically checks whether a standard curve
    is available for each channel specified in `channels`, and throws an
    error otherwise.

    This function is intended to be reduced to the following signature:

        to_mef_reduced(data, channels)

    by using ``functools.partial`` once a list of standard curves and their
    respective channels is available.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int, str, list of int, list of str
        Channels on which to perform the transformation. If `channels` is
        None, perform transformation in all channels specified on
        `sc_channels`.
    sc_list : list of functions
        Functions implementing the standard curves for each channel in
        `sc_channels`.
    sc_channels : list of int or list of str, optional
        List of channels corresponding to each function in `sc_list`. If
        None, use all channels in `data`.

    Returns
    -------
    FCSData or numpy array
        NxD transformed flow cytometry data.

    Raises
    ------
    ValueError
        If any channel specified in `channels` is not in `sc_channels`.

    """
    # Default sc_channels
    if sc_channels is None:
        if data.ndim == 1:
            sc_channels = range(data.shape[0])
        else:
            sc_channels = range(data.shape[1])
    # Check that sc_channels and sc_list have the same length
    if len(sc_channels) != len(sc_list):
        raise ValueError("sc_channels and sc_list should have the same length")
    # Convert sc_channels to indices
    if hasattr(data, '_name_to_index'):
        sc_channels = data._name_to_index(sc_channels)

    # Default channels
    if channels is None:
        channels = sc_channels
    # Convert channels to iterable
    if not hasattr(channels, '__iter__'):
        channels = [channels]
    # Convert channels to index
    if hasattr(data, '_name_to_index'):
        channels_ind = data._name_to_index(channels)
    else:
        channels_ind = channels
    # Check if every channel is in sc_channels
    for chi, chs in zip(channels_ind, channels):
        if chi not in sc_channels:
            raise ValueError("no standard curve for channel {}".format(chs))

    # Copy data array
    data_t = data.copy().astype(np.float64)

    # Iterate over channels
    for chi, sc in zip(sc_channels, sc_list):
        if chi not in channels_ind:
            continue
        # Apply transformation
        data_t[:,chi] = sc(data_t[:,chi])
        # Apply transformation to domain and hist_bin_edges
        if hasattr(data_t, '_domain') and data_t._domain[chi] is not None:
            data_t._domain[chi] = sc(data_t._domain[chi])
        if (hasattr(data_t, '_hist_bin_edges')
                and data_t._hist_bin_edges[chi] is not None):
            data_t._hist_bin_edges[chi] = sc(data_t._hist_bin_edges[chi])

    return data_t
