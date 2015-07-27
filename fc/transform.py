#!/usr/bin/python
#
# transform.py - Module containing flow cytometry transformations.
#
# All transformations should be of the following form:
#
#     data_t = transform(data, channels, params):
#
# where 'data' and 'data_t' are NxD FCSData object or numpy arrays, representing
# N events with D channels, 'channels' indicate the channels in which to apply
# the transformation, and params are transformation-specific parameters.
# Each transformation function can apply its own restrictions or default on 
# 'channels'.
# If data is an FCSData object, transform should rescale 
# data.channel_info['range'] if necessary.
# 
#
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/7/2015
#
# Requires:
#   * numpy

import numpy as np

def transform(data, channels, transform_fxn, def_channels = None):
    '''Generic transformation function, to be used by other functions.

    This function performs basic checks on channels and data. Then it applies
    transform_fxn to the specified channels. Finally, it rescales range in 
    data.channel_info if necessary.

    data            - NxD FCSData object or numpy array
    channels        - channels in which to perform the transformation. If 
                        channels is None, use def_channels.
    def_channels    - default set of channels in which to perform the 
                        transformations. If None, use all channels.
    transform_fxn   - Function that performs the actual transformation.

    returns         - NxD FCSData object or numpy array.
    '''
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
    # Apply transformation to range, bin_values, and bin_edges
    if hasattr(data_t, 'channel_info'):
        for channel in channels:
            if isinstance(channel, basestring):
                ch = data_t.name_to_index(channel)
            else:
                ch = channel
            if 'range' in data_t.channel_info[ch]:
                r = data_t.channel_info[ch]['range']
                r[0] = transform_fxn(r[0])
                r[1] = transform_fxn(r[1])
                data_t.channel_info[ch]['range'] = r
            if 'bin_vals' in data_t.channel_info[ch]:
                b = transform_fxn(data_t.channel_info[ch]['bin_vals'])
                data_t.channel_info[ch]['bin_vals'] = b
            if 'bin_edges' in data_t.channel_info[ch]:
                b = transform_fxn(data_t.channel_info[ch]['bin_edges'])
                data_t.channel_info[ch]['bin_edges'] = b

    return data_t


def exponentiate(data, channels = None):
    '''Exponentiate data using the following transformation:
        y = 10^(x/256)
    This transformation spaces out 10-bit data across 4 logs from 1 to 10,000.
    This rescales the fluorecence units to those observed in most flow 
    cytometry aquisition software, if log scale has been used.

    data     - NxD FCSData object or numpy array
    channels - channels in which to perform the transformation. If channels is
                None, perform transformation in all channels.

    returns  - NxD FCSData object or numpy array. '''

    # Transform and return
    def transform_fxn(x): return 10**(x/256.0)
    return transform(data, channels, transform_fxn)


def to_mef(data, channels, sc_list, sc_channels = None):
    '''Transform data to MEF units using standard curves obtained from 
    calibration beads.

    This function allows for the specification of the channels in which the
    MEF transformation should be applied. The function automatically checks 
    whether a standard curve is available for the specified channel, and throws
    and error otherwise.

    It is recommended that after calculating the standard curves, this function
    be reduced using functools.partial to the following signature:
        to_mef(data, channels)

    Arguments:
    data        - NxD FCSData object or numpy array
    channels    - Channels in which to perform the transformation. If channels 
                    is None, use all channels in sc_channels
    sc_list     - List of functions implementing the standard curves for each
                    of the channels in sc_channels.
    sc_channels - List of channels in which the standard curves are available. 
                    If None, use all channels in data.
    '''

    # Default sc_channels
    if sc_channels is None:
        if data.ndim == 1:
            sc_channels = range(data.shape[0])
        else:
            sc_channels = range(data.shape[1])
    # Check that sc_channels and sc_list have the same length
    assert len(sc_channels) == len(sc_list), \
        "sc_channels and sc_list should have the same length."
    # Convert sc_channels to index
    if isinstance(sc_channels[0], basestring):
        sc_channels = data.name_to_index(sc_channels)

    # Default channels
    if channels is None:
        channels = sc_channels
    # Convert channels to iterable
    if not hasattr(channels, '__iter__'):
        channels = [channels]
    # Convert channels to index
    if isinstance(channels[0], basestring):
        channels_ind = data.name_to_index(channels)
    else:
        channels_ind = channels
    # Check if every channel is in sc_channels
    for chi, chs in zip(channels_ind, channels):
        if chi not in sc_channels:
            raise ValueError("No standard curve for channel {}.".format(chs))

    # Copy data array
    data_t = data.copy().astype(np.float64)

    # Iterate over channels
    for chi, sc in zip(sc_channels, sc_list):
        if chi not in channels_ind:
            continue
        # Apply transformation
        data_t[:,chi] = sc(data_t[:,chi])
        # Apply transformation to range, bin_values, and bin_edges
        if hasattr(data_t, 'channel_info'):
            if 'range' in data_t.channel_info[chi]:
                r = data_t.channel_info[chi]['range']
                r[0] = sc(r[0])
                r[1] = sc(r[1])
                data_t.channel_info[chi]['range'] = r
            if 'bin_vals' in data_t.channel_info[chi]:
                b = sc(data_t.channel_info[chi]['bin_vals'])
                data_t.channel_info[chi]['bin_vals'] = b
            if 'bin_edges' in data_t.channel_info[chi]:
                b = sc(data_t.channel_info[chi]['bin_edges'])
                data_t.channel_info[chi]['bin_edges'] = b

    return data_t
