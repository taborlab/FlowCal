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
#   * scipy
#   * scikit-learn

from collections import namedtuple
import numpy as np
import scipy.ndimage.filters
from sklearn.cluster import DBSCAN 
from scipy.optimize import minimize


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
    # Apply transformation to range
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
        # Apply transformation to range
        if hasattr(data_t, 'channel_info'):
            if 'range' in data_t.channel_info[chi]:
                r = data_t.channel_info[chi]['range']
                r[0] = sc(r[0])
                r[1] = sc(r[1])
                data_t.channel_info[chi]['range'] = r

    return data_t


def _clustering_dbscan(data, eps = 20.0, min_samples = 40):
    '''
    Find clusters in the data array using the DBSCAN method from the 
    scikit-learn library.

    data        - NxD numpy array.
    eps         - Parameter for DBSCAN. Check scikit-learn documentation for
                  more info.
    min_samples - Parameter for DBSCAN. Check scikit-learn documentation for
                  more info.

    returns     - Nx1 numpy array, labeling each sample to a cluster.
    '''

    # Initialize DBSCAN object
    db = DBSCAN(eps = eps, min_samples = min_samples)
    # Fit data
    db.fit(data)
    # Extract labels
    # A value of -1 indicates no assignment to any cluster
    labels = db.labels_

    # Extract individual labels and number of labels
    labels_all = list(set(labels))
    n_labels = len(labels_all)
    n_samples = len(labels)

    # Calculate number of samples in each cluster
    n_samples_cluster = [np.sum(labels==li) for li in labels_all]

    # Check that no cluster is too small.
    # Clusters are assumed to be roughly the same size. Any cluster smaller 
    # than 10 times less than the expected amount will be removed
    min_n_samples = float(n_samples)/n_labels/10.0
    labels_all_checked = []
    for i, li in enumerate(labels_all):
        if n_samples_cluster[i] < min_n_samples:
            labels[labels==li] = -1
        else:
            labels_all_checked.append(li)
    labels_all = labels_all_checked

    # Change the cluster numbers to a contiguous positive sequence
    labels_checked = -1*np.ones(len(labels))
    cn = 0
    for li in labels_all:
        labels_checked[labels==li] = cn
        cn = cn + 1
    labels = labels_checked

    assert(np.any(labels==-1) == False)

    return labels

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

def _fit_mef_standard_curve(peaks_ch, peaks_mef):
    '''Fit a model mapping calibration bead fluroescence in channel space units 
    to their known MEF values.

    We first fit a beads fluroescence model using the peaks_ch and peaks_mef 
    arguments. We have determined from first principles that the appropriate 
    model for bead fluorescence is as follows:

        m*fl_ch_i + b = log(fl_mef_auto + fl_mef_i)

    where fl_ch_i is the fluorescence of peak i in channel space, and fl_mef_i
    is the fluorescence in mef values. The model includes 3 parameters: m, b, 
    and fl_mef_auto.

    This model is fit in a log-mef space using nonlinear least squares
    regression (as opposed to fitting an exponential model in y space). 
    Fitting in the log-mef space weights the residuals more evenly, whereas 
    fitting an exponential would vastly overvalue the brighter peaks.

    After fitting the beads model, this function returns a standard curve 
    function mapping channel space flurescence to MEF values, as follows:

        fl_mef = exp(m*fl_ch + b)

    Note that this is identical to the beads model after solving for fl_mef_i, 
    except that we are setting fl_mef_auto to zero. This is made so that the
    standard curve function returns absolute mef values.

    arguments:
    peaks_ch   - numpy array with fluorescence values of bead peaks in channel 
                 space.
    peaks_mef  - numpy array with fluorescence values of bead peaks in MEF.

    returns:
    sc         - standard curve function from channel space fluorescence to MEF
    sc_beads   - standard curve function from channel space fluorescence to MEF,
                considering the autofluorescence of the beads.
    sc_params  - array with fitted parameters of the beads model: 
                [m, b, fl_mef_auto].
    '''

    # Check that the input data has consistent dimensions
    assert len(peaks_ch) == len(peaks_mef), "peaks_ch and  \
        peaks_mef have different lengths"
    # Check that we have at least three points
    assert len(peaks_ch) > 2, "Standard curve model requires at least three\
        bead peak values."
        
    # Initialize parameters
    params = np.zeros(3)
    # Initial guesses:
    # 0: slope found by putting a line through the highest two peaks.
    # 1: y-intercept found by putting a line through highest two peaks.
    # 2: bead autofluorescence initialized to 100.
    params[0] = (np.log(peaks_mef[-1]) - np.log(peaks_mef[-2])) / \
                    (peaks_ch[-1] - peaks_ch[-2])
    params[1] = np.log(peaks_mef[-1]) - params[0] * peaks_ch[-1]
    params[2] = 100.

    # Error function
    def err_fun(p, x, y):
        return np.sum((np.log(y + p[2]) - ( p[0] * x + p[1] ))**2)
        
    # Bead model function
    def fit_fun(p,x):
        return np.exp(p[0] * x + p[1]) - p[2]

    # Channel-to-MEF standard curve transformation function
    def sc_fun(p,x):
        return np.exp(p[0] * x + p[1])
    
    # Fit parameters
    err_par = lambda p: err_fun(p, peaks_ch, peaks_mef)
    res = minimize(err_par, params)

    # Separate parameters
    sc_params = res.x

    # Beads model function
    sc_beads = lambda x: fit_fun(sc_params, x)

    # Standard curve function
    sc = lambda x: sc_fun(sc_params, x)
    
    return (sc, sc_beads, sc_params)

def get_mef_standard_curve(beads_data, peaks_mef, mef_channels = 0,
    cluster_method = 'dbscan', cluster_params = {}, cluster_channels = 0, 
    min_fl = 0, max_fl = 1023, verbose = False):
    '''Generate a function that transforms channel data into MEF data.

    This is performed using flow cytometry beads data, contained in the 
    beads_data argument. The steps involved in the MEF standard curve 
    generation are:
        1. The individual groups of beads are first clustered using a method
            of choice. 
        2. The value of the peak is identified for each cluster, for each
            channel in mef_channels.
        3. Clusters that are too close to one of the edges are discarded. The 
            corresponding known MEF values in peaks_mef are also discarded. If
            the expected mef value of some peak is unknown (represented as a 
            None value in peaks_mef), the corresponding peak is also discarded.
        4. The peaks identified from the beads are contrasted with the expected
            MEF values, and a standard curve function is generated using the
            appropriate MEF model. 

    The function generated is a transformation function, as specified in the 
    header of this module.

    Arguments:
    
    beads_data       - an NxD numpy array or FCSData object.
    peaks_mef        - a numpy array with the P known MEF values of the beads.
                        If mef_channels is an iterable of lenght C, peaks mef
                        should be a CxP array, where P is the number of MEF
                        peaks.
    mef_channels     - channel name, or iterable with channel names, on which
                        to generate MEF transformation functions.
    cluster_method   - method used for peak clustering.
    cluster_params   - parameters to pass to the clustering method.
    cluster_channels - channels used for clustering.
    min_fl           - minimum possible fluorescence value in data.
    max_fl           - maximum possible fluorescence value in data.
    verbose          - whether to print information about step completion,
                        warnings and errors.

    Returns: 

    Named tuple with fields:
    sc            - A function encoding the standard curve transformation.
    labels        - Labels resulting from the clustering procedure.
    peaks         - Identified peaks for every cluster.
    hists_smooth  - Smoothed histograms for every cluster and channel.
    peaks_ret     - Peaks per channel retained for fitting the beads model.
    mef_ret       - MEF values per channel retained for fitting the beads model.
    sc_beads      - Fitted model for beads (includes bead autoflurescence)
    sc_params     - Fitted parameters from the bead model.
    '''

    # 1. Slice beads_data and cluster
    data_cluster = beads_data[:,cluster_channels]
    if cluster_method == 'dbscan':
        labels = _clustering_dbscan(data_cluster, **cluster_params)
    else:
        raise ValueError("Clustering method {} not recognized."
            .format(cluster_method))
    labels_all = np.array(list(set(labels)))
    n_clusters = len(labels_all)
    if verbose:
        print "Number of clusters found: {}".format(n_clusters)

    # mef_channels and peaks_mef should be iterables.
    if hasattr(mef_channels, '__iter__'):
        mef_channel_all = list(mef_channels)
        peaks_mef_all = np.array(peaks_mef).copy()
    else:
        mef_channel_all = [mef_channels]
        peaks_mef_all = np.array([peaks_mef])

    # Initialize output lists
    peaks_all = []
    hists_smooth_all = []
    peaks_fit_all = []
    peaks_mef_channel_all = []
    sc_all = []
    sc_beads_all = []
    sc_params_all = []

    # Iterate through each mef channel
    for mef_channel, peaks_mef_channel in zip(mef_channel_all, peaks_mef_all):
        if verbose: 
            print "For channel {}...".format(mef_channel)
        # Separate data for the relevant channel
        data_channel = beads_data[:,mef_channel]

        # Step 2. Find peaks in each one of the clusters. 

        # Find peaks on all the channel data
        peaks, hists_smooth = _find_hist_peaks(data_channel, 
                                labels, labels_all = labels_all, 
                                min_val = min_fl, max_val = max_fl)
        
        # Save peaks and smoothed histograms
        peaks_all.append(peaks)
        hists_smooth_all.append(hists_smooth)

        # 3. Discard clusters that are too close to the edges
        # "Close" will be defined as peak being at a lower distance to either 
        # edge than 2.5x the standard deviation. 
        # Only one of two things could happen: either the peaks are being cut 
        # off to the left or to the right. That means that we can discard 
        # the lowest peaks or the highest peaks, but not both.
        
        # We first need to sort the peaks and clusters
        ind_sorted = np.argsort(peaks)
        peaks_sorted = peaks[ind_sorted]
        labels_sorted = labels_all[ind_sorted]
        # Get the standard deviation of each peak
        peaks_std = [np.std(data_channel[labels==li]) \
            for li in labels_sorted]
        if verbose:
            print "Peaks identified:"
            s_peaks = ''
            for p,s in zip(peaks_sorted, peaks_std):
                s_peaks = s_peaks + "{} +/- {:.2f}, ".format(p,s)
            print s_peaks
        # Decide which peaks to discard
        if (peaks_sorted[0] - 2.5*peaks_std[0]) <= min_fl \
            and (peaks_sorted[-1] + 2.5*peaks_std[-1]) >= max_fl:
            raise ValueError('Peaks are being cut off at both sides for \
                channel {}.'.format(mef_channel))
        elif (peaks_sorted[0] - 2.5*peaks_std[0]) <= min_fl:
            discard = 'left'
            discard_n = 1
            while (peaks_sorted[discard_n] - 2.5*peaks_std[discard_n]) \
                <= min_fl:
                discard_n = discard_n + 1
            peaks_fit = peaks_sorted[discard_n:]
            if verbose:
                print "{} peak(s) discarded to the left.".format(discard_n)
        elif (peaks_sorted[-1] + 2.5*peaks_std[-1]) >= max_fl:
            discard = 'right'
            discard_n = 1
            while (peaks_sorted[-1-discard_n] + 2.5*peaks_std[-1-discard_n]) \
                >= max_fl:
                discard_n = discard_n + 1
            peaks_fit = peaks_sorted[:-discard_n]
            if verbose:
                print "{} peak(s) discarded to the right.".format(discard_n)
        else:
            discard = False
            discard_n = 0
            peaks_fit = peaks_sorted
            if verbose:
                print "No peaks discarded."

        # Discard peaks from the MEF array
        peaks_mef_channel = peaks_mef_channel[:]
        if verbose:
            "{} MEF peaks provided.".format(len(peaks_mef_channel))
        mef_discard = len(peaks_mef_channel) - len(peaks_fit)
        if discard == 'left':
            peaks_mef_channel = peaks_mef_channel[mef_discard:]
            if verbose:
                print "{} MEF value(s) discarded to the left.".format(
                    mef_discard)
        elif discard == 'right':
            peaks_mef_channel = peaks_mef_channel[:-mef_discard]
            if verbose:
                print "{} MEF value(s) discarded to the right.".format(
                    mef_discard)
        elif mef_discard > 0:
            ValueError('Number of MEF values and peaks does not match in' + 
                ' channel {}.'.format(mef_channel))
        # Check if first mef peak is None, and discard
        if peaks_mef_channel[0] is None:
            peaks_mef_channel = peaks_mef_channel[1:]
            peaks_fit = peaks_fit[1:]
            if verbose:
                print("First peak's MEF value is unknown, and therefore was" +
                    " discarded.")
        # Cast to to float. If the array had contained a None value, it will be
        # of type 'object'.
        peaks_mef_channel = peaks_mef_channel.astype(np.float64)
        if verbose:
            print "MEF values for channel {}.".format(mef_channel)
            print peaks_mef_channel

        # Store results
        peaks_fit_all.append(peaks_fit)
        peaks_mef_channel_all.append(peaks_mef_channel)

        # 4. Get standard curve
        sc, sc_beads, sc_params = _fit_mef_standard_curve(peaks_fit, 
            peaks_mef_channel)

        sc_all.append(sc)
        sc_beads_all.append(sc_beads)
        sc_params_all.append(sc_params)

    # Assemble output as namedtuple
    Output = namedtuple('Output', ['sc', 'labels', 'peaks', 'hists_smooth', 
        'peaks_ret', 'mef_ret', 'sc_beads', 'sc_params'],
        verbose = False)

    # Unpack arrays if mef_channels was not iterable
    if hasattr(mef_channels, '__iter__'):
        res = Output(sc_all,
                labels, 
                peaks_all, 
                hists_smooth_all, 
                peaks_fit_all,
                peaks_mef_channel_all,
                sc_beads_all,
                sc_params_all)
    else:
        res = Output(sc_all[0],
            labels, 
            peaks_all[0], 
            hists_smooth_all[0], 
            peaks_fit_all[0], 
            peaks_mef_channel_all[0],
            sc_beads_all[0],
            sc_params_all[0])

    return res

def blank(data, white_cells):
    raise NotImplementedError

def compensate(data):
    raise NotImplementedError
