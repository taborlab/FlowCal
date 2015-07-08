#!/usr/bin/python
#
# transform.py - Module containing functions related to calibration beads
#                   analysis and standard curve determination.
#
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/7/2015
#
# Requires:
#   * numpy
#   * scipy
#   * scikit-learn
#   * fc.plot
#   * fc.transform

import os
import functools

import numpy as np
from scipy.optimize import minimize
import scipy.ndimage.filters
from matplotlib import pyplot
from sklearn.cluster import DBSCAN 

import fc.plot
import fc.transform

def clustering_dbscan(data, eps = 20.0, min_samples = 40):
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

def find_hist_peaks(data, labels, labels_all = None, 
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

def fit_standard_curve(peaks_ch, peaks_mef):
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

def get_transform_fxn(beads_data, peaks_mef, mef_channels = 0,
    cluster_method = 'dbscan', cluster_params = {}, cluster_channels = 0, 
    verbose = False, plot = False, plot_dir = None):
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
    header of the transform module.

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
    verbose          - whether to print information about step completion,
                        warnings and errors.
    plot             - If True, produce diagnostic plots.
    plot_dir         - Directory where to save plots.

    Returns: 

    transform_fxn - A transformation function encoding the standard curves.

    '''
    # Create directory if plot is True
    if plot:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    # 1. Slice beads_data and cluster
    data_cluster = beads_data[:,cluster_channels]
    if cluster_method == 'dbscan':
        labels = clustering_dbscan(data_cluster, **cluster_params)
    else:
        raise ValueError("Clustering method {} not recognized."
            .format(cluster_method))
    labels_all = np.array(list(set(labels)))
    n_clusters = len(labels_all)
    # Print information
    if verbose:
        print "Number of clusters found: {}".format(n_clusters)
    # Plot
    if plot:
        if len(cluster_channels) == 3:
            data_list = [beads_data[labels == i] for i in labels_all]
            # Sort
            data_dist = [np.sum((np.mean(di[:,cluster_channels], 
                    axis = 0))**2) for di in data_list]
            data_ind = np.argsort(data_dist)
            data_plot = [data_list[i] for i in data_ind]
            # Plot
            pyplot.figure(figsize = (8,6))
            fc.plot.scatter3d(data_plot, 
                    channels = cluster_channels, 
                    savefig = '{}/cluster.png'.format(plot_dir))

    # mef_channels and peaks_mef should be iterables.
    if hasattr(mef_channels, '__iter__'):
        mef_channel_all = list(mef_channels)
        peaks_mef_all = np.array(peaks_mef).copy()
    else:
        mef_channel_all = [mef_channels]
        peaks_mef_all = np.array([peaks_mef])

    # Initialize output list
    sc_all = []

    # Iterate through each mef channel
    for mef_channel, peaks_mef_channel in zip(mef_channel_all, peaks_mef_all):
        if verbose: 
            print "For channel {}...".format(mef_channel)
        # Separate data for the relevant channel
        data_channel = beads_data[:,mef_channel]

        # Step 2. Find peaks in each one of the clusters. 

        # Find peaks on all the channel data
        min_fl = data_channel.channel_info[0]['range'][0]
        max_fl = data_channel.channel_info[0]['range'][1]
        peaks, hists_smooth = find_hist_peaks(data_channel, 
                                labels, labels_all = labels_all, 
                                min_val = min_fl, max_val = max_fl)
        if plot:
            colors = fc.plot.load_colormap('spectral', n_clusters)
            pyplot.figure(figsize = (8,4))
            fc.plot.hist1d(data_plot, channel = mef_channel, div = 4, 
                alpha = 0.75)
            for c, i in zip(colors, data_ind):
                p = peaks[i]
                h = hists_smooth[i]
                pyplot.plot(np.linspace(min_fl, max_fl, len(h)), h*4, 
                    color = c)
                ylim = pyplot.ylim()
                pyplot.plot([p, p], [ylim[0], ylim[1]], color = c)
            pyplot.tight_layout()
            pyplot.savefig('{}/peaks_{}.png'.format(plot_dir, mef_channel),
                dpi = 300)
            pyplot.close()

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
            print peaks_sorted
            print "Standard deviation:"
            print ["{:.2f}".format(p) for p in peaks_std]
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

        # 4. Get standard curve
        sc, sc_beads, sc_params = fit_standard_curve(peaks_fit, 
            peaks_mef_channel)

        sc_all.append(sc)

        if plot:
            channel_name = data_channel.channel_info[0]['label']
            channel_gain = data_channel.channel_info[0]['pmt_voltage']
            xlabel = '{} (gain = {})'.format(channel_name, channel_gain)
            pyplot.figure(figsize = (6,4))
            fc.plot.mef_std_crv(peaks_fit, 
                    peaks_mef_channel,
                    sc_beads,
                    sc,
                    xlabel = xlabel,
                    ylabel = 'MEF',
                    savefig = '{}/std_crv_{}.png'.format(plot_dir, 
                            mef_channel))

    # Make output transformation function
    transform_fxn = functools.partial(fc.transform.to_mef,
                                    sc_list = sc_all,
                                    sc_channels = mef_channel_all)

    return transform_fxn
