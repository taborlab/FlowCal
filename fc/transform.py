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
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 6/30/2015
#
# Requires:
#   * numpy
#   * scipy
#   * scikit-learn

import numpy as np
import scipy.ndimage.filters
from sklearn.cluster import DBSCAN 
from scipy.optimize import minimize

def exponentiate(data):
    '''Exponentiate data using the following transformation:
        y = 10^(x/256)
    This transformation spaces out the data across 4 logs from 1 to 10,000.
    This is also the transformation typically applied by flow cytometer
    software.

    data    - NxD numpy array (row=event)

    returns - NxD numpy array'''
    
    return np.power(10, (data.astype(float)/256.0))

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

def _fit_mef_standards_curve(fit_peaks, known_peaks, model='log line+auto'):
    '''Fit a model mapping fit calibration bead peaks (in channels) to known
    calibration bead peaks (in MEF). Empirically, we have determined that the
    following model serves best for our purposes:

        log(Y + P(3)) = (P(1)*X) + P(2)

    where Y=known_peaks, X=fit_peaks, and P represents the 3 model parameters.

    This model was derived from a straightforward exponential model (fitting
    a line on a log-y plot; P(1) and P(2)) which was augmented with an
    autofluorescence offset term (P(3)) to account for the fact that, in
    practice, the lowest calibration bead peak (which is supposed to be a
    blank) has a nonzero fluorescence value.

    This model is fit in a log-y space using nonlinear least squares
    regression (as opposed to fitting an exponential model in y space).
    We believe fitting in the log-y space weights the residuals more
    intuitively (roughly evenly in the log-y space), whereas fitting an
    exponential model vastly overvalues the brighter peaks.

    fit_peaks   - list? numpy array?
    known_peaks - list? numpy array?

    returns     - function? array of model parameters?'''

    # Check that the data is input data is appropriate
    assert len(fit_peaks) == len(known_peaks), "fit_peaks and  \
        known_peaks have different lengths"
    
    if model == 'log line':
        raise NotImplementedError("Model '{}' not implemented.".format(model))
    elif model == 'exp':
        raise NotImplementedError("Model '{}' not implemented.".format(model))
    elif model == 'exp+auto':
        raise NotImplementedError("Model '{}' not implemented.".format(model))
    elif model == 'log line+auto':
        # This model requires at least three data points
        assert len(fit_peaks) > 2, "Error: log line+auto standards curve model\
        requires at least three bead peak values."
        
        # This model has three parameters stored in params:
        # 0: The slope of fit_peaks vs. log(known_peaks - bead_auto)
        # 1: The y-intercept if fit_peaks vs. log(known_peaks - bead_auto)
        # 2: The bead autoflurescence (bead_auto).
        params = np.zeros(3)
        
        # Initial guesses to assist the fit:
        # 0: slope found by putting a line through the highest two peaks
        # 1: y-intercept found by putting a line through highest two peaks
        # 2: bead_auto found by a MEFL value of 100
        params[0] = (np.log(known_peaks[-1]) - np.log(known_peaks[-2])) / \
                        (fit_peaks[-1] - fit_peaks[-2])
        params[1] = np.log(known_peaks[-1]) - params[0] * fit_peaks[-1]
        params[2] = 100

        # Define error function
        def err_fun(p, x, y):
            return np.sum((np.log(y + p[2]) - ( p[0] * x + p[1] ))**2)
            
        # Define transformation function
        def fun(p,x):
            return np.exp(p[0] * x + p[1]) - p[2]
            
    elif model == 'log prop+auto':
        # This model requires at least two data points
        assert len(fit_peaks) > 1, "Error: log line+auto standards curve model\
        requires at least three bead peak values."        
        
        # This model has three parameters stored in params:
        # 0: The slope of fit_peaks vs. log(known_peaks - bead_auto)
        # 1: The bead autoflurescence (bead_auto).
        params = np.zeros(2)
        
        # Initial guesses to assist the fit:
        # 0: slope found by putting a line through the highest two peaks
        # 1: bead_auto found by a MEFL value of 100
        params[0] = (np.log(known_peaks[-1]) - np.log(known_peaks[-2])) / \
                        (fit_peaks[-1] - fit_peaks[-2])
        params[1] = 100
        
        # Define error function
        def err_fun(p, x, y):
            return np.sum((np.log(y + p[1]) - ( p[0] * x ))**2)
        
        # Definte transformation function
        def fun(p,x):
            return np.exp(p[0] * x) - p[1]
    else: 
        raise ValueError('Unrecognized model: {}'.format(model))
    
    # Perform fit using scipy.optimize.minimize to minimize error function
    err_par = lambda p: err_fun(p, fit_peaks, known_peaks)
    res = minimize(err_par, params)
    
    # Store best-fit values
    params = res.x

    # Define standards curve function
    sc = lambda x: fun(params, x)
    
    return sc

def get_mef_standards_curve(beads_data, peaks_mef, mef_channels = 0,
    cluster_method = 'dbscan', cluster_params = {}, cluster_channels = 0, 
    mef_model = 'log line+auto', min_fl = 0, max_fl = 1023, verbose = False):
    '''Generate a function that transforms channel data into MEF data.

    This is performed using flow cytometry beads data, contained in the 
    beads_data argument. The steps involved in the MEF standards curve 
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
            MEF values, and a standards curve function is generated using the
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
    mef_model        - Bead fluorescence model
    cluster_method   - method used for peak clustering.
    cluster_params   - parameters to pass to the clustering method.
    cluster_channels - channels used for clustering.
    min_fl           - minimum possible fluorescence value in data.
    max_fl           - maximum possible fluorescence value in data.
    verbose          - whether to print information about step completion,
                        warnings and errors.
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
    sc_all = []

    # Iterate through each mef channel
    for mef_channel, peaks_mef_channel in zip(mef_channel_all, peaks_mef_all):
        if verbose: 
            print "For channel {}...".format(mef_channel)
        # Separate data for the relevant channel
        data_channel = beads_data[:,mef_channel]

        # Step 2. Find peaks in each one of the clusters. 

        # # Remove events in the boundaries first.
        # mask_channel = (data_channel > min_fl) & (data_channel < max_fl)
        # data_masked_channel = data_channel[mask_channel]
        # labels_channel = labels[mask_channel]
        # labels_all_channel = np.array(list(set(labels_channel)))
        # # Some peaks can be lost if they are completely contained at either 
        # # boundary. In this case, "peaks" and "labels_all" will have 
        # # inconsistent dimensions.
        # # TODO: handle this case.
        # if len(labels_all_channel) != len(labels_all):
        #     raise RuntimeError("Completely saturated peaks.")
        # # Find peaks
        # peaks, hists_smooth = _find_hist_peaks(data_masked_channel, 
        #                         labels_channel, labels_all = labels_all_channel, 
        #                         min_val = min_fl, max_val = max_fl)

        # Find peaks on all the channel data
        peaks, hists_smooth = _find_hist_peaks(data_channel, 
                                labels, labels_all = labels_all, 
                                min_val = min_fl, max_val = max_fl)
        
        # Save peaks and smoothed histograms
        peaks_all.append(peaks)
        hists_smooth_all.append(hists_smooth)

        # 3. Discard clusters that are too close to the edges
        # "Close" will be defined as peak being at a lower distance to the left 
        # edge than 2.5x the standard deviation, and lower than 2.5x the std to 
        # the right. 
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

        # Discard equivalent peaks form the MEF array
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

        # Get standards curve
        sc = _fit_mef_standards_curve(peaks_fit, peaks_mef_channel, 
            model = mef_model)
        sc_all.append(sc)

    # Unpack arrays if mef_channels was not iterable
    if hasattr(mef_channels, '__iter__'):
        sc_ret = sc_all
        peaks_ret = peaks_all
        hists_smooth_ret = hists_smooth_all
    else:
        sc_ret = sc_all[0]
        peaks_ret = peaks_all[0]
        hists_smooth_ret = hists_smooth_all[0]
    return sc_ret, labels, peaks_ret, hists_smooth_ret, peaks_fit, peaks_mef_channel

def blank(data, white_cells):
    raise NotImplementedError

def compensate(data):
    raise NotImplementedError
