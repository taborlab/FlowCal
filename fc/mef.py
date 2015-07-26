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
import collections

import numpy
from scipy.optimize import minimize
import scipy.ndimage.filters
from matplotlib import pyplot
from sklearn.cluster import DBSCAN 
from sklearn.mixture import GMM 

import fc.plot
import fc.transform

def clustering_dbscan(data, eps = 20.0, min_samples = None, n_clusters_exp = 8):
    '''
    Find clusters in the data array using the DBSCAN method from the 
    scikit-learn library.

    data           - NxD numpy array.
    eps            - Parameter for DBSCAN. Check scikit-learn documentation for
                     more info.
    min_samples    - Parameter for DBSCAN. Check scikit-learn documentation for
                     more info.
    n_clusters_exp - Number of expected clusters

    returns     - Nx1 numpy array, labeling each sample to a cluster.
    '''
    # Default value of min_samples
    if min_samples is None:
        min_samples = data.shape[0]/200.

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
    n_samples_cluster = [numpy.sum(labels==li) for li in labels_all]

    # Check that no cluster is too small.
    # Clusters are assumed to be uniformly distributed. Any cluster 10 std 
    # smaller than the expected size (under a binomial distribution) will be 
    # assimilated with the next smallest
    # Larger than expected clusters will be assumed to correspond to clusters
    # containing data for 2 or more bead types.
    p = 1./n_clusters_exp
    n_samples_exp = data.shape[0]*p
    n_samples_std = numpy.sqrt(data.shape[0]*p*(1-p))
    while(True):
        cluster_size = numpy.array([numpy.sum(labels==li) for li in labels_all])
        cluster_i = numpy.argsort(cluster_size)
        if cluster_size[cluster_i[0]] < n_samples_exp - n_samples_std*10:
            labels[labels==labels_all[cluster_i[0]]] = labels_all[cluster_i[1]]
            labels_all.remove(labels_all[cluster_i[0]])
        else:
            break

    # Change the cluster numbers to a contiguous positive sequence
    labels_checked = -1*numpy.ones(len(labels))
    cn = 0
    for li in labels_all:
        labels_checked[labels==li] = cn
        cn = cn + 1
    labels = labels_checked

    assert(numpy.any(labels==-1) == False)

    return labels

def clustering_distance(data, n_clusters = 8):
    '''
    Find clusters in the data array based on distance to the origin.

    data        - NxD numpy array.
    n_clusters  - Number of expected clusters

    returns     - Nx1 numpy array, labeling each sample to a cluster.
    '''
    # Number of elements per cluster
    fractions = numpy.ones(n_clusters)*1./n_clusters

    n_per_cluster = fractions*data.shape[0]
    cluster_cum = numpy.append([0], numpy.cumsum(n_per_cluster))

    # Get distance and sort based on it
    dist = numpy.sum(data**2., axis = 1)
    sorted_i = numpy.argsort(dist)

    # Initialize labels
    labels = numpy.ones(data.shape[0])*-1

    # Assign labels
    for i in range(n_clusters):
        il = int(cluster_cum[i])
        ih = int(cluster_cum[i+1])
        sorted_i_i = sorted_i[il:ih]
        labels[sorted_i_i] = i

    return labels


def clustering_gmm(data, n_clusters = 8, initialization = 'distance_sub', 
    tol = 1e-7, min_covar = 1e-2):
    '''
    Find clusters in the data array using the GMM method from the 
    scikit-learn library.

    data            - NxD numpy array.
    n_clusters      - Number of expected clusters
    initialization  - Initialization method
    tol             - Tolerance for convergence of GMM method. Check 
                        scikit-learn documentation for more info.
    min_covar       - Minimum covariance for the GMM method. Check 
                        scikit-learn documentation for more info.
    returns     - Nx1 numpy array, labeling each sample to a cluster.
    '''


    # Initialization method
    if initialization == 'distance':
        # Perform distance-based clustering
        labels = fc.mef.clustering_distance(data, n_clusters)
        labels_all = list(set(labels))
        data_clustered = [data[labels == i] for i in labels_all]
        
        # Initialize parameters for GMM
        weights = numpy.tile(1.0 / n_clusters,
                                        n_clusters)
        means = numpy.array([numpy.mean(di, axis = 0) 
            for di in data_clustered])
        
        if data.shape[1] == 1:
            covars = [numpy.cov(di.T).reshape(1,1) for di in data_clustered]
        else:
            covars = [numpy.cov(di.T) for di in data_clustered]

        # Initialize GMM object
        gmm = GMM(n_components = n_clusters, tol = tol, min_covar = min_covar,
            covariance_type = 'full', params = 'mc', init_params = '')
        gmm.weight_ = weights
        gmm.means_ = means
        gmm.covars_ = covars

    elif initialization == 'distance_sub':
        # Initialize parameters for GMM
        weights = numpy.tile(1.0 / n_clusters,
                                        n_clusters)
        means = []
        covars = []

        # Get distance and sort based on it
        dist = numpy.sum(data**2., axis = 1)
        sorted_i = numpy.argsort(dist)

        # Expected number of elements per cluster
        n_per_cluster = data.shape[0]/float(n_clusters)

        # Get the means and covariances per cluster
        # We will just use a fraction of 1-2*discard_frac of the data.
        # Data at the edges that actually corresponds to another cluster can
        # really mess up the final result.
        discard_frac = 0.25
        for i in range(n_clusters):
            il = int((i+discard_frac)*n_per_cluster)
            ih = int((i+1-discard_frac)*n_per_cluster)
            sorted_i_i = sorted_i[il:ih]
            data_i = data[sorted_i_i]
            means.append(numpy.mean(data_i, axis = 0))
            if data.shape[1] == 1:
                covars.append(numpy.cov(data_i.T).reshape(1,1))
            else:
                covars.append(numpy.cov(data_i.T))
        means = numpy.array(means)

        # Initialize GMM object
        gmm = GMM(n_components = n_clusters, tol = tol, min_covar = min_covar,
            covariance_type = 'full', params = 'mc', init_params = '')
        gmm.weight_ = weights
        gmm.means_ = means
        gmm.covars_ = covars

    else:
        raise ValueError('Initialization method {} not implemented.'\
            .format(initialization))

    # Fit 
    gmm.fit(data)
    # Get labels using the responsibilities
    # This avoids the complete elimination of a cluster if two or more clusters
    # have very similar means.
    resp = gmm.predict_proba(data)
    labels = [numpy.random.choice(range(n_clusters), p = ri) for ri in resp]

    return labels

def find_peaks_smoothed_mode(data, min_val = 0, max_val = 1023):
    '''
    Find histogram peaks using the smoothed mode method.

    The algorithm then proceeds as follows:
        1. Calculate the histogram.
        2. Use a 1D Gaussian filter to smooth out the histogram. The sigma 
            value for the Gaussian filter is chosen based on the standard 
            deviation of the fit Gaussian from the GMM for that paritcular 
            segment.
        3. Identify peak as the maximum value of the Gaussian-blurred histogram.

    data        - Nx1 numpy array with the 1D data from where peaks should be 
                  identified. Data values are assumed to be integer.
    min_val     - minimum possible value in data.
    max_val     - maximum possible value in data.

    returns     - The value of the identified peak.
                - (max_val - min_val + 1) numpy array with a smoothed 
                  histogram for each cluster.
    '''

    # Calculate bin edges and centers
    bin_edges = numpy.arange(min_val, max_val + 2) - 0.5
    bin_edges[0] = -numpy.inf
    bin_edges[-1] = numpy.inf
    bin_centers = numpy.arange(min_val, max_val + 1)

    # Identify peak
    # Calculate sample mean and standard deviation
    # mu = numpy.mean(data)
    sigma = numpy.std(data)
    # Calculate histogram
    hist, __ = numpy.histogram(data, bin_edges)
    # Do Gaussian blur on histogram
    # We have found empirically that using one half of the distribution's 
    # standard deviation results in a nice fit.
    hist_smooth = scipy.ndimage.filters.gaussian_filter1d(hist, sigma/2.)
    # Extract peak
    i_max = numpy.argmax(hist_smooth)
    peak = bin_centers[i_max]

    return peak, hist_smooth

def find_peaks_median(data):
    '''
    Find histogram peaks as the median.

    data        - Nx1 numpy array with the 1D data from where peaks should be 
                  identified. 

    returns     - The median of data.
    '''

    peak = numpy.median(data)

    return peak

def select_peaks(peaks_ch, 
                peaks_mef, 
                peaks_ch_std,
                peaks_ch_std_mult_l = 2.5,
                peaks_ch_std_mult_r = 2.5,
                peaks_ch_min = 0, 
                peaks_ch_max = 1023):
    '''Select peaks for fitting based on proximity to the minimum and maximum 
    values.

    This function discards some peaks on channel space from peaks_ch if they're
    too close to either peaks_ch_min or peaks_ch_max. Next, it discards the
    corresponding peaks in peaks_mef. Finally, it discards peaks from peaks_mef
    that have an undetermined value (NaN), and it also discards the 
    corresponding peaks in peaks_ch.

    Arguments:
    peaks_ch            - Sorted peak values in channel space 
    peaks_mef           - Peak values in MEF units
    peaks_ch_std_mult_l - Tolerance for peaks at the left, in std. devs.
    peaks_ch_std_mult_r - Tolerance for peaks at the right, in std. devs.
    peaks_ch_min        - Minimum tolerable value in channel space
    peaks_ch_max        - Maximum tolerable value in channel space
    '''

    # Minimum peak standard deviation will be 1.0
    min_std = 1.0
    peaks_ch_std = peaks_ch_std.copy()
    peaks_ch_std[peaks_ch_std < min_std] = min_std

    # Discard channel-space peaks
    if (peaks_ch[0] - peaks_ch_std[0]*peaks_ch_std_mult_l) <= peaks_ch_min and\
        (peaks_ch[-1] + peaks_ch_std[-1]*peaks_ch_std_mult_r) >= peaks_ch_max:
        raise ValueError('Peaks are being cut off at both sides.')
    elif (peaks_ch[0] - peaks_ch_std[0]*peaks_ch_std_mult_l) <= peaks_ch_min:
        discard_ch = 'left'
        discard_ch_n = 1
        while (peaks_ch[discard_ch_n] - 
            peaks_ch_std[discard_ch_n]*peaks_ch_std_mult_l) <= peaks_ch_min:
            discard_ch_n = discard_ch_n + 1
        sel_peaks_ch = peaks_ch[discard_ch_n:]
    elif (peaks_ch[-1] + peaks_ch_std[-1]*peaks_ch_std_mult_r) >= peaks_ch_max:
        discard_ch = 'right'
        discard_ch_n = 1
        while (peaks_ch[-1-discard_ch_n] +
            peaks_ch_std[-1-discard_ch_n]*peaks_ch_std_mult_r) >= peaks_ch_max:
            discard_ch_n = discard_ch_n + 1
        sel_peaks_ch = peaks_ch[:-discard_ch_n]
    else:
        discard_ch = False
        discard_ch_n = 0
        sel_peaks_ch = peaks_ch.copy()

    # Discard MEF peaks
    discard_mef_n = len(peaks_mef) - len(sel_peaks_ch)
    if discard_ch == 'left':
        sel_peaks_mef = peaks_mef[discard_mef_n:]
    elif discard_ch == 'right':
        sel_peaks_mef = peaks_mef[:-discard_mef_n]
    elif discard_ch == False and discard_mef_n == 0:
        sel_peaks_mef = peaks_mef.copy()
    else:
        ValueError('Number of MEF values and channel peaks does not match.')
    
    # Discard unknown (NaN) peaks
    unknown_mef = numpy.isnan(sel_peaks_mef)
    n_unknown_mef = numpy.sum(unknown_mef)
    if n_unknown_mef > 0:
        sel_peaks_ch = sel_peaks_ch[numpy.invert(unknown_mef)]
        sel_peaks_mef = sel_peaks_mef[numpy.invert(unknown_mef)]

    return sel_peaks_ch, sel_peaks_mef

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
    params = numpy.zeros(3)
    # Initial guesses:
    # 0: slope found by putting a line through the highest two peaks.
    # 1: y-intercept found by putting a line through highest two peaks.
    # 2: bead autofluorescence initialized to 100.
    params[0] = (numpy.log(peaks_mef[-1]) - numpy.log(peaks_mef[-2])) / \
                    (peaks_ch[-1] - peaks_ch[-2])
    params[1] = numpy.log(peaks_mef[-1]) - params[0] * peaks_ch[-1]
    params[2] = 100.

    # Error function
    def err_fun(p, x, y):
        return numpy.sum((numpy.log(y + p[2]) - ( p[0] * x + p[1] ))**2)
        
    # Bead model function
    def fit_fun(p,x):
        return numpy.exp(p[0] * x + p[1]) - p[2]

    # Channel-to-MEF standard curve transformation function
    def sc_fun(p,x):
        return numpy.exp(p[0] * x + p[1])
    
    # Fit parameters
    err_par = lambda p: err_fun(p, peaks_ch, peaks_mef)
    res = minimize(err_par, params, options = {'gtol': 1e-6})

    # Separate parameters
    sc_params = res.x

    # Beads model function
    sc_beads = lambda x: fit_fun(sc_params, x)

    # Standard curve function
    sc = lambda x: sc_fun(sc_params, x)
    
    return (sc, sc_beads, sc_params)

def get_transform_fxn(data_beads, peaks_mef, mef_channels,
    cluster_method = 'gmm', cluster_params = {}, cluster_channels = 0, 
    find_peaks_method = 'median',
    verbose = False, plot = False, plot_dir = None, full = False):
    '''Generate a function that transforms channel data into MEF data.

    This is performed using flow cytometry beads data, contained in the 
    data_beads argument. The steps involved in the MEF standard curve 
    generation are:
        1. The individual groups of beads are first clustered using a method
            of choice. 
        2. The value of the peak is identified for each cluster, for each
            channel in mef_channels.
        3. Clusters that are too close to one of the edges are discarded. The 
            corresponding known MEF values in peaks_mef are also discarded. If
            the expected mef value of some peak is unknown (represented as a 
            NaN value in peaks_mef), the corresponding peak is also discarded.
        4. The peaks identified from the beads are contrasted with the expected
            MEF values, and a standard curve function is generated using the
            appropriate MEF model. 

    This function can print information about partial results if verbose is
    True, and generate plots after each step if plot is True.

    The function generated is a transformation function, as specified in the 
    header of the transform module.

    Arguments:
    
    data_beads        - an NxD numpy array or FCSData object.
    peaks_mef         - a numpy array with the P known MEF values of the beads.
                         If mef_channels is an iterable of lenght C, peaks mef
                         should be a CxP array, where P is the number of MEF
                         peaks.
    mef_channels      - channel name, or iterable with channel names, on which
                         to generate MEF transformation functions.
    cluster_method    - method used for peak clustering.
    cluster_params    - parameters to pass to the clustering method.
    cluster_channels  - channels used for clustering.
    find_peaks_method - Method used to find the peak value.
    verbose           - whether to print information about step completion,
                         warnings and errors.
    plot              - If True, produce diagnostic plots.
    plot_dir          - Directory where to save diagnostics plots. Ignored if 
                         plot is False.
    full              - Whether to include intermediate results in the output.
                         If full is True, the function returns a named tuple
                         with fields as described below. If full is False, the
                         function only returns the calculated transformation
                         function.

    Returns: 

    transform_fxn   - A transformation function encoding the standard curves.
    clustering_res  - If full == True, this is a dictionary that contains the 
                        results of the clustering step.
    peak_find_res   - If full == True, this is a dictionary that contains the 
                        results of the peak finding step.
    peak_sel_res    - If full == True, this is a dictionary that contains the 
                        results of the peak selection step.
    fitting_res     - If full == True, this is a dictionary that contains the 
                        results of the model fitting step.

    '''
    if verbose:
        prev_precision = numpy.get_printoptions()['precision']
        numpy.set_printoptions(precision=2)
    # Create directory if plot is True
    if plot:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    # Extract beads file name
    data_file_name = str(data_beads)

    # 1. Cluster
    # ===========
    if cluster_method == 'dbscan':
        labels = clustering_dbscan(data_beads[:,cluster_channels], 
            **cluster_params)
    elif cluster_method == 'distance':
        labels = clustering_distance(data_beads[:,cluster_channels], 
            **cluster_params)
    elif cluster_method == 'gmm':
        labels = clustering_gmm(data_beads[:,cluster_channels], 
            **cluster_params)
    else:
        raise ValueError("Clustering method {} not recognized."
            .format(cluster_method))

    labels_all = numpy.array(list(set(labels)))
    n_clusters = len(labels_all)
    data_clustered = [data_beads[labels == i] for i in labels_all]

    # Print information
    if verbose:
        print "- STEP 1. CLUSTERING."
        print "Number of clusters found: {}".format(n_clusters)
        # Calculate percentage of each cluster
        data_count = numpy.array([di.shape[0] for di in data_clustered])
        data_perc = data_count*100.0/data_count.sum()
        print "Percentage of samples in each cluster:"
        print data_perc
    # Plot
    if plot:
        # Sort
        cluster_dist = [numpy.sum((numpy.mean(di[:,cluster_channels], 
                axis = 0))**2) for di in data_clustered]
        cluster_sorted_ind = numpy.argsort(cluster_dist)
        data_plot = [data_clustered[i] for i in cluster_sorted_ind]
            
        if len(cluster_channels) == 2:
            # Plot
            pyplot.figure(figsize = (6,4))
            fc.plot.scatter2d(data_plot, 
                    channels = cluster_channels, 
                    savefig = '{}/cluster_{}.png'.format(plot_dir,
                                                    data_file_name))
            pyplot.close()
            
        if len(cluster_channels) == 3:
            # Plot
            pyplot.figure(figsize = (8,6))
            fc.plot.scatter3d(data_plot, 
                    channels = cluster_channels, 
                    savefig = '{}/cluster_{}.png'.format(plot_dir,
                                                    data_file_name))
            pyplot.close()

    # mef_channels and peaks_mef should be iterables.
    if hasattr(mef_channels, '__iter__'):
        mef_channel_all = list(mef_channels)
        peaks_mef_all = numpy.array(peaks_mef).copy()
    else:
        mef_channel_all = [mef_channels]
        peaks_mef_all = numpy.array([peaks_mef])

    # Initialize lists to acumulate results
    sc_all = []
    if full:
        peaks_ch_all = []
        if find_peaks_method == 'smoothed_mode':
            peaks_hists_all = []
        sel_peaks_ch_all = []
        sel_peaks_mef_all = []
        sc_beads_all = []
        sc_params_all =[]

    # Iterate through each mef channel
    for mef_channel, peaks_mef_channel in zip(mef_channel_all, peaks_mef_all):

        # Slice relevant channel's data
        data_channel = data_beads[:,mef_channel]
        # Print information
        if verbose: 
            print "- MEF transformation for channel {}...".format(mef_channel)

        # Step 2. Find peaks in each one of the clusters. 
        # ===============================================

        # Find peaks on all the channel data
        min_fl = data_channel.channel_info[0]['range'][0]
        max_fl = data_channel.channel_info[0]['range'][1]
        if find_peaks_method == 'smoothed_mode':
            peaks_hists = [find_peaks_smoothed_mode(di[:,mef_channel], 
                                min_val = min_fl, max_val = max_fl)
                                for di in data_clustered]
            peaks_ch = numpy.array([ph[0] for ph in peaks_hists])
            hists_smooth = [ph[1] for ph in peaks_hists]
        elif find_peaks_method == 'median':
            peaks_ch = numpy.array([find_peaks_median(di[:,mef_channel]) 
                                for di in data_clustered])
        # Sort peaks and clusters
        ind_sorted = numpy.argsort(peaks_ch)
        peaks_sorted = peaks_ch[ind_sorted]
        data_sorted = [data_clustered[i] for i in ind_sorted]

        # Accumulate results
        if full:
            peaks_ch_all.append(peaks_ch)
            if find_peaks_method == 'smoothed_mode':
                peaks_hists_all.append(hists_smooth)
        # Print information
        if verbose:
            print "- STEP 2. PEAK IDENTIFICATION."
            print "Channel peaks identified:"
            print peaks_sorted
        # Plot
        if plot:
            # Get colors for peaks
            colors = fc.plot.load_colormap('spectral', n_clusters)
            # Plot histograms
            pyplot.figure(figsize = (8,4))
            fc.plot.hist1d(data_plot, channel = mef_channel, div = 4, 
                alpha = 0.75)
            # Plot smoothed histograms and peaks
            for c, i in zip(colors, cluster_sorted_ind):
                # Smoothed histogram, if applicable
                if find_peaks_method == 'smoothed_mode':
                    h = hists_smooth[i]
                    pyplot.plot(numpy.linspace(min_fl, max_fl, len(h)), h*4, 
                        color = c)
                # Peak values
                p = peaks_ch[i]
                ylim = pyplot.ylim()
                pyplot.plot([p, p], [ylim[0], ylim[1]], color = c)
                pyplot.ylim(ylim)
            # Save and close
            pyplot.tight_layout()
            pyplot.savefig('{}/peaks_{}_{}.png'.format(plot_dir,
                                    mef_channel, data_file_name), dpi = 300)
            pyplot.close()

        # 3. Select peaks for fitting
        # ===========================
        
        # Get the standard deviation of each peak
        peaks_std = numpy.array([numpy.std(di[:,mef_channel]) \
            for di in data_sorted])
        
        # Print information
        if verbose:
            print "Standard deviations:"
            print peaks_std
            print "MEF peaks provided:"
            print peaks_mef_channel
            print "- STEP 3. PEAK SELECTION."

        # Select peaks
        sel_peaks_ch, sel_peaks_mef = select_peaks(peaks_sorted, 
                peaks_mef_channel, peaks_ch_std = peaks_std,
                peaks_ch_min = min_fl, peaks_ch_max = max_fl)

        # Accumulate results
        if full:
            sel_peaks_ch_all.append(sel_peaks_ch)
            sel_peaks_mef_all.append(sel_peaks_mef)
        # Print information
        if verbose:
            print "{} peaks retained.".format(len(sel_peaks_ch))
            print "Selected channel peaks:"
            print sel_peaks_ch
            print "Selected MEF peaks:"
            print sel_peaks_mef

        # 4. Get standard curve
        # ======================
        sc, sc_beads, sc_params = fit_standard_curve(sel_peaks_ch, 
            sel_peaks_mef)
        if verbose:
            print "- STEP 4. STANDARDS CURVE FITTING."
            print "Fitted parameters:"
            print sc_params

        sc_all.append(sc)
        if full:
            sc_beads_all.append(sc_beads)
            sc_params_all.append(sc_params)

        # Plot
        if plot:
            # Make label for x axis
            channel_name = data_channel.channel_info[0]['label']
            channel_gain = data_channel.channel_info[0]['pmt_voltage']
            xlabel = '{} (gain = {})'.format(channel_name, channel_gain)
            # Plot standard curve
            pyplot.figure(figsize = (6,4))
            fc.plot.mef_std_crv(sel_peaks_ch, 
                    sel_peaks_mef,
                    sc_beads,
                    sc,
                    xlabel = xlabel,
                    ylabel = 'MEF',
                    savefig = '{}/std_crv_{}_{}.png'.format(plot_dir,
                                                            mef_channel,
                                                            data_file_name, 
                                                            ))
            pyplot.close()

    # Make output transformation function
    transform_fxn = functools.partial(fc.transform.to_mef,
                                    sc_list = sc_all,
                                    sc_channels = mef_channel_all)

    if verbose:
        numpy.set_printoptions(precision = prev_precision)

    if full:
        # Clustering results
        clustering_res = {}
        clustering_res['labels'] = labels
        # Peak finding results
        peak_find_res = {}
        peak_find_res['peaks_ch'] = peaks_ch_all
        if find_peaks_method == 'smoothed_mode':
            peak_find_res['peaks_hists'] = peaks_hists_all
        # Peak selection results
        peak_sel_res = {}
        peak_sel_res['sel_peaks_ch'] = sel_peaks_ch_all
        peak_sel_res['sel_peaks_mef'] = sel_peaks_mef_all
        # Fitting results
        fitting_res = {}
        fitting_res['sc'] = sc_all
        fitting_res['sc_beads'] = sc_beads_all
        fitting_res['sc_params'] = sc_params_all

        # Make namedtuple
        fields = ['transform_fxn',
                  'clustering_res',
                  'peak_find_res',
                  'peak_sel_res',
                  'fitting_res']
        MEFOutput = collections.namedtuple('MEFOutput', fields, verbose=False)
        out = MEFOutput(transform_fxn = transform_fxn,
                        clustering_res = clustering_res,
                        peak_find_res = peak_find_res,
                        peak_sel_res = peak_sel_res,
                        fitting_res = fitting_res,
                        )
        return out
    else:
        return transform_fxn
