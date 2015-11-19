"""
Functions for transforming flow cytometer data to MEF units.

"""

import os
import functools
import collections

import numpy as np
from scipy.optimize import minimize
import scipy.ndimage.filters
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN 
from sklearn.mixture import GMM 

import fc.plot
import fc.transform

def clustering_dbscan(data, eps = 20.0, min_samples = None, n_clusters_exp = 8):
    """
    Find clusters in an array using the DBSCAN method.

    Parameters
    ----------
    data : array_like
        NxD array to cluster.
    eps : float, optional
        Maximum distance between core samples.
    min_samples : int, optional
        Minimum number of neighbors for a sample to be considered core
        sample. If not specified, it defaults to the number of events in
        `data`, divided by 200. Passed directly to ``scikit-learn``'s
        DBSCAN.
    n_clusters_exp : int, optional
        Expected number of clusters. Passed directly to ``scikit-learn``'s
        DBSCAN.

    Returns
    -------
    labels : array
        Nx1 array with labels for each element in `data`, assigning
        ``data[i]`` to cluster ``labels[i]``.

    Notes
    -----
    The DBSCAN method views clusters as areas of high density of samples,
    separated by areas of low density. This algorithm works by defining
    'core samples' as samples that have `min_samples` neighbors separated
    by a distance of `eps` or less. To get a cluster, start with a core
    sample, find all its neighbors that are core samples, find the
    neighbors of these core samples that are also core samples, and so on.
    The cluster is then defined as this set of core samples, plus their
    neighbors that are not core samples.

    DBSCAN normally finds the number of clusters automatically. However,
    `clustering_dbscan` makes some post-processing that we have found
    improves results when clustering calibration beads. In particular,
    `clustering_dbscan` accepts a parameter `n_clusters_exp` which contains
    the expected number of clusters. The clusters are expected to be
    approximately the same size. If the size of a cluster is found to be 10
    standard deviations smaller than the average cluster size, it is
    automatically eliminated, and its samples are added to the 'outliers'
    cluster. Additionally, the 'outliers' cluster has been found to
    correspond most of the time to non-fluorescent beads, given the
    relatively high distance between events. Therefore, we consider it a
    distinct cluster.

    `clustering_dbscan` internally uses `DBSCAN` from the ``scikit-learn``
    library. For more information, consult their documentation.

    """
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
    n_samples_cluster = [np.sum(labels==li) for li in labels_all]

    # Check that no cluster is too small.
    # Clusters are assumed to be uniformly distributed. Any cluster 10 std 
    # smaller than the expected size (under a binomial distribution) will be 
    # assimilated with the next smallest
    # Larger than expected clusters will be assumed to correspond to clusters
    # containing data for 2 or more bead types.
    p = 1./n_clusters_exp
    n_samples_exp = data.shape[0]*p
    n_samples_std = np.sqrt(data.shape[0]*p*(1-p))
    while(True):
        cluster_size = np.array([np.sum(labels==li) for li in labels_all])
        cluster_i = np.argsort(cluster_size)
        if cluster_size[cluster_i[0]] < n_samples_exp - n_samples_std*10:
            labels[labels==labels_all[cluster_i[0]]] = labels_all[cluster_i[1]]
            labels_all.remove(labels_all[cluster_i[0]])
        else:
            break

    # Change the cluster numbers to a contiguous positive sequence
    labels_checked = -1*np.ones(len(labels))
    cn = 0
    for li in labels_all:
        labels_checked[labels==li] = cn
        cn = cn + 1
    labels = labels_checked

    assert(np.any(labels==-1) == False)

    return labels

def clustering_distance(data, n_clusters = 8):
    """
    Find clusters in the data array based on distance to the origin.

    This function sorts all the samples in `data` based on their Euclidean
    distance to the origin. Then, the ``n/n_clusters`` samples closest to
    the origin are assigned to cluster 0, the next ``n/n_clusters`` are
    assigned to cluster 1, and so on.

    Parameters
    ----------
    data : array_like
        NxD array to cluster.
    n_clusters : int, optional
        Number of clusters to find.

    Returns
    -------
    labels : array
        Nx1 array with labels for each element in `data`, assigning
        ``data[i]`` to cluster ``labels[i]``.

    """
    # Number of elements per cluster
    fractions = np.ones(n_clusters)*1./n_clusters

    n_per_cluster = fractions*data.shape[0]
    cluster_cum = np.append([0], np.cumsum(n_per_cluster))

    # Get distance and sort based on it
    dist = np.sum(data**2., axis = 1)
    sorted_i = np.argsort(dist)

    # Initialize labels
    labels = np.ones(data.shape[0])*-1

    # Assign labels
    for i in range(n_clusters):
        il = int(cluster_cum[i])
        ih = int(cluster_cum[i+1])
        sorted_i_i = sorted_i[il:ih]
        labels[sorted_i_i] = i

    return labels


def clustering_gmm(data, n_clusters = 8, initialization = 'distance_sub', 
    tol = 1e-7, min_covar = 1e-2):
    """
    Find clusters in an array using Gaussian Mixture Models (GMM).

    The likelihood maximization method used requires an initial parameter
    choice for the Gaussian pdfs, and the results can be fairly sensitive
    to it. `clustering_gmm` can perform two types of initialization, which
    we have found work well with calibration beads data. On the first one,
    the function group samples in `data` by their Euclidean distance to the
    origin, similarly to what is done in `clustering_distance`. Then, the
    function calculates the Gaussian Mixture parameters, assuming that this
    clustering is correct. These parameters are used as the initial
    conditions. The second initialization procedure also starts by
    clustering based on distance to the origin. Then, for each cluster,
    the 50% datapoints farther away from the mean are discarded, and the
    rest are used to calculate the initial parameters. The parameter
    `initialization` selects any of these two procedures.

    Parameters
    ----------
    data : array_like
        NxD array to cluster.
    n_clusters : int, optional
        Number of clusters to find.
    initialization : {'distance', 'distance_sub'}, optional
        Initialization method.
    tol : float, optional
        Tolerance for convergence of GMM method. Passed directly to
        ``scikit-learn``'s GMM.
    min_covar : float, optional
        Minimum covariance. Passed directly to ``scikit-learn``'s GMM.

    Returns
    -------
    labels : array
        Nx1 array with labels for each element in `data`, assigning
        ``data[i]`` to cluster ``labels[i]``.

    Notes
    -----
    GMM finds clusters by fitting a linear combination of `n_clusters`
    Gaussian probability density functions (pdf) to `data`, by likelihood
    maximization.

    `clustering_gmm` internally uses `GMM` from the ``scikit-learn``
    library. For more information, consult their documentation.

    """
    # Initialization method
    if initialization == 'distance':
        # Perform distance-based clustering
        labels = fc.mef.clustering_distance(data, n_clusters)
        labels_all = list(set(labels))
        data_clustered = [data[labels == i] for i in labels_all]
        
        # Initialize parameters for GMM
        weights = np.tile(1.0 / n_clusters,
                                        n_clusters)
        means = np.array([np.mean(di, axis = 0) 
            for di in data_clustered])
        
        if data.shape[1] == 1:
            covars = [np.cov(di.T).reshape(1,1) for di in data_clustered]
        else:
            covars = [np.cov(di.T) for di in data_clustered]

        # Initialize GMM object
        gmm = GMM(n_components = n_clusters, tol = tol, min_covar = min_covar,
            covariance_type = 'full', params = 'mc', init_params = '')
        gmm.weight_ = weights
        gmm.means_ = means
        gmm.covars_ = covars

    elif initialization == 'distance_sub':
        # Initialize parameters for GMM
        weights = np.tile(1.0 / n_clusters,
                                        n_clusters)
        means = []
        covars = []

        # Get distance and sort based on it
        dist = np.sum(data**2., axis = 1)
        sorted_i = np.argsort(dist)

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
            means.append(np.mean(data_i, axis = 0))
            if data.shape[1] == 1:
                covars.append(np.cov(data_i.T).reshape(1,1))
            else:
                covars.append(np.cov(data_i.T))
        means = np.array(means)

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
    labels = [np.random.choice(range(n_clusters), p = ri) for ri in resp]

    return labels

def find_peaks_smoothed_mode(data, min_val = 0, max_val = 1023):
    """
    Find the mode of a dataset by finding the peak of a smoothed histogram.

    This function finds the mode of a dataset by calculating the histogram,
    using a 1D Gaussian filter to smooth out the histogram, and identifying
    the maximum value of the smoothed histogram. The ``sigma`` parameter of
    the Gaussian filter is taken from the standard deviation of `data`.

    Parameters
    ----------
    data : array
        Nx1 array to calculate the mode from. `data` is assumed to contain
        only integers.
    min_val : int, optional
        Minimum possible value in `data`.
    max_val : int, optional
        Maximum possible value in `data`.

    Returns
    -------
    peak : float
        Mode of `data`.
    hist_smooth : array
        ``(max_val - min_val + 1)``-long array containing the smoothed
        histogram.

    """
    # Calculate bin edges and centers
    bin_edges = np.arange(min_val, max_val + 2) - 0.5
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_centers = np.arange(min_val, max_val + 1)

    # Identify peak
    # Calculate sample mean and standard deviation
    # mu = np.mean(data)
    sigma = np.std(data)
    # Calculate histogram
    hist, __ = np.histogram(data, bin_edges)
    # Do Gaussian blur on histogram
    # We have found empirically that using one half of the distribution's 
    # standard deviation results in a nice fit.
    hist_smooth = scipy.ndimage.filters.gaussian_filter1d(hist, sigma/2.)
    # Extract peak
    i_max = np.argmax(hist_smooth)
    peak = bin_centers[i_max]

    return peak, hist_smooth

def find_peaks_median(data):
    """
    Find the median of a data array.

    Parameters
    ----------
    data : array
        Nx1 array to calculate the median from.

    Returns
    -------
    peak : int or float
        Median of `data`.

    """
    peak = np.median(data)
    return peak

def select_peaks_proximity(peaks_ch,
                           peaks_mef,
                           peaks_ch_std,
                           peaks_ch_std_mult_l = 2.5,
                           peaks_ch_std_mult_r = 2.5,
                           peaks_ch_min = 0,
                           peaks_ch_max = 1023):
    """
    Select bead subpopulations based on proximity to a minimum and maximum.

    This function discards some values from `peaks_ch` if they're closer
    than `peaks_ch_std_mult_l` standard deviations to `peaks_ch_min`, or
    `peaks_ch_std_mult_r` standard deviations to `peaks_ch_max`. Standard
    deviations should be provided in `peaks_ch_std`. Then, it discards the
    corresponding values in `peaks_mef`. Finally, it discards the values in
    `peaks_mef` that have an undetermined value (NaN), and the
    corresponding values in peaks_ch.

    Parameters
    ----------
    peaks_ch : array
        Sorted fluorescence values of bead populations in channel units.
    peaks_mef : array
        Sorted fluorescence values of bead populations in MEF units.
    peaks_ch_std_mult_l, peaks_ch_std_mult_r : float, optional
        Number of standard deviations from `peaks_ch_min` and
        `peaks_ch_max`, respectively, that a value in `peaks_ch` has to be
        closer than to be discarded.
    peaks_ch_min, peaks_ch_max : int, optional
        Minimum and maximum tolerable fluorescence value in channel units.

    Returns
    -------
    sel_peaks_ch : array
        Selected fluorescence values of bead populations in channel units.
    sel_peaks_mef : array
        Selected fluorescence values of bead populations in MEF units.

    """
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
    unknown_mef = np.isnan(sel_peaks_mef)
    n_unknown_mef = np.sum(unknown_mef)
    if n_unknown_mef > 0:
        sel_peaks_ch = sel_peaks_ch[np.invert(unknown_mef)]
        sel_peaks_mef = sel_peaks_mef[np.invert(unknown_mef)]

    return sel_peaks_ch, sel_peaks_mef

def fit_standard_curve(peaks_ch, peaks_mef):
    """
    Fit a standard curve to known fluorescence values of calibration beads.

    Parameters
    ----------
    peaks_ch : array
        Fluorescence values of bead populations in channel units.
    peaks_mef : array
        Fluorescence values of bead populations in MEF units.

    Returns
    -------
    sc : function
        Standard curve that transforms fluorescence from channel units to
        MEF units.
    sc_beads : function
        Bead fluorescence model, mapping bead fluorescence in channel space
        to bead fluorescence in MEF units, without autofluorescence.
    sc_params : array
        Fitted parameters of the bead fluorescence model: ``[m, b,
        fl_mef_auto]``.

    Notes
    -----
    The following model is used to describe bead fluorescence:

        m*fl_ch[i] + b = log(fl_mef_auto + fl_mef[i])

    where fl_ch[i] is the fluorescence of bead subpopulation i in channel
    units and fl_mef[i] is the corresponding fluorescence in MEF units. The
    model includes 3 parameters: m (slope), b (intercept), and fl_mef_auto
    (bead autofluorescence).

    The bead fluorescence model is fit in a log-MEF space using nonlinear
    least squares regression (as opposed to fitting an exponential model in
    MEF space). In our experience, fitting in the log-MEF space weights the
    residuals more evenly, whereas fitting an exponential vastly overvalues
    the brighter peaks.

    A standard curve is constructed by solving for fl_mef. As cell samples
    may not have the same autofluorescence as beads, the bead
    autofluorescence term (fl_mef_auto) is omitted from the standard curve;
    the user is expected to use an appropriate white cell sample to account
    for cellular autofluorescence if necessary. The returned standard curve
    mapping fluorescence in channel units to MEF units is thus of the
    following form:

        fl_mef = exp(m*fl_ch + b)

    """
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
    find_peaks_method = 'median', find_peaks_params = {},
    select_peaks_method = 'proximity', select_peaks_params = {},
    verbose = False, plot = False, plot_dir = None, plot_filename = None,
    full = False):
    """
    Get a transformation function to convert flow cytometry data to MEF.

    Parameters
    ----------
    data_beads : FCSData object
        Flow cytometry data, taken from calibration beads.
    peaks_mef : array
        Known MEF values of the calibration beads' subpopulations, for
        each channel specified in `mef_channels`.
    mef_channels : int, or str, or list of int, or list of str
        Channels for which to generate transformation functions.
    cluster_channels : list, optional
        Channels used for clustering. If not specified, used the first
        channel in `data_beads`.
    verbose : bool, optional
        Flag specifying whether to print information about step completion
        and warnings.
    plot : bool, optional
        Flag specifying whether to produce diagnostic plots.
    plot_dir : str, optional
        Directory where to save diagnostics plots. Ignored if `plot` is
        False. If ``plot==True`` and ``plot_dir is None``, plot without
        saving.
    plot_filename : str, optional
        Name to use for plot files. If None, use ``str(data_beads)``.
    full : bool, optional
        Flag specifying whether to include intermediate results in the
        output. If `full` is True, the function returns a named tuple with
        fields as described below. If `full` is False, the function only
        returns the calculated transformation function.

    Returns
    -------
    transform_fxn : function, if ``full==False``
        Transformation function to convert flow cytometry data from channel
        units to MEF. This function has the same basic signature as the
        general transformation function specified in ``fc.transform``.
    namedtuple, if ``full==True``
        ``namedtuple``, containing the following fields in this order:
        transform_fxn : function
            Transformation function to convert flow cytometry data from
            channel units to MEF. This function has the same basic
            signature as the general transformation function specified in
            ``fc.transform``.
        clustering_res : dict
            Results of the clustering step, containing the following
            fields:
            labels : array
                Labels for each element in `data_beads`.
        peak_find_res : dict
            Results of the calculation of bead subpopulations'
            fluorescence, containing the following fields:
            peaks_ch : list
                The representative fluorescence of each subpopulation, for
                each channel in `mef_channels`.
            peaks_hists : list
                Only included if ``find_peaks_method=='smoothed_mode'. The
                smoothed histogram of each subpopulation, for each channel
                in `mef_channels`.
        peak_sel_res : dict
            Results of the subpopulation selection step, containing the
            following fields:
            sel_peaks_ch : list
                The fluorescence values of each selected subpopulation in
                channel units, for each channel in `mef_channels`.
            sel_peaks_mef : list
                The fluorescence values of each selected subpopulation in
                MEF units, for each channel in `mef_channels`.
        fitting_res : dict
            Results of the model fitting step, containing the following
            fields:
            sc : list
                Functions encoding the standard curves, for each channel in
                `mef_channels`.
            sc_beads : list
                Functions encoding the fluorescence model of the
                calibration beads, for each channel in `mef_channels`.
            sc_params : list
                Fitted parameters of the bead fluorescence model: ``[m, b,
                fl_mef_auto]``, for each channel in `mef_chanels`.

    Other parameters
    ----------------
    cluster_method : {'dbscan', 'distance', 'gmm'}, optional
        Method used for clustering, or identification of subpopulations.
    cluster_params : dict, optional
        Parameters to pass to the clustering method.
    find_peaks_method : {'smoothed_mode', 'median'}, optional
        Method used to calculate the representative fluorescence of each
        subpopulation.
    find_peaks_params : dict, optional
        Parameters to pass to the method that calculates the fluorescence
        of each subpopulation.
    select_peaks_method : {'proximity'}, optional
        Method to use for peak selection.
    select_peaks_params : dict, optional
        Parameters to pass to the peak selection method.

    Notes
    -----
    The steps involved in generating the MEF transformation function are:

    1. The individual subpopulations of beads are first identified using a
       clustering method of choice.
    2. The fluorescence of each subpopulation is calculated for each
       cluster, for each channel in `mef_channels`.
    3. Some subpopulations are then discarded if they are close to either
       the minimum or the maximum channel value. In addition, if the MEF
       value of some subpopulation is unknown (represented as a ``NaN`` in
       `peaks_mef`), the whole subpopulation is also discarded.
    4. The measured fluorescence of each subpopulation is compared with
       the known MEF values in `peaks_mef`, and a standard curve function
       is generated using the appropriate MEF model.

    At the end, a transformation function is generated using the calculated
    standard curves, `mef_channels`, and ``fc.transform.to_mef()``.

    Note that applying the resulting transformation function to other
    flow cytometry samples only yields correct results if they have been
    taken at the same settings as the calibration beads, for all channels
    in `mef_channels`.

    """
    if verbose:
        prev_precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=2)
    # Create directory if plot is True
    if plot and plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    # Default plot filename
    if plot_filename is None:
        plot_filename = str(data_beads)

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

    labels_all = np.array(list(set(labels)))
    n_clusters = len(labels_all)
    data_clustered = [data_beads[labels == i] for i in labels_all]

    # Print information
    if verbose:
        print("- STEP 1. CLUSTERING.")
        print("Number of clusters found: {}".format(n_clusters))
        # Calculate percentage of each cluster
        data_count = np.array([di.shape[0] for di in data_clustered])
        data_perc = data_count*100.0/data_count.sum()
        print("Percentage of samples in each cluster:")
        print(data_perc)
    # Plot
    if plot:
        # Sort
        cluster_dist = [np.sum((np.mean(di[:,cluster_channels], 
                axis = 0))**2) for di in data_clustered]
        cluster_sorted_ind = np.argsort(cluster_dist)
        data_plot = [data_clustered[i] for i in cluster_sorted_ind]
            
        if len(cluster_channels) == 2:
            if plot_dir is not None:
                savefig = '{}/cluster_{}.png'.format(plot_dir, plot_filename)
            else:
                savefig = None
            # Plot
            plt.figure(figsize = (6,4))
            fc.plot.scatter2d(data_plot, 
                    channels = cluster_channels,
                    savefig = savefig)
            if plot_dir is not None:
                plt.close()
            
        if len(cluster_channels) == 3:
            if plot_dir is not None:
                savefig = '{}/cluster_{}.png'.format(plot_dir, plot_filename)
            else:
                savefig = None
            # Plot
            plt.figure(figsize = (8,6))
            fc.plot.scatter3d(data_plot, 
                    channels = cluster_channels,
                    savefig = savefig)
            if plot_dir is not None:
                plt.close()

    # mef_channels and peaks_mef should be iterables.
    if hasattr(mef_channels, '__iter__'):
        mef_channel_all = list(mef_channels)
        peaks_mef_all = np.array(peaks_mef).copy()
    else:
        mef_channel_all = [mef_channels]
        peaks_mef_all = np.array([peaks_mef])

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
            print("- MEF transformation for channel {}...".format(mef_channel))

        # Step 2. Find peaks in each one of the clusters. 
        # ===============================================

        # Find peaks on all the channel data
        min_fl = data_channel.channel_info[0]['range'][0]
        max_fl = data_channel.channel_info[0]['range'][1]
        if find_peaks_method == 'smoothed_mode':
            # Set default values for limit values
            if 'min_val' not in find_peaks_params:
                find_peaks_params['min_val'] = min_fl
            if 'max_val' not in find_peaks_params:
                find_peaks_params['max_val'] = max_fl
            # Get peak values
            peaks_hists = [find_peaks_smoothed_mode(di[:,mef_channel], 
                                                        **find_peaks_params)
                                    for di in data_clustered]
            peaks_ch = np.array([ph[0] for ph in peaks_hists])
            hists_smooth = [ph[1] for ph in peaks_hists]
        elif find_peaks_method == 'median':
            peaks_ch = np.array([find_peaks_median(di[:,mef_channel],
                                                        **find_peaks_params)
                                    for di in data_clustered])
        else:
            raise ValueError("Peak finding method {} not recognized."
                .format(find_peaks_method))
        # Sort peaks and clusters
        ind_sorted = np.argsort(peaks_ch)
        peaks_sorted = peaks_ch[ind_sorted]
        data_sorted = [data_clustered[i] for i in ind_sorted]

        # Accumulate results
        if full:
            peaks_ch_all.append(peaks_ch)
            if find_peaks_method == 'smoothed_mode':
                peaks_hists_all.append(hists_smooth)
        # Print information
        if verbose:
            print("- STEP 2. PEAK IDENTIFICATION.")
            print("Channel peaks identified:")
            print(peaks_sorted)
        # Plot
        if plot:
            # Get colors for peaks
            colors = [fc.plot.cmap_default(i)\
                                for i in np.linspace(0, 1, n_clusters)]
            # Plot histograms
            plt.figure(figsize = (8,4))
            fc.plot.hist1d(data_plot, channel = mef_channel, div = 4, 
                alpha = 0.75)
            # Plot smoothed histograms and peaks
            for c, i in zip(colors, cluster_sorted_ind):
                # Smoothed histogram, if applicable
                if find_peaks_method == 'smoothed_mode':
                    h = hists_smooth[i]
                    plt.plot(np.linspace(min_fl, max_fl, len(h)), h*4, 
                        color = c)
                # Peak values
                p = peaks_ch[i]
                ylim = plt.ylim()
                plt.plot([p, p], [ylim[0], ylim[1]], color = c)
                plt.ylim(ylim)
            # Save and close
            if plot_dir is not None:
                plt.tight_layout()
                plt.savefig('{}/peaks_{}_{}.png'.format(plot_dir,
                                    mef_channel, plot_filename), dpi = 300)
                plt.close()

        # 3. Select peaks for fitting
        # ===========================
        
        # Print information
        if verbose:
            print("- STEP 3. PEAK SELECTION.")
            print("MEF peaks provided:")
            print(peaks_mef_channel)

        if select_peaks_method == 'proximity':
            # Get the standard deviation of each peak
            peaks_std = np.array([np.std(di[:,mef_channel]) \
                for di in data_sorted])
            if verbose:
                print("Standard deviations of channel peaks:")
                print(peaks_std)
            # Set default limits
            if 'peaks_ch_min' not in select_peaks_params:
                select_peaks_params['peaks_ch_min'] = min_fl
            if 'peaks_ch_max' not in select_peaks_params:
                select_peaks_params['peaks_ch_max'] = max_fl
            # Select peaks
            sel_peaks_ch, sel_peaks_mef = select_peaks_proximity(peaks_sorted,
                    peaks_mef_channel, peaks_ch_std = peaks_std,
                    **select_peaks_params)
        else:
            raise ValueError("Peak selection method {} not recognized."
                .format(select_peaks_method))

        # Accumulate results
        if full:
            sel_peaks_ch_all.append(sel_peaks_ch)
            sel_peaks_mef_all.append(sel_peaks_mef)
        # Print information
        if verbose:
            print("{} peaks retained.".format(len(sel_peaks_ch)))
            print("Selected channel peaks:")
            print(sel_peaks_ch)
            print("Selected MEF peaks:")
            print(sel_peaks_mef)

        # 4. Get standard curve
        # ======================
        sc, sc_beads, sc_params = fit_standard_curve(sel_peaks_ch, 
            sel_peaks_mef)
        if verbose:
            print("- STEP 4. STANDARDS CURVE FITTING.")
            print("Fitted parameters:")
            print(sc_params)

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
            # Compute filename to save
            if plot_dir is not None:
                savefig = '{}/std_crv_{}_{}.png'.format(plot_dir,
                                                        mef_channel,
                                                        plot_filename,
                                                        )
            else:
                savefig = None
            # Plot standard curve
            plt.figure(figsize = (6,4))
            fc.plot.mef_std_crv(sel_peaks_ch, 
                    sel_peaks_mef,
                    sc_beads,
                    sc,
                    xlabel = xlabel,
                    ylabel = 'MEF',
                    savefig = savefig)
            if plot_dir is not None:
                plt.close()

    # Make output transformation function
    transform_fxn = functools.partial(fc.transform.to_mef,
                                    sc_list = sc_all,
                                    sc_channels = mef_channel_all)

    if verbose:
        np.set_printoptions(precision = prev_precision)

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
