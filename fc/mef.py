"""
Functions for transforming flow cytometer data to MEF units.

"""

import os
import functools
import collections

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.mixture import GMM 

import fc.plot
import fc.transform

# Use default colors from palettable if available
try:
    import palettable
except ImportError, e:
    standard_curve_colors = ['b', 'g', 'r']
else:
    standard_curve_colors = \
        palettable.colorbrewer.qualitative.Paired_12.mpl_colors[1::2]

def clustering_gmm(data,
                   n_clusters=8,
                   initialization='distance_sub',
                   tol=1e-7,
                   min_covar=1e-2):
    """
    Find clusters in an array using Gaussian Mixture Models (GMM).

    The likelihood maximization method used requires an initial parameter
    choice for the Gaussian pdfs, and the results can be fairly sensitive
    to it. `clustering_gmm` can perform two types of initialization, which
    we have found work well with calibration beads data. On the first one,
    the function group samples in `data` by their Euclidean distance to the
    origin, Then, the function calculates the Gaussian Mixture parameters,
    assuming that this clustering is correct. These parameters are used as
    the initial conditions. The second initialization procedure also starts
    by clustering based on distance to the origin. Then, for each cluster,
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
        weights = np.tile(1.0 / n_clusters, n_clusters)
        means = np.array([np.mean(di, axis=0) for di in data_clustered])
        
        if data.shape[1] == 1:
            covars = [np.cov(di.T).reshape(1,1) for di in data_clustered]
        else:
            covars = [np.cov(di.T) for di in data_clustered]

        # Initialize GMM object
        gmm = GMM(n_components=n_clusters,
                  tol=tol,
                  min_covar=min_covar,
                  covariance_type='full',
                  params='mc',
                  init_params='')
        gmm.weight_ = weights
        gmm.means_ = means
        gmm.covars_ = covars

    elif initialization == 'distance_sub':
        # Initialize parameters for GMM
        weights = np.tile(1.0 / n_clusters, n_clusters)
        means = []
        covars = []

        # Get distance and sort based on it
        dist = np.sum(data**2., axis=1)
        sorted_i = np.argsort(dist)

        # Expected number of elements per cluster
        n_per_cluster = data.shape[0]/float(n_clusters)

        # Get the means and covariances per cluster
        # We will just use a fraction of ``1 - 2*discard_frac`` of the data.
        # Data at the edges that actually corresponds to another cluster can
        # really mess up the final result.
        discard_frac = 0.25
        for i in range(n_clusters):
            il = int((i + discard_frac)*n_per_cluster)
            ih = int((i + 1 - discard_frac)*n_per_cluster)
            sorted_i_i = sorted_i[il:ih]
            data_i = data[sorted_i_i]
            means.append(np.mean(data_i, axis=0))
            if data.shape[1] == 1:
                covars.append(np.cov(data_i.T).reshape(1,1))
            else:
                covars.append(np.cov(data_i.T))
        means = np.array(means)

        # Initialize GMM object
        gmm = GMM(n_components=n_clusters,
                  tol=tol,
                  min_covar=min_covar,
                  covariance_type='full',
                  params='mc',
                  init_params='')
        gmm.weight_ = weights
        gmm.means_ = means
        gmm.covars_ = covars

    else:
        raise ValueError("initialization method {} not implemented"
            .format(initialization))

    # Fit 
    gmm.fit(data)
    # Get labels using the responsibilities
    # This avoids the complete elimination of a cluster if two or more clusters
    # have very similar means.
    resp = gmm.predict_proba(data)
    labels = [np.random.choice(range(n_clusters), p=ri) for ri in resp]

    return labels

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
                           peaks_ch_std_mult_l=2.5,
                           peaks_ch_std_mult_r=2.5,
                           peaks_ch_min=0,
                           peaks_ch_max=1023):
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
    if ((peaks_ch[0] - peaks_ch_std[0]*peaks_ch_std_mult_l) <= peaks_ch_min and
        (peaks_ch[-1] + peaks_ch_std[-1]*peaks_ch_std_mult_r) >= peaks_ch_max):
        raise ValueError("peaks are being cut off at both sides")
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
               peaks_ch_std[-1-discard_ch_n]*peaks_ch_std_mult_r)>=peaks_ch_max:
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
        ValueError("number of MEF values and channel peaks does not match")
    
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
    if len(peaks_ch) != len(peaks_mef):
        raise ValueError("peaks_ch and peaks_mef have different lengths")
    # Check that we have at least three points
    if len(peaks_ch) <= 2:
        raise ValueError("standard curve model requires at least three "
            "bead peak values")
        
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

def plot_standard_curve(peaks_ch, 
                        peaks_mef,
                        sc_beads,
                        sc_abs,
                        xlim=(0.,1023.),
                        ylim=(1.,1e8)):
    """
    Plot a standard curve with fluorescence of calibration beads.

    Parameters
    ----------
    peaks_ch : array_like
        Fluorescence of the calibration beads' subpopulations, in channel
        numbers.
    peaks_mef : array_like
        Fluorescence of the calibration beads' subpopulations, in MEF
        units.
    sc_beads : function
        The calibration beads fluorescence model.
    sc_abs : function
        The standard curve (transformation function from channel number to
        MEF units).

    """
    # Generate x data
    xdata = np.linspace(xlim[0], xlim[1], 200)

    # Plot
    plt.plot(peaks_ch,
             peaks_mef,
             'o',
             label='Beads',
             color=standard_curve_colors[0])
    plt.plot(xdata,
             sc_beads(xdata),
             label='Beads model',
             color=standard_curve_colors[1])
    plt.plot(xdata,
             sc_abs(xdata),
             label='Standard curve',
             color=standard_curve_colors[2])

    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.legend(loc = 'best')

def get_transform_fxn(data_beads,
                      peaks_mef,
                      mef_channels,
                      cluster_method='gmm',
                      cluster_params={},
                      cluster_channels=0,
                      find_peaks_method='median',
                      find_peaks_params={},
                      select_peaks_method='proximity',
                      select_peaks_params={},
                      verbose=False,
                      plot=False,
                      plot_dir=None,
                      plot_filename=None,
                      full=False):
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
    cluster_method : {'gmm'}, optional
        Method used for clustering, or identification of subpopulations.
    cluster_params : dict, optional
        Parameters to pass to the clustering method.
    find_peaks_method : {'median'}, optional
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

    ###
    # 1. Clustering
    ###
    if cluster_method == 'gmm':
        labels = clustering_gmm(data_beads[:,cluster_channels], 
            **cluster_params)
    else:
        raise ValueError("clustering method {} not recognized"
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
        cluster_dist = [np.sum((np.mean(di[:,cluster_channels], axis=0))**2)
                        for di in data_clustered]
        cluster_sorted_ind = np.argsort(cluster_dist)
        data_plot = [data_clustered[i] for i in cluster_sorted_ind]
            
        if len(cluster_channels) == 2:
            if plot_dir is not None:
                savefig = '{}/cluster_{}.png'.format(plot_dir, plot_filename)
            else:
                savefig = None
            # Plot
            plt.figure(figsize=(6,4))
            fc.plot.scatter2d(data_plot, 
                              channels=cluster_channels,
                              savefig=savefig)
            if plot_dir is not None:
                plt.close()
            
        elif len(cluster_channels) == 3:
            if plot_dir is not None:
                savefig = '{}/cluster_{}.png'.format(plot_dir, plot_filename)
            else:
                savefig = None
            # Plot
            plt.figure(figsize=(8,6))
            fc.plot.scatter3d_and_projections(data_plot,
                                              channels=cluster_channels,
                                              savefig=savefig)
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

        ###
        # 2. Find peaks in each one of the clusters.
        ###

        # Find peaks on all the channel data
        min_fl = data_channel.domain(0)[0]
        max_fl = data_channel.domain(0)[-1]
        if find_peaks_method == 'median':
            peaks_ch = np.array([find_peaks_median(di[:,mef_channel],
                                                   **find_peaks_params)
                                 for di in data_clustered])
        else:
            raise ValueError("peak finding method {} not recognized"
                .format(find_peaks_method))
        # Sort peaks and clusters
        ind_sorted = np.argsort(peaks_ch)
        peaks_sorted = peaks_ch[ind_sorted]
        data_sorted = [data_clustered[i] for i in ind_sorted]

        # Accumulate results
        if full:
            peaks_ch_all.append(peaks_ch)
        # Print information
        if verbose:
            print("- STEP 2. PEAK IDENTIFICATION.")
            print("Channel peaks identified:")
            print(peaks_sorted)
        # Plot
        if plot:
            # Get colors for peaks
            colors = [fc.plot.cmap_default(i)
                      for i in np.linspace(0, 1, n_clusters)]
            # Plot histograms
            plt.figure(figsize=(8,4))
            fc.plot.hist1d(data_plot,
                           channel=mef_channel,
                           div=4,
                           alpha=0.75)
            # Plot histograms and peaks
            for c, i in zip(colors, cluster_sorted_ind):
                # Peak values
                p = peaks_ch[i]
                ylim = plt.ylim()
                plt.plot([p, p], [ylim[0], ylim[1]], color=c)
                plt.ylim(ylim)
            # Save and close
            if plot_dir is not None:
                plt.tight_layout()
                plt.savefig('{}/peaks_{}_{}.png'.format(plot_dir,
                                                        mef_channel,
                                                        plot_filename),
                            dpi=fc.plot.savefig_dpi)
                plt.close()

        ###
        # 3. Select peaks for fitting
        ###
        
        # Print information
        if verbose:
            print("- STEP 3. PEAK SELECTION.")
            print("MEF peaks provided:")
            print(peaks_mef_channel)

        if select_peaks_method == 'proximity':
            # Get the standard deviation of each peak
            peaks_std = np.array([np.std(di[:,mef_channel])
                                  for di in data_sorted])
            if verbose:
                print("Standard deviations of channel peaks:")
                print(peaks_std)
            # Set default limits: throw away 1% of the range
            if 'peaks_ch_min' not in select_peaks_params:
                select_peaks_params['peaks_ch_min'] = min_fl*0.015
            if 'peaks_ch_max' not in select_peaks_params:
                select_peaks_params['peaks_ch_max'] = max_fl*0.985
            # Select peaks
            sel_peaks_ch, sel_peaks_mef = select_peaks_proximity(
                peaks_sorted,
                peaks_mef_channel,
                peaks_ch_std=peaks_std,
                **select_peaks_params)
        else:
            raise ValueError("peak selection method {} not recognized"
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

        ###
        # 4. Get standard curve
        ###
        sc, sc_beads, sc_params = fit_standard_curve(
            sel_peaks_ch,
            sel_peaks_mef)
        if verbose:
            print("- STEP 4. STANDARD CURVE FITTING.")
            print("Fitted parameters:")
            print(sc_params)

        sc_all.append(sc)
        if full:
            sc_beads_all.append(sc_beads)
            sc_params_all.append(sc_params)

        # Plot
        if plot:
            # Plot standard curve
            plt.figure(figsize=(6,4))
            plot_standard_curve(sel_peaks_ch,
                                sel_peaks_mef,
                                sc_beads,
                                sc,
                                xlim=(min_fl, max_fl))
            plt.xlabel('{} (Channel Units)'.format(data_channel.channels[0]))
            plt.ylabel('MEF')
            # Save if required
            if plot_dir is not None:
                plt.tight_layout()
                plt.savefig('{}/std_crv_{}_{}.png'.format(plot_dir,
                                                          mef_channel,
                                                          plot_filename),
                            dpi=fc.plot.savefig_dpi)
                plt.close()

    # Make output transformation function
    transform_fxn = functools.partial(fc.transform.to_mef,
                                      sc_list=sc_all,
                                      sc_channels=mef_channel_all)

    if verbose:
        np.set_printoptions(precision=prev_precision)

    if full:
        # Clustering results
        clustering_res = {}
        clustering_res['labels'] = labels
        # Peak finding results
        peak_find_res = {}
        peak_find_res['peaks_ch'] = peaks_ch_all
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
        out = MEFOutput(transform_fxn=transform_fxn,
                        clustering_res=clustering_res,
                        peak_find_res=peak_find_res,
                        peak_sel_res=peak_sel_res,
                        fitting_res=fitting_res)
        return out
    else:
        return transform_fxn
