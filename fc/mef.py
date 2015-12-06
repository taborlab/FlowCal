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
                   tol=1e-7,
                   min_covar=1e-2):
    """
    Find clusters in an array using Gaussian Mixture Models (GMM).

    Parameters
    ----------
    data : array_like
        NxD array to cluster.
    n_clusters : int, optional
        Number of clusters to find.
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
    Gaussian probability density functions (pdf) to `data` using
    Expectation Maximization (EM).

    GMM can be fairly sensitive to the initial parameter choice. To
    generate a reasonable set of initial conditions, `clustering_gmm`
    first divides all samples in `data` into `n_clusters` groups of the
    same size based on their Euclidean distance to the origin. Then, for
    each group, the 50% samples farther away from the mean are discarded.
    The mean and covariance are calculated from the remaining samples of
    each group, and used as initial conditions for the GMM EM algorithm.

    `clustering_gmm` internally uses `GMM` from the ``scikit-learn``
    library, with full covariance matrices for each cluster, and a fixed,
    uniform set of weights. This means that `clustering_gmm` implicitly
    assumes that all bead subpopulations have roughly the same amount of
    events. For more information, consult ``scikit-learn``'s documentation.

    """
    ###
    # Parameter initialization
    ###
    weights = np.tile(1.0 / n_clusters, n_clusters)
    means = []
    covars = []

    # Get distance and sort based on it
    dist = np.sum(data**2., axis=1)
    sorted_i = np.argsort(dist)

    # Expected number of elements per cluster
    n_per_cluster = data.shape[0]/float(n_clusters)

    # Get means and covariances per cluster
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

    ###
    # Run Gaussian Mixture Model Clustering
    ###

    # Initialize GMM object
    gmm = GMM(n_components=n_clusters,
              tol=tol,
              min_covar=min_covar,
              covariance_type='full',
              params='mc',
              init_params='')

    # Set initial parameters
    gmm.weight_ = weights
    gmm.means_ = means
    gmm.covars_ = covars

    # Fit 
    gmm.fit(data)

    # Get labels by sampling from the responsibilities
    # This avoids the complete elimination of a cluster if two or more clusters
    # have very similar means.
    resp = gmm.predict_proba(data)
    labels = [np.random.choice(range(n_clusters), p=ri) for ri in resp]

    return labels

def selection_proximity(populations,
                        th_l=None,
                        th_r=None,
                        std_mult_l=2.5,
                        std_mult_r=2.5):
    """
    Select populations based on proximity to a low and high thresholds.

    This function selects populations from `populations` if their means are
    farther than `std_mult_l` standard deviations to `th_l`, or
    `std_mult_r` standard deviations to `th_r`.

    Parameters
    ----------
    populations : list of FCSData objects
        Populations to select or discard.
    th_l, th_r : int or float, optional
        Left and right threshold. If None, use 0.015 of the lowest value
        and 0.0985 of the highest value in the first population's domain.
    std_mult_l, std_mult_r : float, optional
        Number of standard deviations from `th_l` and `th_r`, respectively,
        that a population's mean has to be closer than to be discarded.

    Returns
    -------
    m : boolean array
        Set of flags indicating whether a population has been selected.

    """
    # Default thresholds
    if th_l is None:
        th_l = 0.015*populations[0].domain(0)[0]
    if th_r is None:
        th_r = 0.985*populations[0].domain(0)[-1]

    # Calculate means and standard deviations
    pop_mean = np.array([fc.stats.mean(p) for p in populations])
    pop_std = np.array([fc.stats.std(p) for p in populations])

    # Set minimum standard deviation to one.
    min_std = 1.0
    pop_std[pop_std < min_std] = min_std

    # Return populations that don't cross either threshold
    m = np.logical_and((pop_mean - std_mult_l*pop_std) > th_l,
                       (pop_mean - std_mult_r*pop_std) < th_r)
    return m

def fit_standard_curve(fl_channel, fl_mef):
    """
    Fit a standard curve to known fluorescence values of calibration beads.

    Parameters
    ----------
    fl_channel : array
        Fluorescence values of bead populations in channel units.
    fl_mef : array
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
    the brighter beads.

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
    if len(fl_channel) != len(fl_mef):
        raise ValueError("fl_channel and fl_mef have different lengths")
    # Check that we have at least three points
    if len(fl_channel) <= 2:
        raise ValueError("standard curve model requires at least three "
            "values")
        
    # Initialize parameters
    params = np.zeros(3)
    # Initial guesses:
    # 0: slope found by putting a line through the highest two points.
    # 1: y-intercept found by putting a line through highest two points.
    # 2: bead autofluorescence initialized to 100.
    params[0] = (np.log(fl_mef[-1]) - np.log(fl_mef[-2])) / \
                    (fl_channel[-1] - fl_channel[-2])
    params[1] = np.log(fl_mef[-1]) - params[0] * fl_channel[-1]
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
    err_par = lambda p: err_fun(p, fl_channel, fl_mef)
    res = minimize(err_par, params, options = {'gtol': 1e-6})

    # Separate parameters
    sc_params = res.x

    # Beads model function
    sc_beads = lambda x: fit_fun(sc_params, x)

    # Standard curve function
    sc = lambda x: sc_fun(sc_params, x)
    
    return (sc, sc_beads, sc_params)

def plot_standard_curve(fl_channel,
                        fl_mef,
                        sc_beads,
                        sc_abs,
                        xlim=(0.,1023.),
                        ylim=(1.,1e8)):
    """
    Plot a standard curve with fluorescence of calibration beads.

    Parameters
    ----------
    fl_channel : array_like
        Fluorescence of the calibration beads' subpopulations, in channel
        numbers.
    fl_mef : array_like
        Fluorescence of the calibration beads' subpopulations, in MEF
        units.
    sc_beads : function
        The calibration beads fluorescence model.
    sc_abs : function
        The standard curve (transformation function from channel number to
        MEF units).

    Other parameters:
    -----------------
    xlim : tuple, optional
        Limits for the x axis.
    ylim : tuple, optional
        Limits for the y axis.

    """
    # Generate x data
    xdata = np.linspace(xlim[0], xlim[1], 200)

    # Plot
    plt.plot(fl_channel,
             fl_mef,
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
                      mef_values,
                      mef_channels,
                      clustering_func=clustering_gmm,
                      clustering_params={},
                      clustering_channels=None,
                      statistic_func=fc.stats.median,
                      statistic_params={},
                      selection_func=selection_proximity,
                      selection_params={},
                      verbose=False,
                      plot=False,
                      plot_dir=None,
                      plot_filename=None,
                      full_output=False):
    """
    Get a transformation function to convert flow cytometry data to MEF.

    Parameters
    ----------
    data_beads : FCSData object
        Flow cytometry data, taken from calibration beads.
    mef_values : array
        Known MEF values of the calibration beads' subpopulations, for
        each channel specified in `mef_channels`.
    mef_channels : int, or str, or list of int, or list of str
        Channels for which to generate transformation functions.
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
    full_output : bool, optional
        Flag specifying whether to include intermediate results in the
        output. If `full_output` is True, the function returns a named
        tuple with fields as described below. If `full_output` is False,
        the function only returns the calculated transformation function.

    Returns
    -------
    transform_fxn : function, if ``full_output==False``
        Transformation function to convert flow cytometry data from channel
        units to MEF. This function has the same basic signature as the
        general transformation function specified in ``fc.transform``.
    namedtuple, if ``full_output==True``
        ``namedtuple``, containing the following fields in this order:
        transform_fxn : function
            Transformation function to convert flow cytometry data from
            channel units to MEF. This function has the same basic
            signature as the general transformation function specified in
            ``fc.transform``.
        clustering : dict
            Results of the clustering step, containing the following
            fields:
            labels : array
                Labels for each element in `data_beads`.
        statistic : dict
            Results of the calculation of bead subpopulations'
            fluorescence, containing the following fields:
            values : list
                The representative fluorescence values of each
                subpopulation, for each channel in `mef_channels`.
        selection : dict
            Results of the subpopulation selection step, containing the
            following fields:
            channel : list
                The fluorescence values of each selected subpopulation in
                channel units, for each channel in `mef_channels`.
            mef : list
                The fluorescence values of each selected subpopulation in
                MEF units, for each channel in `mef_channels`.
        fitting : dict
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
    clustering_func : function, optional
        Function used for clustering, or identification of subpopulations.
        The following signature is required: ``labels = clustering_func(
        data, n_clusters, **clustering_params)``, where `data` is a NxD
        FCSData object or numpy array, `n_clusters` is the expected number
        of bead subpopulations, and `labels` is a 1D numpy array of length
        N, assigning each event in `data` to one subpopulation.
    clustering_params : dict, optional
        Keyword parameters to pass to clustering_func.
    clustering_channels : list, optional
        Channels used for clustering. If not specified, use `mef_channels`.
        If more than three channels are specified, and `plot` is True, only
        a 3D scatter plot will be produced, using the first three channels.
    statistic_func : function, optional
        Function used to calculate the representative fluorescence of each
        subpopulation. Must have the following signature:
        ``s = statistic_func(data, **statistic_params)``, where `data` is a
        1D FCSData object or 1Dnumpy array, and `s` is a float. Statistical
        functions from numpy, scipy, or fc.stats are valid options.
    statistic_params : dict, optional
        Additional parameters to pass to `statistic_func`.
    selection_func : function, optional
        Function to use for bead population selection. Must have the
        following signature: ``m = selection_func(data_list,
        **selection_params)``, where `data_list` is a list of FCSData
        objects, each one cotaining the events of one population, and `m`
        is a boolean array indicating whether the population has been
        selected (True) or discarded (False). If None, don't use a
        population selection procedure.
    selection_params : dict, optional
        Parameters to pass to the population selection method.

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
       `mef_values`), the whole subpopulation is also discarded.
    4. The measured fluorescence of each subpopulation is compared with
       the known MEF values in `mef_values`, and a standard curve function
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

    # mef_channels and mef_values should be iterables.
    if hasattr(mef_channels, '__iter__'):
        mef_channel_all = list(mef_channels)
        mef_values_all = np.array(mef_values).copy()
    else:
        mef_channel_all = [mef_channels]
        mef_values_all = np.array([mef_values])

    ###
    # 1. Clustering
    ###
    # If clustering channels not specified, use channels in mef_channels
    if clustering_channels is None:
        clustering_channels = mef_channels

    # Get number of clusters from number of specified MEF values
    n_clusters = mef_values_all.shape[1]

    # Run clustering function
    labels = clustering_func(data_beads[:, clustering_channels],
                             n_clusters,
                             **clustering_params)

    # Separate events corresponding to each cluster
    labels_all = np.array(list(set(labels)))
    data_clustered = [data_beads[labels == i] for i in labels_all]

    # Sort clusters based on distance to the origin
    cluster_dist = [np.sum((np.mean(di[:,clustering_channels], axis=0))**2)
                    for di in data_clustered]
    cluster_sorted_ind = np.argsort(cluster_dist)
    data_clustered = [data_clustered[i] for i in cluster_sorted_ind]

    # Print information
    if verbose:
        print("Step 1: Clustering")
        print("  Number of populations to find: {}".format(n_clusters))
        # Calculate percentage of each cluster
        data_count = np.array([di.shape[0] for di in data_clustered])
        data_perc = data_count * 100.0 / data_count.sum()
        print("  Percentage of events in each population:")
        print("    " + str(data_perc))

    # Plot
    if plot:
        if plot_dir is not None:
            savefig = '{}/clustering_{}.png'.format(plot_dir, plot_filename)
        else:
            savefig = None

        # If used one channel for clustering, make histogram
        if len(clustering_channels) == 1:
            plt.figure(figsize=(8,4))
            fc.plot.hist1d(data_clustered,
                           channel=clustering_channels[0],
                           div=4,
                           alpha=0.75,
                           savefig=savefig)

        # If used two channels for clustering, make 2D scatter plot
        elif len(clustering_channels) == 2:
            plt.figure(figsize=(6,4))
            fc.plot.scatter2d(data_clustered,
                              channels=clustering_channels,
                              savefig=savefig)

        # If used three channels or more for clustering, make 3D scatter plot
        # with the first three.
        elif len(clustering_channels) >= 3:
            plt.figure(figsize=(8,6))
            fc.plot.scatter3d_and_projections(data_clustered,
                                              channels=clustering_channels[:3],
                                              savefig=savefig)

        if plot_dir is not None:
            plt.close()

    # Initialize lists to acumulate results
    sc_all = []
    if full_output:
        stats_values_all = []
        selected_channel_all = []
        selected_mef_all = []
        sc_beads_all = []
        sc_params_all =[]

    # Iterate through each mef channel
    for mef_channel, mef_values_channel in zip(mef_channel_all, mef_values_all):
        ###
        # 2. Calculate statistics in each subpopulation.
        ###

        # Calculate statistics
        stats_values = np.array(
            [statistic_func(di[:,mef_channel], **statistic_params)
             for di in data_clustered])

        # Accumulate results
        if full_output:
            stats_values_all.append(stats_values)

        # Print information
        if verbose:
            print("({}) Step 2: Population Statistic".format(mef_channel))
            print("  Fluorescence per population (Channel Units):")
            print("    " + str(stats_values))

        # Plot
        if plot:
            # Get colors for populations
            colors = [fc.plot.cmap_default(i)
                      for i in np.linspace(0, 1, n_clusters)]
            # Plot histograms
            plt.figure(figsize=(8,4))
            fc.plot.hist1d(data_clustered,
                           channel=mef_channel,
                           div=4,
                           alpha=0.75)
            # Plot histograms and populations
            for c, i in zip(colors, cluster_sorted_ind):
                # Peak values
                p = stats_values[i]
                ylim = plt.ylim()
                plt.plot([p, p], [ylim[0], ylim[1]], color=c)
                plt.ylim(ylim)
            # Save and close
            if plot_dir is not None:
                plt.tight_layout()
                plt.savefig('{}/hist_{}_{}.png'.format(plot_dir,
                                                       mef_channel,
                                                       plot_filename),
                            dpi=fc.plot.savefig_dpi)
                plt.close()

        ###
        # 3. Select populations to be used for fitting
        ###

        # Select populations based on selection_func
        if selection_func is not None:
            m = selection_func([di[:, mef_channel]
                                for di in data_clustered])
        else:
            m = np.ones(n_clusters, dtype=bool)

        # Discard values specified as nan in mef_values_channel
        m = np.logical_and(m, ~np.isnan(mef_values_channel))

        # Get selected channel and mef values
        selected_channel = stats_values[m]
        selected_mef = mef_values_channel[m]

        # Accumulate results
        if full_output:
            selected_channel_all.append(selected_channel)
            selected_mef_all.append(selected_mef)

        # Print information
        if verbose:
            print("({}) Step 3: Population Selection".format(mef_channel))
            print("  {} populations retained.".format(len(selected_channel)))
            print("  Fluorescence of selected populations (Channel Units):")
            print("    " + str(selected_channel))
            print("  Fluorescence of selected populations (MEF Units):")
            print("    " + str(selected_mef))

        ###
        # 4. Get standard curve
        ###

        # Fit
        sc, sc_beads, sc_params = fit_standard_curve(
            selected_channel,
            selected_mef)
        # Accumulate results
        sc_all.append(sc)
        if full_output:
            sc_beads_all.append(sc_beads)
            sc_params_all.append(sc_params)

        # Print information
        if verbose:
            print("({}) Step 4: Standard Curve Fitting".format(mef_channel))
            print("  Parameters of bead fluorescence model:")
            print("    " + str(sc_params))
        # Plot
        if plot:
            # Get channel range
            min_fl = data_clustered[0].domain(mef_channel)[0]
            max_fl = data_clustered[0].domain(mef_channel)[-1]
            # Plot standard curve
            plt.figure(figsize=(6,4))
            plot_standard_curve(selected_channel,
                                selected_mef,
                                sc_beads,
                                sc,
                                xlim=(min_fl, max_fl))
            plt.xlabel('{} (Channel Units)'.format(mef_channel))
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

    if full_output:
        # Clustering results
        clustering_res = {}
        clustering_res['labels'] = labels
        # Population stats results
        statistic_res = {}
        statistic_res['values'] = stats_values_all
        # Population selection results
        selection_res = {}
        selection_res['channel'] = selected_channel_all
        selection_res['mef'] = selected_mef_all
        # Fitting results
        fitting_res = {}
        fitting_res['sc'] = sc_all
        fitting_res['sc_beads'] = sc_beads_all
        fitting_res['sc_params'] = sc_params_all

        # Make namedtuple
        fields = ['transform_fxn',
                  'clustering',
                  'statistic',
                  'selection',
                  'fitting']
        MEFOutput = collections.namedtuple('MEFOutput', fields, verbose=False)
        out = MEFOutput(transform_fxn=transform_fxn,
                        clustering=clustering_res,
                        statistic=statistic_res,
                        selection=selection_res,
                        fitting=fitting_res)
        return out
    else:
        return transform_fxn
