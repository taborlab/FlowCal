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

import FlowCal.plot
import FlowCal.transform
import FlowCal.stats

# Use default colors from palettable if available
try:
    import palettable
except ImportError, e:
    standard_curve_colors = ['b', 'g', 'r']
else:
    standard_curve_colors = \
        palettable.colorbrewer.qualitative.Paired_12.mpl_colors[1::2]

def clustering_gmm(data,
                   n_clusters,
                   tol=1e-7,
                   min_covar=5e-5,
                   scale='linear'):
    """
    Find clusters in an array using Gaussian Mixture Models (GMM).

    Parameters
    ----------
    data : array_like
        NxD array to cluster.
    n_clusters : int
        Number of clusters to find.
    tol : float, optional
        Tolerance for convergence of GMM method. Passed directly to
        ``scikit-learn``'s GMM.
    min_covar : float, optional
        Minimum covariance. Passed directly to ``scikit-learn``'s GMM.
    scale : str, optional
        Rescaling applied to `data` before performing clustering. Can be
        either ``linear``, ``log``, or ``logicle``.

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
    library, with full covariance matrices for each cluster and a fixed,
    uniform set of weights. This means that `clustering_gmm` implicitly
    assumes that all bead subpopulations have roughly the same number of
    events. For more information, consult ``scikit-learn``'s documentation.

    """
    # Check that ``scale`` is supported
    if scale not in ['linear', 'log', 'logicle']:
        raise ValueError("scale {} not supported".format(scale))

    # Rescale if necessary
    if scale=='log':
        # Copy events
        data = data.copy()
        # Logarithm of zero and negatives is undefined. Therefore, saturate
        # any non-positives to a small positive value.
        # The machine epsilon `eps` is the smallest number such that
        # `1.0 + eps != eps`. For a 64-bit floating point, `eps ~= 1e-15`.
        data[data < 1e-15] = 1e-15
        # Rescale
        data = np.log10(data)
    elif scale=='logicle':
        # Copy events
        data = data.copy()
        # Rescale
        # Use the logicle transform class in the plot module, and transform
        # data one channel at a time.
        for i in range(data.shape[1]):
            # We need a transformation from "data value" to "display scale"
            # units. To do so, we use an inverse logicle transformation.
            t = FlowCal.plot._LogicleTransform(data=data, channel=i).inverted()
            data[:,i] = t.transform_non_affine(data[:,i])

    ###
    # Parameter initialization
    ###
    weights = np.tile(1.0 / n_clusters, n_clusters)
    means = []
    covars = []

    # Get distance and sort based on it
    dist = np.sum(data**2., axis=1)
    sorted_idx = np.argsort(dist)

    # Expected number of elements per cluster
    n_per_cluster = data.shape[0]/float(n_clusters)

    # Get means and covariances per cluster
    # We will just use a fraction of ``1 - discard_frac`` of the data.
    # Data at the edges that actually corresponds to another cluster can
    # really mess up the final result.
    discard_frac = 0.5
    for i in range(n_clusters):
        il = int((i + discard_frac/2)*n_per_cluster)
        ih = int((i + 1 - discard_frac/2)*n_per_cluster)
        sorted_idx_cluster = sorted_idx[il:ih]
        data_cluster = data[sorted_idx_cluster]
        means.append(np.mean(data_cluster, axis=0))
        if data.shape[1] == 1:
            covars.append(np.cov(data_cluster.T).reshape(1,1))
        else:
            covars.append(np.cov(data_cluster.T))
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

def selection_std(populations,
                  low=None,
                  high=None,
                  n_std_low=2.5,
                  n_std_high=2.5,
                  scale='linear'):
    """
    Select populations if most of their elements are between two values.

    This function selects populations from `populations` if their means are
    more than `n_std_low` standard deviations greater than `low` and
    `n_std_high` standard deviations lower than `high`.

    Parameters
    ----------
    populations : list of 1D arrays or 1-channel FCSData objects
        Populations to select or discard.
    low, high : int or float
        Low and high thresholds. Required if the elements in `populations`
        are numpy arrays. If not specified, and the elements in
        `populations` are FCSData objects, use 1.5% and 98.5% of the range
        in ``populations[0].range``.
    n_std_low, n_std_high : float, optional
        Number of standard deviations from `low` and `high`, respectively,
        that a population's mean has to be closer than to be discarded.
    scale : str, optional
        Rescaling applied to `populations` before calculating means and
        standard deviations. Can be either ``linear``, ``log``, or
        ``logicle``.

    Returns
    -------
    selected_mask : boolean array
        Flags indicating whether a population has been selected.

    """
    # Generate scaling functions
    if scale == 'linear':
        # Identity function
        sf = lambda x: x
    elif scale == 'log':
        sf = np.log10
    elif scale == 'logicle':
        # We need a transformation from "data value" to "display scale"
        # units. To do so, we use an inverse logicle transformation.
        t = FlowCal.plot._LogicleTransform(data=populations[0],
                                           channel=0).inverted()
        sf = lambda x: t.transform_non_affine(x, mask_out_of_range=False)
    else:
        raise ValueError("scale {} not supported".format(scale))

    # If thresholds were provided, apply scaling function. Else, obtain and
    # rescale thresholds from range.
    if low is None:
        if hasattr(populations[0], 'hist_bins'):
            # Obtain default thresholds from range
            r = populations[0].range(channels=0)
            # If using log scale and the lower limit is non-positive, change to
            # a very small positive number.
            # The machine epsilon `eps` is the smallest number such that
            # `1.0 + eps != eps`. For a 64-bit floating point, `eps ~= 1e-15`.
            if scale == 'log' and r[0] <= 0:
                r[0] = 1e-15
            low = sf(r[0]) + 0.015*(sf(r[1]) - sf(r[0]))
        else:
            raise TypeError("argument 'low' not specified")
    else:
        low = sf(low)
    if high is None:
        if hasattr(populations[0], 'hist_bins'):
            # Obtain default thresholds from range
            r = populations[0].range(channels=0)
            # If using log scale and the lower limit is non-positive, change to
            # a very small positive number.
            # The machine epsilon `eps` is the smallest number such that
            # `1.0 + eps != eps`. For a 64-bit floating point, `eps ~= 1e-15`.
            if scale == 'log' and r[0] <= 0:
                r[0] = 1e-15
            high = sf(r[0]) + 0.985*(sf(r[1]) - sf(r[0]))
        else:
            raise TypeError("argument 'high' not specified")
    else:
        high = sf(high)

    # Copy events
    for i in range(len(populations)):
        populations[i] = populations[i].copy()

    # For log scaling, logarithm of zero and negatives is undefined. Therefore,
    # saturate any non-positives to a small positive value.
    # The machine epsilon `eps` is the smallest number such that
    # `1.0 + eps != eps`. For a 64-bit floating point, `eps ~= 1e-15`.
    if scale == 'log':
        for p in populations:
            p[p < 1e-15] = 1e-15

    # Rescale events
    for i in range(len(populations)):
        populations[i] = sf(populations[i])

    # Calculate means and standard deviations
    pop_mean = np.array([FlowCal.stats.mean(p) for p in populations])
    pop_std = np.array([FlowCal.stats.std(p) for p in populations])

    # Some populations, especially the highest ones when they are near
    # saturation, tend to aggregate mostly on one bin and give a standard
    # deviation of almost zero. This is an effect of the finite bin resolution
    # and probably gives a bad estimate of the standard deviation. We choose
    # to be conservative and overestimate the standard deviation in these
    # cases. Therefore, we set the minimum standard deviation to 0.005.
    min_std = 0.005
    pop_std[pop_std < min_std] = min_std

    # Return populations that don't cross either threshold
    selected_mask = np.logical_and(
        (pop_mean - n_std_low*pop_std) > low,
        (pop_mean + n_std_high*pop_std) < high)
    return selected_mask

def fit_beads_autofluorescence(fl_rfi, fl_mef):
    """
    Fit a standard curve using a beads model with autofluorescence.

    Parameters
    ----------
    fl_rfi : array
        Fluorescence values of bead populations in units of Relative
        Fluorescence Intensity (RFI).
    fl_mef : array
        Fluorescence values of bead populations in MEF units.

    Returns
    -------
    std_crv : function
        Standard curve that transforms fluorescence values from RFI to MEF
        units. This function has the signature ``y = std_crv(x)``, where
        `x` is some fluorescence value in RFI and `y` is the same
        fluorescence expressed in MEF units.
    beads_model : function
        Fluorescence model of calibration beads. This function has the
        signature ``y = beads_model(x)``, where `x` is the fluorescence of
        some bead population in RFI units and `y` is the same fluorescence
        expressed in MEF units, without autofluorescence.
    beads_params : array
        Fitted parameters of the bead fluorescence model: ``[m, b,
        fl_mef_auto]``.
    beads_model_str : str
        String representation of the beads model used.
    beads_params_names : list of str
        Names of the parameters in a list, in the same order as they are
        given in `beads_params`.

    Notes
    -----
    The following model is used to describe bead fluorescence::

        m*log(fl_mef[i]) + b = log(fl_mef_auto + fl_mef[i])

    where ``fl_mef[i]`` is the fluorescence of bead subpopulation ``i`` in
    RFI units and ``fl_mef[i]`` is the corresponding fluorescence in MEF
    units. The model includes 3 parameters: ``m`` (slope), ``b``
    (intercept), and ``fl_mef_auto`` (bead autofluorescence). The last term
    is constrained to be greater or equal to zero.

    The bead fluorescence model is fit in log space using nonlinear least
    squares regression. In our experience, fitting in log space weights
    the residuals more evenly, whereas fitting in linear space vastly
    overvalues the brighter beads.

    A standard curve is constructed by solving for ``fl_mef``. As cell
    samples may not have the same autofluorescence as beads, the bead
    autofluorescence term (``fl_mef_auto``) is omitted from the standard
    curve; the user is expected to use an appropriate white cell sample to
    account for cellular autofluorescence if necessary. The returned
    standard curve mapping fluorescence in channel units to MEF units is
    thus of the following form::

        fl_mef = exp(m*log(fl_rfi) + b)

    """
    # Check that the input data has consistent dimensions
    if len(fl_rfi) != len(fl_mef):
        raise ValueError("fl_rfi and fl_mef have different lengths")
    # Check that we have at least three points
    if len(fl_rfi) <= 2:
        raise ValueError("standard curve model requires at least three "
            "values")
        
    # Initialize parameters
    params = np.zeros(3)
    # Initial guesses:
    # 0: slope found by putting a line through the highest two points.
    # 1: y-intercept found by putting a line through highest two points.
    # 2: bead autofluorescence initialized using the first point.
    params[0] = (np.log(fl_mef[-1]) - np.log(fl_mef[-2])) / \
                (np.log(fl_rfi[-1]) - np.log(fl_rfi[-2]))
    params[1] = np.log(fl_mef[-1]) - params[0] * np.log(fl_rfi[-1])
    params[2] = np.exp(params[0]*np.log(fl_rfi[0]) + params[1]) - fl_mef[0]

    # Error function
    def err_fun(p, x, y):
        return np.sum((np.log(y + p[2]) - ( p[0] * np.log(x) + p[1] ))**2)

    # Bead model function
    def fit_fun(p,x):
        return np.exp(p[0] * np.log(x) + p[1]) - p[2]

    # RFI-to-MEF standard curve transformation function
    # We use the equivalent form ``exp(p[1]) * x**p[0]``. Mathematically, this
    # is undefined for x < 0. In this case, we redefine the output as
    # ``- exp(p[1]) * (-x)**p[0]``. These two can be combined as follows:
    # ``np.sign(x) * np.exp(p[1]) * (np.abs(x)**p[0])``.
    def sc_fun(p,x):
        return np.sign(x) * np.exp(p[1]) * (np.abs(x)**p[0])
    
    # Fit parameters
    err_par = lambda p: err_fun(p, fl_rfi, fl_mef)
    res = minimize(err_par,
                   params,
                   bounds=((None, None), (None, None), (0, None)),
                   options = {'gtol': 1e-10, 'ftol': 1e-10})

    # Separate parameters
    beads_params = res.x

    # Beads model function
    beads_model = lambda x: fit_fun(beads_params, x)

    # Standard curve function
    std_crv = lambda x: sc_fun(beads_params, x)

    # Model string representation
    beads_model_str = 'm*log(fl_rfi) + b = log(fl_mef_auto + fl_mef)'

    # Parameter names
    beads_params_names = ['m', 'b', 'fl_mef_auto']
    
    return (std_crv,
            beads_model,
            beads_params,
            beads_model_str,
            beads_params_names)

def plot_standard_curve(fl_rfi,
                        fl_mef,
                        beads_model,
                        std_crv,
                        xscale='linear',
                        yscale='linear',
                        xlim=None,
                        ylim=(1.,1e8)):
    """
    Plot a standard curve with fluorescence of calibration beads.

    Parameters
    ----------
    fl_rfi : array_like
        Fluorescence of the calibration beads' subpopulations, in RFI
        units.
    fl_mef : array_like
        Fluorescence of the calibration beads' subpopulations, in MEF
        units.
    beads_model : function
        Fluorescence model of the calibration beads.
    std_crv : function
        The standard curve, mapping relative fluorescence (RFI) units to
        MEF units.

    Other parameters:
    -----------------
    xscale : str, optional
        Scale of the x axis, either ``linear`` or ``log``.
    yscale : str, optional
        Scale of the y axis, either ``linear`` or ``log``.
    xlim : tuple, optional
        Limits for the x axis.
    ylim : tuple, optional
        Limits for the y axis.

    """
    # Plot fluorescence of beads populations
    plt.plot(fl_rfi,
             fl_mef,
             'o',
             label='Beads',
             color=standard_curve_colors[0])

    # Generate points in x axis to plot beads model and standard curve.
    if xlim is None:
        xlim = plt.xlim()
    if xscale=='linear':
        xdata = np.linspace(xlim[0], xlim[1], 200)
    elif xscale=='log':
        xdata = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 200)

    # Plot beads model and standard curve
    plt.plot(xdata,
             beads_model(xdata),
             label='Beads model',
             color=standard_curve_colors[1])
    plt.plot(xdata,
             std_crv(xdata),
             label='Standard curve',
             color=standard_curve_colors[2])

    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.legend(loc = 'best')

def get_transform_fxn(data_beads,
                      mef_values,
                      mef_channels,
                      clustering_fxn=clustering_gmm,
                      clustering_params={},
                      clustering_channels=None,
                      statistic_fxn=FlowCal.stats.median,
                      statistic_params={},
                      selection_fxn=selection_std,
                      selection_params={},
                      fitting_fxn=fit_beads_autofluorescence,
                      fitting_params={},
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
    transform_fxn : function
        Transformation function to convert flow cytometry data from RFI
        units to MEF. This function has the same basic signature as the
        general transformation function specified in ``FlowCal.transform``.

    mef_channels : int, or str, or list, only if ``full_output==True``
        Channels on which transformation functions have been generated.
        Directly copied from the `mef_channels` argument.

    clustering : dict, only if ``full_output==True``
        Results of the clustering step, containing the following fields:

        labels : array
            Labels for each event in `data_beads`.

    statistic : dict, only if ``full_output==True``
        Results of the calculation of bead subpopulations' fluorescence,
        containing the following fields:

        values : list
            The representative fluorescence values of each
            subpopulation, for each channel in `mef_channels`.

    selection : dict, only if ``full_output==True``
        Results of the subpopulation selection step, containing the
        following fields:

        rfi : list
            The fluorescence values of each selected subpopulation in
            RFI units, for each channel in `mef_channels`.
        mef : list
            The fluorescence values of each selected subpopulation in
            MEF units, for each channel in `mef_channels`.

    fitting : dict, only if ``full_output==True``
        Results of the model fitting step, containing the following fields:

        std_crv : list
            Functions encoding the standard curves, for each channel in
            `mef_channels`.
        beads_model : list
            Functions encoding the fluorescence model of the
            calibration beads, for each channel in `mef_channels`.
        beads_params : list
            Fitted parameters of the bead fluorescence model, for each
            channel in `mef_chanels`.
        beads_model_str : list
            String representation of the bead models used, for each channel
            in `mef_channels`.
        beads_params_names : list
            Names of the parameters given in `beads_params`, for each
            channel in `mef_channels`.

    Other parameters
    ----------------
    clustering_fxn : function, optional
        Function used for clustering, or identification of subpopulations.
        Must have the following signature::

            labels = clustering_fxn(data, n_clusters, **clustering_params)

        where `data` is a NxD FCSData object or numpy array, `n_clusters`
        is the expected number of bead subpopulations, and `labels` is a 1D
        numpy array of length N, assigning each event in `data` to one
        subpopulation.

    clustering_params : dict, optional
        Additional keyword parameters to pass to `clustering_fxn`.

    clustering_channels : list, optional
        Channels used for clustering. If not specified, use `mef_channels`.
        If more than three channels are specified and `plot` is True, only
        a 3D scatter plot will be produced using the first three channels.

    statistic_fxn : function, optional
        Function used to calculate the representative fluorescence of each
        subpopulation. Must have the following signature::

            s = statistic_fxn(data, **statistic_params)

        where `data` is a 1D FCSData object or numpy array, and `s` is a
        float. Statistical functions from numpy, scipy, or FlowCal.stats
        are valid options.

    statistic_params : dict, optional
        Additional keyword parameters to pass to `statistic_fxn`.

    selection_fxn : function, optional
        Function to use for bead population selection. Must have the
        following signature::

            selected_mask = selection_fxn(data_list, **selection_params)

        where `data_list` is a list of FCSData objects, each one cotaining
        the events of one population, and `selected_mask` is a boolean
        array indicating whether the population has been selected (True) or
        discarded (False). If None, don't use a population selection
        procedure.

    selection_params : dict, optional
        Additional keyword parameters to pass to `selection_fxn`.

    fitting_fxn : function, optional
        Function used to fit the beads fluorescence model and obtain a
        standard curve. Must have the following signature::

            std_crv, beads_model, beads_params, \\
            beads_model_str, beads_params_names = fitting_fxn(
                fl_rfi, fl_mef, **fitting_params)

        where `std_crv` is a function implementing the standard curve,
        `beads_model` is a function implementing the beads fluorescence
        model, `beads_params` is an array containing the fitted parameters
        of the beads model, `beads_model_str` is a string representation
        of the beads model used, `beads_params_names` is a list with the
        parameter names in the same order as they are given in
        `beads_params`, and `fl_rfi` and `fl_mef` are the fluorescence
        values of the beads in RFI units and MEF units, respectively.
        Note that the standard curve and the fitted beads model are not
        necessarily the same.

    fitting_params : dict, optional
        Additional keyword parameters to pass to `fitting_fxn`.

    Notes
    -----
    The steps involved in generating the MEF transformation function are:

    1. The individual subpopulations of beads are first identified using a
       clustering method of choice.
    2. The fluorescence of each subpopulation is calculated, for each
       channel in `mef_channels`.
    3. Some subpopulations are then discarded if they are close to either
       the minimum or the maximum channel range limits. In addition, if the
       MEF value of some subpopulation is unknown (represented as a
       ``np.nan`` in `mef_values`), the whole subpopulation is also
        discarded.
    4. The measured fluorescence of each subpopulation is compared with
       the known MEF values in `mef_values`, and a standard curve function
       is generated using the appropriate MEF model.

    At the end, a transformation function is generated using the calculated
    standard curves, `mef_channels`, and ``FlowCal.transform.to_mef()``.

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
        mef_channels = list(mef_channels)
        mef_values = np.array(mef_values).copy()
    else:
        mef_channels = [mef_channels]
        mef_values = np.array([mef_values]).copy()

    ###
    # 1. Clustering
    ###

    # If clustering channels not specified, use channels in mef_channels
    if clustering_channels is None:
        clustering_channels = mef_channels

    # Get number of clusters from number of specified MEF values
    n_clusters = mef_values.shape[1]

    # Copy clustering_params to avoid modifying the original object
    clustering_params = clustering_params.copy()
    # If scale has not been specified in clustering_params, select clustering
    # scale automatically.
    if 'scale' not in clustering_params:
        # If at least one channel has been acquired in linear scale, use
        # logicle. Else, use log.
        clustering_scale = 'log'
        for ch in clustering_channels:
            # Extract amplification type
            # The amplification type for each channel is a tuple of two numbers,
            # in which the first number indicates the number of decades covered
            # by a logarithmic amplifier, and the second indicates the linear
            # value corresponding to the channel value zero. If the first value
            # is zero, the amplifier used is linear, otherwise it is
            # logarithmic.
            if data_beads.amplification_type(ch)[0] == 0:
                clustering_scale = 'logicle'
                break
        clustering_params['scale'] = clustering_scale

    # Run clustering function
    labels = clustering_fxn(data_beads[:, clustering_channels],
                            n_clusters,
                            **clustering_params)

    # Separate events corresponding to each cluster
    unique_labels = np.array(list(set(labels)))
    populations = [data_beads[labels == i] for i in unique_labels]

    # Sort populations based on distance to the origin
    population_dist = [np.sum((np.mean(population[:,clustering_channels],
                                       axis=0))**2)
                       for population in populations]
    population_sorted_idx = np.argsort(population_dist)
    populations = [populations[i] for i in population_sorted_idx]

    # Print information
    if verbose:
        # Calculate and display percentage of events on each population
        population_count = np.array([population.shape[0]
                                     for population in populations])
        population_perc = population_count * 100.0 / population_count.sum()

        # Print information
        print("Step 1: Clustering")
        print("  Number of populations to find: {}".format(n_clusters))
        print("  Percentage of events in each population:")
        print("    " + str(population_perc))

    # Plot
    if plot:
        if plot_dir is not None:
            savefig = '{}/clustering_{}.png'.format(plot_dir, plot_filename)
        else:
            savefig = None

        # Extract amplification type of clustering channels
        # The amplification type for each channel is a tuple of two numbers,
        # in which the first number indicates the number of decades covered by
        # a logarithmic amplifier, and the second indicates the linear value
        # corresponding to the channel value zero. If the first value is zero,
        # the amplifier used is linear, otherwise it is logarithmic.
        clustering_amp_type = \
            populations[0].amplification_type(clustering_channels)

        # If used one channel for clustering, make histogram
        if len(clustering_channels) == 1:
            plt.figure(figsize=(8,4))
            FlowCal.plot.hist1d(
                populations,
                channel=clustering_channels[0],
                xscale='log' if clustering_amp_type[0][0] else 'logicle',
                bins=256,
                alpha=0.75,
                savefig=savefig)

        # If used two channels for clustering, make 2D scatter plot
        elif len(clustering_channels) == 2:
            plt.figure(figsize=(6,4))
            FlowCal.plot.scatter2d(
                populations,
                channels=clustering_channels,
                xscale='log' if clustering_amp_type[0][0] else 'logicle',
                yscale='log' if clustering_amp_type[1][0] else 'logicle',
                savefig=savefig)

        # If used three channels or more for clustering, make 3D scatter plot
        # with the first three.
        elif len(clustering_channels) >= 3:
            plt.figure(figsize=(8,6))
            FlowCal.plot.scatter3d_and_projections(
                populations,
                channels=clustering_channels[:3],
                xscale='log' if clustering_amp_type[0][0] else 'logicle',
                yscale='log' if clustering_amp_type[1][0] else 'logicle',
                zscale='log' if clustering_amp_type[2][0] else 'logicle',
                savefig=savefig)

        if plot_dir is not None:
            plt.close()

    # Initialize lists to acumulate results
    std_crv_res = []
    if full_output:
        stats_values_res = []
        selected_rfi_res = []
        selected_mef_res = []
        beads_model_res = []
        beads_params_res =[]
        beads_model_str_res =[]
        beads_params_names_res =[]

    # Iterate through each mef channel
    for mef_channel, mef_values_channel in zip(mef_channels, mef_values):

        populations_channel = [population[:, mef_channel]
                               for population in populations]

        ###
        # 2. Calculate statistics in each subpopulation.
        ###

        # Calculate statistics
        stats_values = [statistic_fxn(population, **statistic_params)
                        for population in populations_channel]
        stats_values = np.array(stats_values)

        # Accumulate results
        if full_output:
            stats_values_res.append(stats_values)

        # Print information
        if verbose:
            print("({}) Step 2: Population Statistic".format(mef_channel))
            print("  Fluorescence per population (Channel Units):")
            print("    " + str(stats_values))

        ###
        # 3. Select populations to be used for fitting
        ###

        # Copy selection_params to avoid modifying the original object
        selection_params_channel = selection_params.copy()
        # If scale has not been specified in selection_params_channel, choose
        # scale automatically.
        if 'scale' not in selection_params_channel:
            # If amplification type is logarithmic, use log scaling. Otherwise,
            # use logicle scaling.
            # The amplification type for each channel is a tuple of two numbers,
            # in which the first number indicates the number of decades covered
            # by a logarithmic amplifier, and the second indicates the linear
            # value corresponding to the channel value zero. If the first value
            # is zero, the amplifier used is linear, otherwise it is
            # logarithmic.
            mef_amp_type = populations[0].amplification_type(mef_channel)
            selection_params_channel['scale'] = 'log' if mef_amp_type[0] else 'logicle'

        # Select populations based on selection_fxn
        if selection_fxn is not None:
            selected_mask = selection_fxn(
                [population for population in populations_channel],
                **selection_params_channel)
        else:
            selected_mask = np.ones(n_clusters, dtype=bool)

        # Discard values specified as nan in mef_values_channel
        selected_mask = np.logical_and(selected_mask,
                                       ~np.isnan(mef_values_channel))

        # Get selected rfi and mef values
        selected_rfi = stats_values[selected_mask]
        selected_mef = mef_values_channel[selected_mask]

        # Accumulate results
        if full_output:
            selected_rfi_res.append(selected_rfi)
            selected_mef_res.append(selected_mef)

        # Print information
        if verbose:
            print("({}) Step 3: Population Selection".format(mef_channel))
            print("  {} populations selected.".format(len(selected_rfi)))
            print("  Fluorescence of selected populations (a.u.):")
            print("    " + str(selected_rfi))
            print("  Fluorescence of selected populations (MEF):")
            print("    " + str(selected_mef))

        # Plot
        if plot:
            # Get colors for each population. Colors are taken from the default
            # colormap in FlowCal.plot, if the population has been selected.
            # Otherwise, the population is displayed in gray.
            color_levels = np.linspace(0, 1, n_clusters)
            colors = [FlowCal.plot.cmap_default(level)
                          if selected else (0.6, 0.6, 0.6)
                      for selected, level in zip(selected_mask, color_levels)]

            # Decide scale to plot based on amplification type
            mef_amp_type = populations[0].amplification_type(mef_channel)
            # Plot histograms
            plt.figure(figsize=(8,4))
            FlowCal.plot.hist1d(populations,
                                channel=mef_channel,
                                xscale='log' if mef_amp_type[0] else 'logicle',
                                bins=256,
                                alpha=0.75,
                                facecolor=colors)

            # Plot a vertical line for each population, with an x coordinate
            # corresponding to their statistic value.
            ylim = plt.ylim()
            for val, color in zip(stats_values, colors):
                plt.plot([val, val], [ylim[0], ylim[1]], color=color)
            plt.ylim(ylim)

            # Save and close
            if plot_dir is not None:
                plt.tight_layout()
                plt.savefig('{}/populations_{}_{}.png'.format(plot_dir,
                                                              mef_channel,
                                                              plot_filename),
                            dpi=FlowCal.plot.savefig_dpi)
                plt.close()

        ###
        # 4. Get standard curve
        ###

        # Fit
        fitting_output = fitting_fxn(selected_rfi,
                                     selected_mef,
                                     **fitting_params)
        std_crv = fitting_output[0]
        beads_model = fitting_output[1]
        beads_params = fitting_output[2]
        beads_model_str = fitting_output[3]
        beads_params_names = fitting_output[4]
        # Accumulate results
        std_crv_res.append(std_crv)
        if full_output:
            beads_model_res.append(beads_model)
            beads_params_res.append(beads_params)
            beads_model_str_res.append(beads_model_str)
            beads_params_names_res.append(beads_params_names)

        # Print information
        if verbose:
            print("({}) Step 4: Standard Curve Fitting".format(mef_channel))
            print("  Parameters of bead fluorescence model:")
            print("    " + str(beads_params))

        # Plot
        if plot:
            # Get channel range
            xlim = populations[0].range(channels=mef_channel)
            # The plot will be made in log scale. If the lower limit of the
            # range is zero or less, replace by one or some lower value, such
            # that the range covers at least five decades.
            if xlim[0] <= 0:
                xlim[0] = min(1., xlim[1]/1e5)

            # Plot standard curve
            plt.figure(figsize=(6,4))
            plot_standard_curve(selected_rfi,
                                selected_mef,
                                beads_model,
                                std_crv,
                                xscale='log',
                                yscale='log',
                                xlim=xlim)
            plt.xlabel('{} (a.u.)'.format(mef_channel))
            plt.ylabel('{} (MEF)'.format(mef_channel))

            # Save if required
            if plot_dir is not None:
                plt.tight_layout()
                plt.savefig('{}/std_crv_{}_{}.png'.format(plot_dir,
                                                          mef_channel,
                                                          plot_filename),
                            dpi=FlowCal.plot.savefig_dpi)
                plt.close()

    # Make output transformation function
    transform_fxn = functools.partial(FlowCal.transform.to_mef,
                                      sc_list=std_crv_res,
                                      sc_channels=mef_channels)

    if verbose:
        np.set_printoptions(precision=prev_precision)

    if full_output:
        # Clustering results
        clustering_res = {}
        clustering_res['labels'] = labels
        # Population stats results
        statistic_res = {}
        statistic_res['values'] = stats_values_res
        # Population selection results
        selection_res = {}
        selection_res['rfi'] = selected_rfi_res
        selection_res['mef'] = selected_mef_res
        # Fitting results
        fitting_res = {}
        fitting_res['std_crv'] = std_crv_res
        fitting_res['beads_model'] = beads_model_res
        fitting_res['beads_params'] = beads_params_res
        fitting_res['beads_model_str'] = beads_model_str_res
        fitting_res['beads_params_names'] = beads_params_names_res

        # Make namedtuple
        fields = ['mef_channels',
                  'transform_fxn',
                  'clustering',
                  'statistic',
                  'selection',
                  'fitting']
        MEFOutput = collections.namedtuple('MEFOutput', fields, verbose=False)
        out = MEFOutput(mef_channels=mef_channels,
                        transform_fxn=transform_fxn,
                        clustering=clustering_res,
                        statistic=statistic_res,
                        selection=selection_res,
                        fitting=fitting_res)
        return out
    else:
        return transform_fxn
