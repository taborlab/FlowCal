"""
Functions for gating flow cytometry data.

All gate functions are of the following form::

    gated_data = gate(data, channels, *args, **kwargs)

    (gated_data, mask, contour, ...) = gate(data, channels, *args,
                                            **kwargs, full_output=True)

where `data` is a NxD FCSData object or numpy array describing N cytometry
events with D channels, `channels` specifies the channels in which to
perform gating, and `args` and `kwargs` are gate-specific parameters.
`gated_data` is the gated result, as an FCSData object or numpy array,
`mask` is a bool array specifying the gate mask, and `contour` is an
optional list of 2D numpy arrays containing the x-y coordinates of the
contour surrounding the gated region, which can be used when plotting a 2D
density diagram or scatter plot.

"""

import numpy as np
import scipy.ndimage.filters
import matplotlib._cntr         # matplotlib contour, implemented in C
import collections

###
# Gate Functions
###

def start_end(data, num_start=250, num_end=100, full_output=False):
    """
    Gate out first and last events.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    num_start, num_end : int, optional
        Number of events to gate out from beginning and end of `data`.
        Ignored if less than 0.
    full_output : bool, optional
        Flag specifying to return additional outputs. If true, the outputs
        are given as a namedtuple.

    Returns
    -------
    gated_data : FCSData or numpy array
        Gated flow cytometry data of the same format as `data`.
    mask : numpy array of bool, only if ``full_output==True``
        Boolean gate mask used to gate data such that ``gated_data =
        data[mask]``.

    Raises
    ------
    ValueError
        If the number of events to discard is greater than the total
        number of events in `data`.

    """

    if num_start < 0:
        num_start = 0
    if num_end < 0:
        num_end = 0

    if data.shape[0] < (num_start + num_end):
        raise ValueError('Number of events to discard greater than total' + 
            ' number of events.')
    
    mask = np.ones(shape=data.shape[0],dtype=bool)
    mask[:num_start] = False
    if num_end > 0:
        # catch the edge case where `num_end=0` causes mask[-num_end:] to mask
        # off all events
        mask[-num_end:] = False
    gated_data = data[mask]

    if full_output:
        StartEndGateOutput = collections.namedtuple(
            'StartEndGateOutput',
            ['gated_data', 'mask'])
        return StartEndGateOutput(gated_data=gated_data, mask=mask)
    else:
        return gated_data

def high_low(data, channels=None, high=None, low=None, full_output=False):
    """
    Gate out high and low values across all specified channels.

    Gate out events in `data` with values in the specified channels which
    are larger than or equal to `high` or less than or equal to `low`.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : int, str, list of int, list of str, optional
        Channels on which to perform gating. If None, use all channels.
    high, low : int, float, optional
        High and low threshold values. If None, `high` and `low` will be
        taken from ``data.range`` if available, otherwise
        ``np.inf`` and ``-np.inf`` will be used.
    full_output : bool, optional
        Flag specifying to return additional outputs. If true, the outputs
        are given as a namedtuple.

    Returns
    -------
    gated_data : FCSData or numpy array
        Gated flow cytometry data of the same format as `data`.
    mask : numpy array of bool, only if ``full_output==True``
        Boolean gate mask used to gate data such that ``gated_data =
        data[mask]``.

    """
    # Extract channels in which to gate
    if channels is None:
        data_ch = data
    else:
        data_ch = data[:,channels]
        if data_ch.ndim == 1:
            data_ch = data_ch.reshape((-1,1))

    # Default values for high and low
    if high is None:
        if hasattr(data_ch, 'range'):
            high = [np.Inf if di is None else di[1] for di in data_ch.range()]
            high = np.array(high)
        else:
            high = np.Inf
    if low is None:
        if hasattr(data_ch, 'range'):
            low = [-np.Inf if di is None else di[0] for di in data_ch.range()]
            low = np.array(low)
        else:
            low = -np.Inf

    # Gate
    mask = np.all((data_ch < high) & (data_ch > low), axis = 1)
    gated_data = data[mask]

    if full_output:
        HighLowGateOutput = collections.namedtuple(
            'HighLowGateOutput',
            ['gated_data', 'mask'])
        return HighLowGateOutput(gated_data=gated_data, mask=mask)
    else:
        return gated_data

def ellipse(data, channels,
            center, a, b, theta=0,
            log=False, full_output=False):
    """
    Gate that preserves events inside an ellipse-shaped region.

    Events are kept if they satisfy the following relationship::

        (x/a)**2 + (y/b)**2 <= 1

    where `x` and `y` are the coordinates of the event list, after
    substracting `center` and rotating by -`theta`. This is mathematically
    equivalent to maintaining the events inside an ellipse with major
    axis `a`, minor axis `b`, center at `center`, and tilted by `theta`.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : list of int, list of str
        Two channels on which to perform gating.
    center, a, b, theta (optional) : float
        Ellipse parameters. `a` is the major axis, `b` is the minor axis.
    log : bool, optional
        Flag specifying that log10 transformation should be applied to
        `data` before gating.
    full_output : bool, optional
        Flag specifying to return additional outputs. If true, the outputs
        are given as a namedtuple.

    Returns
    -------
    gated_data : FCSData or numpy array
        Gated flow cytometry data of the same format as `data`.
    mask : numpy array of bool, only if ``full_output==True``
        Boolean gate mask used to gate data such that ``gated_data =
        data[mask]``.
    contour : list of 2D numpy arrays, only if ``full_output==True``
        List of 2D numpy array(s) of x-y coordinates tracing out
        the edge of the gated region.

    Raises
    ------
    ValueError
        If more or less than 2 channels are specified.

    """
    # Extract channels in which to gate
    if len(channels) != 2:
        raise ValueError('2 channels should be specified.')
    data_ch = data[:,channels].view(np.ndarray)

    # Log if necessary
    if log:
        data_ch = np.log10(data_ch)

    # Center
    center = np.array(center)
    data_centered = data_ch - center

    # Rotate
    R = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])
    data_rotated = np.dot(data_centered, R.T)

    # Generate mask
    mask = ((data_rotated[:,0]/a)**2 + (data_rotated[:,1]/b)**2 <= 1)

    # Gate
    data_gated = data[mask]

    if full_output:
        # Calculate contour
        t = np.linspace(0,1,100)*2*np.pi
        ci = np.array([a*np.cos(t), b*np.sin(t)]).T
        ci = np.dot(ci, R) + center
        if log:
            ci = 10**ci
        cntr = [ci]

        # Build output namedtuple
        EllipseGateOutput = collections.namedtuple(
            'EllipseGateOutput',
            ['gated_data', 'mask', 'contour'])
        return EllipseGateOutput(
            gated_data=data_gated, mask=mask, contour=cntr)
    else:
        return data_gated

def density2d(data,
              channels=[0,1],
              bins=1024,
              gate_fraction=0.65,
              xscale='logicle',
              yscale='logicle',
              sigma=10.0,
              full_output=False):
    """
    Gate that preserves events in the region with highest density.

    Gate out all events in `data` but those near regions of highest
    density for the two specified channels.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    channels : list of int, list of str, optional
        Two channels on which to perform gating.
    bins : int or array_like or [int, int] or [array, array], optional
        Bins used for gating:

          - If None, use ``data.hist_bins`` to obtain bin edges for both
            axes. None is not allowed if ``data.hist_bins`` is not
            available.
          - If int, `bins` specifies the number of bins to use for both
            axes. If ``data.hist_bins`` exists, it will be used to generate
            a number `bins` of bins.
          - If array_like, `bins` directly specifies the bin edges to use
            for both axes.
          - If [int, int], each element of `bins` specifies the number of
            bins for each axis. If ``data.hist_bins`` exists, use it to
            generate ``bins[0]`` and ``bins[1]`` bin edges, respectively.
          - If [array, array], each element of `bins` directly specifies
            the bin edges to use for each axis.
          - Any combination of the above, such as [int, array], [None,
            int], or [array, int]. In this case, None indicates to generate
            bin edges using ``data.hist_bins`` as above, int indicates the
            number of bins to generate, and an array directly indicates the
            bin edges. Note that None is not allowed if ``data.hist_bins``
            does not exist.
    gate_fraction : float, optional
        Fraction of events to retain after gating. Should be between 0 and
        1, inclusive.
    xscale : str, optional
        Scale of the bins generated for the x axis, either ``linear``,
        ``log``, or ``logicle``. `xscale` is ignored in `bins` is an array
        or a list of arrays.
    yscale : str, optional
        Scale of the bins generated for the y axis, either ``linear``,
        ``log``, or ``logicle``. `yscale` is ignored in `bins` is an array
        or a list of arrays.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel used by
        `scipy.ndimage.filters.gaussian_filter` to smooth 2D histogram
        into a density.
    full_output : bool, optional
        Flag specifying to return additional outputs. If true, the outputs
        are given as a namedtuple.

    Returns
    -------
    gated_data : FCSData or numpy array
        Gated flow cytometry data of the same format as `data`.
    mask : numpy array of bool, only if ``full_output==True``
        Boolean gate mask used to gate data such that ``gated_data =
        data[mask]``.
    contour : list of 2D numpy arrays, only if ``full_output==True``
        List of 2D numpy array(s) of x-y coordinates tracing out
        the edge of the gated region.

    Raises
    ------
    ValueError
        If more or less than 2 channels are specified.
    ValueError
        If `data` has less than 2 dimensions or less than 2 events.
    Exception
        If an unrecognized matplotlib Path code is encountered when
        attempting to generate contours.

    Notes
    -----
    The algorithm for gating based on density works as follows:

        1) Calculate 2D histogram of `data` in the specified channels.
        2) Map each event from `data` to its histogram bin (implicitly
           gating out any events which exist outside specified `bins`).
        3) Use `gate_fraction` to determine number of events to retain
           (rounded up). Only events which are not implicitly gated out
           are considered.
        4) Smooth 2D histogram using a 2D Gaussian filter.
        5) Normalize smoothed histogram to obtain valid probability mass
           function (PMF).
        6) Sort bins by probability.
        7) Accumulate events (starting with events belonging to bin with
           highest probability ("densest") and proceeding to events
           belonging to bins with lowest probability) until at least the
           desired number of events is achieved. While the algorithm
           attempts to get as close to `gate_fraction` fraction of events
           as possible, more events may be retained based on how many
           events fall into each histogram bin (since entire bins are
           retained at a time, not individual events).

    """

    # Extract channels in which to gate
    if len(channels) != 2:
        raise ValueError('2 channels should be specified')
    data_ch = data[:,channels]
    if data_ch.ndim == 1:
        data_ch = data_ch.reshape((-1,1))

    # Check gating fraction
    if gate_fraction < 0 or gate_fraction > 1:
        raise ValueError('gate fraction should be between 0 and 1, inclusive')

    # Check dimensions
    if data_ch.ndim < 2:
        raise ValueError('data should have at least 2 dimensions')
    if data_ch.shape[0] <= 1:
        raise ValueError('data should have more than one event')

    # Build output namedtuple if necessary
    if full_output:
        Density2dGateOutput = collections.namedtuple(
            'Density2dGateOutput',
            ['gated_data', 'mask', 'contour'])

    # If ``data_ch.hist_bins()`` exists, obtain bin edges from it if
    # necessary.
    if hasattr(data_ch, 'hist_bins') and \
            hasattr(data_ch.hist_bins, '__call__'):
        # Check whether `bins` contains information for one or two axes
        if hasattr(bins, '__iter__') and len(bins)==2:
            # `bins` contains separate information for both axes
            # If bins for the X axis is not an iterable, get bin edges from
            # ``data_ch.hist_bins()``.
            if not hasattr(bins[0], '__iter__'):
                bins[0] = data_ch.hist_bins(channels=0,
                                            nbins=bins[0],
                                            scale=xscale)
            # If bins for the Y axis is not an iterable, get bin edges from
            # ``data_ch.hist_bins()``.
            if not hasattr(bins[1], '__iter__'):
                bins[1] = data_ch.hist_bins(channels=1,
                                            nbins=bins[1],
                                            scale=yscale)
        else:
            # `bins` contains information for one axis, which will be used
            # twice.
            # If bins is not an iterable, get bin edges from
            # ``data_ch.hist_bins()``.
            if not hasattr(bins, '__iter__'):
                bins = [data_ch.hist_bins(channels=0,
                                          nbins=bins,
                                          scale=xscale),
                        data_ch.hist_bins(channels=1,
                                          nbins=bins,
                                          scale=yscale)]

    # Make 2D histogram
    H,xe,ye = np.histogram2d(data_ch[:,0], data_ch[:,1], bins=bins)

    # Map each event to its histogram bin by sorting events into a 2D array of
    # lists which mimics the histogram.
    #
    # Use np.digitize to calculate the histogram bin index for each event
    # given the histogram bin edges. Note that the index returned by
    # np.digitize is such that bins[i-1] <= x < bins[i], whereas indexing the
    # histogram will result in the following: hist[i,j] = bin corresponding to
    # xedges[i] <= x < xedges[i+1] and yedges[i] <= y < yedges[i+1].
    # Therefore, we need to subtract 1 from the np.digitize result to be able
    # to index into the appropriate bin in the histogram.
    event_indices = np.arange(data_ch.shape[0])
    x_bin_indices = np.digitize(data_ch[:,0], bins=xe) - 1
    y_bin_indices = np.digitize(data_ch[:,1], bins=ye) - 1

    # In the current version of numpy, there exists a disparity in how
    # np.histogram and np.digitize treat the rightmost bin edge (np.digitize
    # is not the strict inverse of np.histogram). Specifically, np.histogram
    # treats the rightmost bin interval as fully closed (rightmost bin edge is
    # included in rightmost bin), whereas np.digitize treats all bins as
    # half-open (you can specify which side is closed and which side is open;
    # `right` parameter). The expected behavior for this gating function is to
    # mimic np.histogram behavior, so we must reconcile this disparity.
    x_bin_indices[data_ch[:,0] == xe[-1]] = len(xe)-2
    y_bin_indices[data_ch[:,1] == ye[-1]] = len(ye)-2

    # Ignore (gate out) events which exist outside specified bins.
    # `np.digitize()-1` will assign events less than `bins` to bin "-1" and
    # events greater than `bins` to len(bins)-1.
    outlier_mask = (
        (x_bin_indices == -1) |
        (x_bin_indices == len(xe)-1) |
        (y_bin_indices == -1) |
        (y_bin_indices == len(ye)-1))

    event_indices = event_indices[~outlier_mask]
    x_bin_indices = x_bin_indices[~outlier_mask]
    y_bin_indices = y_bin_indices[~outlier_mask]

    # Create a 2D array of lists mimicking the histogram to accumulate events
    # associated with each bin.
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    H_events = np.empty_like(H, dtype=np.object)
    filler(H_events, H_events)

    for event_idx, x_bin_idx, y_bin_idx in \
            zip(event_indices, x_bin_indices, y_bin_indices):
        H_events[x_bin_idx, y_bin_idx].append(event_idx)

    # Determine number of events to keep. Only consider events which have not
    # been thrown out as outliers.
    n = int(np.ceil(gate_fraction*float(len(event_indices))))

    # n = 0 edge case (e.g. if gate_fraction = 0.0); incorrectly handled below
    if n == 0:
        mask = np.zeros(shape=data_ch.shape[0], dtype=bool)
        gated_data = data[mask]
        if full_output:
            return Density2dGateOutput(
                gated_data=gated_data, mask=mask, contour=[])
        else:
            return gated_data

    # Smooth 2D histogram
    sH = scipy.ndimage.filters.gaussian_filter(
        H,
        sigma=sigma,
        order=0,
        mode='constant',
        cval=0.0,
        truncate=6.0)

    # Normalize smoothed histogram to make it a valid probability mass function
    D = sH / np.sum(sH)

    # Sort bins by density
    vD = D.ravel()
    vH = H.ravel()
    sidx = np.argsort(vD)[::-1]
    svH = vH[sidx]  # linearized counts array sorted by density

    # Find minimum number of accepted bins needed to reach specified number
    # of events
    csvH = np.cumsum(svH)
    Nidx = np.nonzero(csvH >= n)[0][0]    # we want to include this index

    # Get indices of events to keep
    vH_events = H_events.ravel()
    accepted_indices = vH_events[sidx[:(Nidx+1)]]
    accepted_indices = np.array([item       # flatten list of lists
                                 for sublist in accepted_indices
                                 for item in sublist])
    accepted_indices = np.sort(accepted_indices)

    # Convert list of accepted indices to boolean mask array
    mask = np.zeros(shape=data.shape[0], dtype=bool)
    mask[accepted_indices] = True

    gated_data = data[mask]

    if full_output:
        # Use matplotlib contour plotter (implemented in C) to generate contour(s)
        # at the probability associated with the last accepted bin.
        xc = (xe[:-1] + xe[1:]) / 2.0   # x-axis bin centers
        yc = (ye[:-1] + ye[1:]) / 2.0   # y-axis bin centers
        x,y = np.meshgrid(xc, yc, indexing='ij')
        mpl_cntr = matplotlib._cntr.Cntr(x,y,D)
        tr = mpl_cntr.trace(vD[sidx[Nidx]])

        # trace returns a list of arrays which contain vertices and path codes
        # used in matplotlib Path objects (see http://stackoverflow.com/a/18309914
        # and the documentation for matplotlib.path.Path for more details). I'm
        # just going to make sure the path codes aren't unfamiliar and then extract
        # all of the vertices and pack them into a list of 2D contours.
        cntr = []
        num_cntrs = len(tr)/2
        for idx in xrange(num_cntrs):
            vertices = tr[idx]
            codes = tr[num_cntrs+idx]

            # I am only expecting codes 1 and 2 ('MOVETO' and 'LINETO' codes)
            if not np.all((codes==1)|(codes==2)):
                raise Exception('Contour error: unrecognized path code')

            cntr.append(vertices)

        return Density2dGateOutput(
            gated_data=gated_data, mask=mask, contour=cntr)
    else:
        return gated_data
