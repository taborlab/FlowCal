"""
Functions for gating flow cytometry data.

All gate functions are of the following form:

    gated_data = gate(data, channels, parameters)
    (gated_data, mask, contour, ...) = gate(data, channels, parameters,
                                            full_output=True)

where `data` is a NxD FCSData object or numpy array describing N cytometry
events observing D data dimensions, `channels` specifies the channels in
which to perform gating, and `parameters` are gate-specific parameters.
`gated_data` is the gated result, as an FCSData object or numpy array,
`mask` is a bool array specifying the gate mask, and `contour` is an
optional list of 2D numpy arrays containing the x-y coordinates of the
contour surrounding the gated region (useful for plotting).

"""

import numpy as np
import scipy.ndimage.filters
import matplotlib._cntr         # matplotlib contour, implemented in C
import collections

###
# namedtuple Gate Function Output Classes
###

StartEndGateOutput = collections.namedtuple(
    'StartEndGateOutput',
    ['gated_data', 'mask'])
HighLowGateOutput = collections.namedtuple(
    'HighLowGateOutput',
    ['gated_data', 'mask'])
EllipseGateOutput = collections.namedtuple(
    'EllipseGateOutput',
    ['gated_data', 'mask', 'contour'])
Density2dGateOutput = collections.namedtuple(
    'Density2dGateOutput',
    ['gated_data', 'mask', 'contour'])

###
# Gate Functions
###

def start_end(data, num_start = 250, num_end = 100, full_output=False):
    """
    Gate out first and last events.

    Parameters
    ----------
    data : FCSData or numpy array
        NxD flow cytometry data where N is the number of events and D is
        the number of parameters (aka channels).
    num_start, num_end : int
        Number of events to gate out from beginning and end of `data`.
    full_output : bool
        Flag specifying to return ``namedtuple`` with additional outputs.

    Returns
    -------
    gated_data : FCSData or numpy array, if ``full_output==False``
        Gated flow cytometry data of the same format as `data`.
    namedtuple, if ``full_output==True``
        ``namedtuple`` containing the following fields in this order:
        gated_data : FCSData or numpy array
            Gated flow cytometry data of the same format as `data`.
        mask : numpy array of bool
            Boolean gate mask used to gate data such that
            `gated_data = data[mask]`.

    Raises
    ------
    ValueError
        If the number of events to discard is greater than the total
        number of events in `data`.

    """
    if data.shape[0] < (num_start + num_end):
        raise ValueError('Number of events to discard greater than total' + 
            ' number of events.')
    
    mask = np.ones(shape=data.shape[0],dtype=bool)
    mask[:num_start] = False
    mask[-num_end:] = False
    gated_data = data[mask]

    if full_output:
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
    channels : int, str, list of int, list of str
        Channels on which to perform gating. If None, use all channels.
    high, low : int, float
        High and low threshold values. If None, `high` and `low` will be
        taken from ``data.channel_info`` if available, otherwise
        ``np.Inf`` and ``-np.Inf`` will be used.
    full_output : bool
        Flag specifying to return ``namedtuple`` with additional outputs.

    Returns
    -------
    gated_data : FCSData or numpy array, if ``full_output==False``
        Gated flow cytometry data of the same format as `data`.
    namedtuple, if ``full_output==True``
        ``namedtuple`` containing the following fields in this order:
        gated_data : FCSData or numpy array
            Gated flow cytometry data of the same format as `data`.
        mask : numpy array of bool
            Boolean gate mask used to gate data such that
            `gated_data = data[mask]`.

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
        if hasattr(data_ch, 'channel_info'):
            high = [data_ch[:, channel].channel_info[0]['bin_vals'][-1] 
                    for channel in data_ch.channels]
            high = np.array(high)
        else:
            high = np.Inf
    if low is None:
        if hasattr(data_ch, 'channel_info'):
            low = [data_ch[:, channel].channel_info[0]['bin_vals'][0]
                   for channel in data_ch.channels]
            low = np.array(low)
        else:
            low = -np.Inf

    # Gate
    mask = np.all((data_ch < high) & (data_ch > low), axis = 1)
    gated_data = data[mask]

    if full_output:
        return HighLowGateOutput(gated_data=gated_data, mask=mask)
    else:
        return gated_data

def ellipse(data, channels,
            center, a, b, theta=0,
            log=False, full_output=False):
    """
    Gate that preserves events inside an ellipse-shaped region.

    Events are kept if they satisfy the following relationship:

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
    center, a, b, theta : float
        Ellipse parameters. `a` is the major axis, `b` is the minor axis.
    log : bool
        Flag specifying that log10 transformation should be applied to
        `data` before gating.
    full_output : bool
        Flag specifying to return ``namedtuple`` with additional outputs.

    Returns
    -------
    gated_data : FCSData or numpy array, if ``full_output==False``
        Gated flow cytometry data of the same format as `data`.
    namedtuple, if ``full_output==True``
        ``namedtuple`` containing the following fields in this order:
        gated_data : FCSData or numpy array
            Gated flow cytometry data of the same format as `data`.
        mask : numpy array of bool
            Boolean gate mask used to gate data such that
            `gated_data = data[mask]`.
        contour : list of 2D numpy arrays
            List of 2D numpy array(s) of x-y coordinates tracing out
            line(s) which represent the gate (useful for plotting).

    Raises
    ------
    ValueError
        If more or less than 2 channels are specified.

    """
    # Extract channels in which to gate
    assert len(channels) == 2, '2 channels should be specified.'
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

        return EllipseGateOutput(
            gated_data=data_gated, mask=mask, contour=cntr)
    else:
        return data_gated

def density2d(data, channels=[0,1],
              bins=None, gate_fraction=0.65, sigma=10.0,
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
    channels : list of int, list of str
        Two channels on which to perform gating.
    bins : int or array_like or [int, int] or [array, array]
        `bins` argument passed to `np.histogram2d`. If `None`, extracted
        from `FCSData` if possible. `bins` parameter supercedes `FCSData`
        attribute.
    gate_fraction : float
        Fraction of events to retain after gating.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel used by
        `scipy.ndimage.filters.gaussian_filter` to smooth 2D histogram
        into a density.
    full_output : bool
        Flag specifying to return ``namedtuple`` with additional outputs.

    Returns
    -------
    gated_data : FCSData or numpy array, if ``full_output==False``
        Gated flow cytometry data of the same format as `data`.
    namedtuple, if ``full_output==True``
        ``namedtuple`` containing the following fields in this order:
        gated_data : FCSData or numpy array
            Gated flow cytometry data of the same format as `data`.
        mask : numpy array of bool
            Boolean gate mask used to gate data such that
            `gated_data = data[mask]`.
        contour : list of 2D numpy arrays
            List of 2D numpy array(s) of x-y coordinates tracing out
            line(s) which represent the gate (useful for plotting).

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
        3) Use `gate_fraction` to determine number of points to retain
           (rounded up). Only points which are not implicitly gated out
           are considered.
        4) Smooth 2D histogram using a 2D Gaussian filter.
        5) Normalize smoothed histogram to obtain valid probability mass
           function (PMF).
        6) Sort bins by probability.
        7) Accumulate events (starting with events belonging to bin with
           highest probability ("densest") and proceeding to events
           belonging to bins with lowest probability) until at least the
           desired number of points is acheived. While the algorithm
           attempts to get as close to `gate_fraction` fraction of events
           as possible, more events may be retained based on how many
           events fall into each histogram bin (since entire bins are
           retained at a time, not individual events).

    """
    # Extract channels in which to gate
    assert len(channels) == 2, '2 channels should be specified.'
    data_ch = data[:,channels]
    if data_ch.ndim == 1:
        data_ch = data_ch.reshape((-1,1))

    # Check dimensions
    assert data_ch.ndim > 1, 'Data should have at least 2 dimensions'
    assert data_ch.shape[0] > 1, 'Data must have more than 1 event'

    # Extract default bins if necessary
    if bins is None and hasattr(data_ch, 'channel_info'):
        bins = np.array([data_ch.channel_info[0]['bin_edges'],
                            data_ch.channel_info[1]['bin_edges'],
                            ])

    # Make 2D histogram and get bins
    H,xe,ye = np.histogram2d(data_ch[:,0], data_ch[:,1], bins=bins)

    # Map each data point to its histogram bin.
    #
    # Note that the index returned by np.digitize is such that
    # bins[i-1] <= x < bins[i], whereas indexing the histogram will result in
    # the following: hist[i,j] = bin corresponding to
    # xedges[i] <= x < xedges[i+1] and yedges[i] <= y < yedges[i+1].
    # Therefore, we need to subtract 1 from the np.digitize result to be able
    # to index into the appropriate bin in the histogram.
    ix = np.digitize(data_ch[:,0], bins=xe) - 1
    iy = np.digitize(data_ch[:,1], bins=ye) - 1

    # In the current version of numpy, there exists a disparity in how
    # np.histogram and np.digitize treat the rightmost bin edge (np.digitize
    # is not the strict inverse of np.histogram). Specifically, np.histogram
    # treats the rightmost bin interval as fully closed (rightmost bin edge is
    # included in rightmost bin), whereas np.digitize treats all bins as
    # half-open (you can specify which side is closed and which side is open;
    # `right` parameter). The expected behavior for this gating function is to
    # mimic np.histogram behavior, so we must reconcile this disparity.
    ix[data_ch[:,0] == xe[-1]] = len(xe)-2
    iy[data_ch[:,1] == ye[-1]] = len(ye)-2

    # Create a 2D array of lists corresponding to the 2D histogram to
    # accumulate events associated with each bin.
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    Hi = np.empty_like(H, dtype=np.object)
    filler(Hi, Hi)

    x_outliers = (-1, len(xe)-1)    # Ignore (gate out) points which exist
    y_outliers = (-1, len(ye)-1)    # outside specified bins.
                                    # `np.digitize()-1` will assign points
                                    # less than `bins` to bin "-1" and points
                                    # greater than `bins` to len(bins)-1.

    N = 0   # Keep track of total number of events after implicit gating

    for i, (xi, yi) in enumerate(zip(ix, iy)):
        if (xi not in x_outliers and yi not in y_outliers):
            Hi[xi, yi].append(i)
            N = N+1

    # Determine number of points to keep. Only consider points which have not
    # been thrown out as outliers.
    n = int(np.ceil(gate_fraction*float(N)))

    # Blur 2D histogram
    sH = scipy.ndimage.filters.gaussian_filter(
        H,
        sigma=sigma,
        order=0,
        mode='constant',
        cval=0.0,
        truncate=6.0)

    # Normalize filtered histogram to make it a valid probability mass function
    D = sH / np.sum(sH)

    # Sort each (x,y) point by density
    vD = D.ravel()
    vH = H.ravel()
    sidx = np.argsort(vD)[::-1]
    svH = vH[sidx]  # linearized counts array sorted by density

    # Find minimum number of accepted (x,y) points needed to reach specified
    # number of data points
    csvH = np.cumsum(svH)
    Nidx = np.nonzero(csvH >= n)[0][0]    # we want to include this index

    # Get indices of events to keep
    vHi = Hi.ravel()
    mask = vHi[sidx[:(Nidx+1)]]
    mask = np.array([item for sublist in mask for item in sublist])
    mask = np.sort(mask)
    gated_data = data[mask]

    if full_output:
        # Use matplotlib contour plotter (implemented in C) to generate contour(s)
        # at the probability associated with the last accepted point.
        x,y = np.meshgrid(xe[:-1], ye[:-1], indexing = 'ij')
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
