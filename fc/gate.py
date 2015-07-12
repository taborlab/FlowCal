#!/usr/bin/python
#
# gate.py - Module containing flow cytometry gate functions.
#
# All gate functions should be of one of the following forms:
#
#     gated = gate(data, channels, parameters)
#     gated, contour = gate(data, channels, parameters)
#
# where data is a NxD numpy array describing N cytometry events observing D
# data dimensions (channels), channels specify the channels in which to perform
# gating, parameters are gate-specific parameters, and gated is the gated 
# result. If channels is not specified, gating should be performed on all 
# channels. Contour is an optional 2D numpy array of x-y coordinates tracing 
# out a line which represents the gate (useful for plotting).
#
# Authors: John T. Sexton (john.t.sexton@rice.edu)
#          Sebastian M. Castillo-Hair (smc9@rice.edu)
# Date: 7/5/2015
#
# Requires:
#   * numpy
#   * scipy
#   * matplotlib

import numpy
import scipy.ndimage.filters
import matplotlib._cntr         # matplotlib contour, implemented in C
    
def start_end(data, num_start=250, num_end=100):
    '''Gate out num_start first and num_end last events collected.

    data      - NxD FCSData object or numpy array
    num_start - number of points to discard from the beginning of data
    num_end   - number of points to discard from the end of data

    returns  - Gated MxD FCSData object or numpy array'''
    
    if data.shape[0] < (num_start + num_end):
        raise ValueError('Number of events to discard greater than total' + 
            ' number.')
    
    mask = numpy.ones(shape=data.shape[0],dtype=bool)
    mask[:num_start] = False
    mask[-num_end:] = False
    gated_data = data[mask]
    
    return gated_data

def high_low(data, channels=None, high=(2**10)-1, low=0):
    '''Gate out high and low values across all specified dimensions.

    For every i, if any value of data[i,channels] is less or equal than low, 
    or greater or equal than high, it will not be included in the final result.

    data     - NxD FCSData object or numpy array
    channels - channels on which to perform gating
    high     - high value to discard
    low      - low value to discard

    returns  - Gated MxD FCSData object or numpy array'''
    
    # Extract channels in which to gate
    if channels is None:
        data_ch = data
    else:
        data_ch = data[:,channels]
        if data_ch.ndim == 1:
            data_ch = data_ch.reshape((-1,1))

    # Gate
    mask = numpy.all((data_ch < high) & (data_ch > low), axis=1)
    gated_data = data[mask]

    return gated_data

def density2d(data, channels = [0,1], bins = None, bins_log = False,
    sigma = 10.0, gate_fraction = 0.65):
    '''Gate that preserves the points in the region with highest density.

    First, obtain a 2D histogram and blur it using a 2D Gaussian filter. Then
    normalize the resulting blurred histogram to make it a valid probability 
    mass function. Finally, gate out all but the points in the densest region
    (points with the largest probability).

    data            - NxD FCSData object or numpy array
    channels        - channels on which to perform gating
    bins            - bins argument to numpy.histogram2d. Autogenerate if None.
    bins_log        - If bins is None, autogenerate bins in log space.
    sigma           - standard deviation for Gaussian kernel
    gate_fraction   - fraction of data points to keep

    returns         - Gated MxD FCSData object or numpy array, 
                    - list of 2D numpy arrays of (x,y) coordinates of gate 
                        contour(s)
    '''

    # Extract channels in which to gate
    assert len(channels) == 2, '2 channels should be specified.'
    data_ch = data[:,channels]
    if data_ch.ndim == 1:
        data_ch = data_ch.reshape((-1,1))

    # Check dimensions
    assert data_ch.ndim > 1, 'Data should have at least 2 dimensions'
    assert data_ch.shape[0] > 1, 'Data must have more than 1 event'

    # Generate bins if necessary
    if bins is None:
        rx = data_ch.channel_info[0]['range']
        ry = data_ch.channel_info[1]['range']
        if bins_log:
            drx = (numpy.log10(rx[1]) - numpy.log10(rx[0]))/float(rx[2] - 1)
            dry = (numpy.log10(ry[1]) - numpy.log10(ry[0]))/float(ry[2] - 1)
            bins = numpy.array([numpy.logspace(numpy.log10(rx[0]), 
                                               numpy.log10(rx[1]) + drx, 
                                               (rx[2] + 1)),
                                numpy.logspace(numpy.log10(ry[0]), 
                                               numpy.log10(ry[1]) + dry, 
                                               (ry[2] + 1)),
                                ])
        else:
            drx = (rx[1] - rx[0])/float(rx[2] - 1)
            dry = (ry[1] - ry[0])/float(ry[2] - 1)
            bins = numpy.array([
                numpy.linspace(rx[0], rx[1] + drx, (rx[2] + 1)),
                numpy.linspace(ry[0], ry[1] + dry, (ry[2] + 1)),
                ])

    # Determine number of points to keep
    n = int(numpy.ceil(gate_fraction*float(data_ch.shape[0])))

    # Make 2D histogram and get bins
    H,xe,ye = numpy.histogram2d(data_ch[:,0], data_ch[:,1], bins=bins)

    # Get lists of indices per bin
    ix = numpy.digitize(data_ch[:,0], bins=xe) - 1
    iy = numpy.digitize(data_ch[:,1], bins=ye) - 1

    filler = numpy.frompyfunc(lambda x: list(), 1, 1)
    Hi = numpy.empty_like(H, dtype=numpy.object)
    filler(Hi, Hi)
    for i, (xi, yi) in enumerate(zip(ix, iy)):
        Hi[xi, yi].append(i)

    # Blur 2D histogram
    bH = scipy.ndimage.filters.gaussian_filter(
        H,
        sigma=sigma,
        order=0,
        mode='constant',
        cval=0.0,
        truncate=6.0)

    # Normalize filtered histogram to make it a valid probability mass function
    D = bH / numpy.sum(bH)

    # Sort each (x,y) point by density
    vD = D.ravel()
    vH = H.ravel()
    sidx = numpy.argsort(vD)[::-1]
    svH = vH[sidx]  # linearized counts array sorted by density

    # Find minimum number of accepted (x,y) points needed to reach specified
    # number of data points
    csvH = numpy.cumsum(svH)
    Nidx = numpy.nonzero(csvH >= n)[0][0]    # we want to include this index

    # Get indices of events to keep
    vHi = Hi.ravel()
    mask = vHi[sidx[:(Nidx+1)]]
    mask = numpy.array([item for sublist in mask for item in sublist])
    mask = numpy.sort(mask)
    gated_data = data[mask]

    # Use matplotlib contour plotter (implemented in C) to generate contour(s)
    # at the probability associated with the last accepted point.
    x,y = numpy.meshgrid(xe[:-1], ye[:-1], indexing = 'ij')
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
        if not numpy.all((codes==1)|(codes==2)):
            raise Exception('Contour error: unrecognized path code')

        cntr.append(vertices)

    return gated_data, cntr
