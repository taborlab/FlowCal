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

import numpy as np
import scipy.ndimage.filters
import matplotlib._cntr         # matplotlib contour, implemented in C

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
    mask = np.all((data_ch < high) & (data_ch > low), axis=1)
    gated_data = data[mask]

    return gated_data
    
def start_end(data, num_start=250, num_end=100):
    '''Gate out num_start first and num_end last events collected.

    data      - NxD FCSData object or numpy array
    num_start - number of points to discard from the beginning of data
    num_end   - number of points to discard from the end of data

    returns  - Gated MxD FCSData object or numpy array'''
    
    if data.shape[0] < (num_start + num_end):
        raise ValueError('Number of events to discard greater than total' + 
            ' number.')
    
    mask = np.ones(shape=data.shape[0],dtype=bool)
    mask[:num_start] = False
    mask[-num_end:] = False
    gated_data = data[mask]
    
    return gated_data

# def whitening2d(data, gate_fraction=0.65):
#     '''Use whitening transformation to transform (x,y) values into a space
#     where median-based covariance is the identity matrix and gate out all
#     events but those closest to the transformed 2D (x,y) median.

#     data          - NxD numpy array (only first 2 dimensions [columns] are
#                     used)
#     gate_fraction - fraction of data points to keep

#     returns       - Boolean numpy array of length N, 2D numpy array of (x,y)
#                     coordinates of gate contour'''

#     if len(data.shape) < 2:
#         raise ValueError('must specify at least 2 dimensions')

#     if data.shape[0] < 2:
#         raise ValueError('data must have more than 1 event')

#     # Determine number of points to keep
#     n = int(np.ceil(gate_fraction*float(data.shape[0])))

#     # Calculate median-based covariance matrix (measure mean squared distance
#     # to median instead of mean)
#     m = np.median(data[:,0:2],0)
#     X = m-data[:,0:2]
#     S = X.T.dot(X) / float(data.shape[0])

#     # Calculate eigenvectors
#     w,v = np.linalg.eig(S)

#     # Transform median-centered data into new eigenspace. This is equivalent
#     # to "whitening" the data; scales data to data with median-based
#     # covariance of the indentity matrix.
#     transformed_data = X.dot(v).dot(np.diag(1.0/np.sqrt(w)))

#     # Calculate distance to median (which is the origin in the new eigenspace)
#     d = np.sqrt(np.sum(np.square(transformed_data),1))

#     # Select closest points
#     idx = sorted(xrange(d.shape[0]), key=lambda k: d[k])
#     mask = np.zeros(shape=data.shape[0],dtype=bool)
#     mask[idx[:n]] = True
    
#     # Last point defines boundary of circle in eigenspace which can be
#     # transformed back into original data space and serve as gate contour
#     theta = np.arange(0,2*np.pi,2*np.pi/100)
#     r = d[idx[n-1]]
#     x = [r*np.cos(t) for t in theta]
#     y = [r*np.sin(t) for t in theta]

#     # Close the circle
#     x.append(x[0])
#     y.append(y[0])
    
#     c = np.array([x,y]).T
    
#     # Transform circle back into original space and add the median.
#     # Note: inv(v) = v.T since columns are orthonormal
#     cntr = m + c.dot(np.diag(np.sqrt(w))).dot(v.T)

#     return mask, cntr

# def density2d(data, bins=np.arange(1025)-0.5, sigma=10.0, gate_fraction=0.65):
#     '''Blur 2D histogram using a 2D Gaussian filter, normalize the resulting
#     blurred histogram to make it a valid probability mass function, and gate
#     out all but the "densest" points (points with the largest probability).

#     data          - NxD numpy array (only first 2 dimensions [columns] are
#                     used)
#     bins          - bins argument to np.histogram2d
#                     (default=np.arange(1025)-0.5)
#     sigma         - standard deviation for Gaussian kernel
#     gate_fraction - fraction of data points to keep

#     returns       - Boolean numpy array of length N, list of 2D numpy arrays
#                     of (x,y) coordinates of gate contour(s)'''

#     if len(data.shape) < 2:
#         raise ValueError('must specify at least 2 dimensions')

#     if data.shape[0] < 2:
#         raise ValueError('data must have more than 1 event')

#     # Determine number of points to keep
#     n = int(np.ceil(gate_fraction*float(data.shape[0])))

#     # Make 2D histogram
#     H,xe,ye = np.histogram2d(data[:,0], data[:,1], bins=bins)

#     # Blur 2D histogram
#     bH = scipy.ndimage.filters.gaussian_filter(
#         H,
#         sigma=sigma,
#         order=0,
#         mode='constant',
#         cval=0.0,
#         truncate=6.0)

#     # Normalize filtered histogram to make it a valid probability mass function
#     D = bH / np.sum(bH)

#     # Sort each (x,y) point by density
#     vD = D.ravel()
#     vH = H.ravel()
#     sidx = sorted(xrange(len(vD)), key=lambda idx: vD[idx], reverse=True)
#     svH = vH[sidx]  # linearized counts array sorted by density

#     # Find minimum number of accepted (x,y) points needed to reach specified
#     # number of data points
#     csvH = np.cumsum(svH)
#     Nidx = np.nonzero(csvH>=n)[0][0]    # we want to include this index

#     # Convert accepted (x,y) linear indices into 2D indices into the histogram
#     # matrix
#     fsc,ssc = np.unravel_index(sidx[:(Nidx+1)], H.shape)
#     accepted_points = set(zip(fsc,ssc))
#     mask = np.array([tuple(event) in accepted_points for event in data[:,0:2]])

#     # Use matplotlib contour plotter (implemented in C) to generate contour(s)
#     # at the probability associated with the last accepted point.
#     x,y = np.mgrid[0:1024,0:1024]
#     mpl_cntr = matplotlib._cntr.Cntr(x,y,D)
#     tr = mpl_cntr.trace(vD[sidx[Nidx]])

#     # trace returns a list of arrays which contain vertices and path codes
#     # used in matplotlib Path objects (see http://stackoverflow.com/a/18309914
#     # and the documentation for matplotlib.path.Path for more details). I'm
#     # just going to make sure the path codes aren't unfamiliar and then extract
#     # all of the vertices and pack them into a list of 2D contours.
#     cntr = []
#     num_cntrs = len(tr)/2
#     for idx in xrange(num_cntrs):
#         vertices = tr[idx]
#         codes = tr[num_cntrs+idx]

#         # I am only expecting codes 1 and 2 ('MOVETO' and 'LINETO' codes)
#         if not np.all((codes==1)|(codes==2)):
#             raise Exception('contour error: unrecognized path code')

#         cntr.append(vertices)

#     return mask, cntr
