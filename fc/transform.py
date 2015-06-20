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

def _fit_mef_standards_curve(fit_peaks, known_peaks, method='log line+auto'):
    '''Fit a model mapping fit calibration bead peaks (in channels) to known
    calibration bead peaks (in MEF). Empirically, we have determined that the
    following model serves best for our purposes:

        log(Y + M(3)) = (M(1)*X) + M(2)

    where Y=known_peaks, X=fit_peaks, and M represents the 3 model parameters.

    This model was derived from a straightforward exponential model (fitting
    a line on a log-y plot; M(1) and M(2)) which was augmented with an
    autofluorescence offset term (M(3)) to account for the fact that, in
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

    # error checking? fit_peaks and known_peaks need to be same length

    if model == 'log line':
        pass
    elif model == 'exp':
        pass
    elif model == 'exp+auto':
        pass
    elif model == 'log line+auto':
        pass
    else: 
        raise ValueError('Unrecognized model')
    
    def standards_curve(data):
        '''Transform data from Channels to Molecules of Equivalent
        Fluorophore (MEF).'''
        
        return data
    
    return standards_curve

def to_mef(data, beads, known_mef, peaks=None, sc=None):
    '''description'''
    
    if sc is None:
        if peaks is None:
            peaks = _find_hist_peaks(beads)
            
        sc = _fit_mef_standards_curve(known_mef, peaks)
            
    return sc(data)

def blank(data, white_cells):
    pass

def compensate(data):
    pass
