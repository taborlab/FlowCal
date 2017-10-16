"""
Functions for performing multicolor compensation.

"""

import numpy
import FlowCal.stats

def compensate(sample, channels, a0, A):
    """
    Apply multicolor compensation to a flow cytometry sample.

    Parameters
    ----------
    sample : FCSData object
        Flow cytometry data to compensate.
    channels : list
        Channels to compensate.
    a0 : array
        Autofluorescence vector.
    A : 2D array
        Bleedthrough matrix.

    Returns
    -------
    sample_comp : FCSData object
        Compensated flow cytometry data.

    Notes
    -----
    TODO: Explain compensation algorithm and how a0 and A are calculated.

    """
    # Check appropriate dimensions of a0 and A
    if a0.shape != (len(channels),):
        ValueError('length of a0 should be the same as the number of channels')
    if A.shape != (len(channels), len(channels)):
        ValueError('A should be a square matrix with size equal to the number'
            ' of channels')

    # Copy sample so that the input paramenter is not modified
    sample_comp = sample.copy()
    # Apply compensation
    sample_comp[:, channels] = \
        numpy.linalg.solve(A, (sample[:, channels] - a0).T).T

    return sample_comp

def get_compensation_params(nfc_sample,
                            sfc_samples,
                            channels,
                            statistic_fxn=FlowCal.stats.median):
    """
    Calculate the coefficients necessary for multicolor compensation.

    Parameters
    ----------
    nfc_sample : FCSData object
        Data corresponding to the no-fluorophore control sample.
    sfc_samples : list of FCSData object
        Data corresponding to the single-fluorophore control samples.
    channels : list
        Channels to compensate. After applying compensation, the
        fluorophore in ``sfc_samples[i]`` will be expressed in units of
        ``channels[i]``.

    Returns
    -------
    a0 : array
        Autofluorescence vector.
    A : 2D array
        Bleedthrough matrix.

    Other parameters
    ----------------
    statistic_fxn : function, optional
        Function used to calculate the representative fluorescence of each
        control sample. Must have the following signature::

            s = statistic_fxn(data, **statistic_params)

        where `data` is a 1D FCSData object or numpy array, and `s` is a
        float. Statistical functions from numpy, scipy, or FlowCal.stats
        are valid options.

    Notes
    -----
    TODO: Explain compensation algorithm and how a0 and A are calculated.

    """
    # Check for appropriate number of single fluorophore controls
    if len(sfc_samples) != len(channels):
        ValueError('number of single fluorophore controls should match'
            ' the number of fluorophores specified')

    # Autofluorescence vector
    a0 = numpy.array(statistic_fxn(nfc_sample[:,channels]))
    # Signal on the single-fluorophore controls
    # s_sfc[i,j] = s_sfc^j_i (fluorophore i, channel j)
    s_sfc = [numpy.array(statistic_fxn(s[:,channels])) for s in sfc_samples]
    # Get signal minus autofluorescence
    # s_bs[i,j] = s_sfc^j_i - a_0^j (fluorophore i, channel j)
    s_bs = numpy.array([s - a0 for s in s_sfc])
    # Calculate matrix A
    # A[i,j] = (s_sfc^i_j - a_0^i)/(s_sfc^j_j - a_0^j)
    A = s_bs.T / numpy.diag(s_bs)
    
    return a0, A