"""
Functions for performing multicolor compensation.

"""

import functools

import numpy as np
import FlowCal.stats

def get_transform_fxn(nfc_sample,
                      sfc_samples,
                      comp_channels,
                      statistic_fxn=FlowCal.stats.mean):
    """
    Get a transformation function to perform multicolor compensation.

    Parameters
    ----------
    nfc_sample : FCSData object or None
        Data corresponding to the no-fluorophore control sample. If None,
        autofluorescnece is set to zero.
    sfc_samples : list of FCSData object
        Data corresponding to the single-fluorophore control samples.
    comp_channels : list
        Channels to compensate. The number of channels should match the
        number of single-fluorophore control samples.

    Returns
    -------
    transform_fxn : function
        Transformation function to compensate flow cytometry data.
        This function has the following signature::

            data_compensated = transform_fxn(data, channels)

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
    The compensatiom method used here is based on the following analysis.

    On an instrument with :math:`n` channels and a sample with :math:`n`
    fluorophores:

    .. math:: s^i = a_0^i + \\sum_{j=1}^{n} a^i_j \\cdot f_j

    where:

    - :math:`s^i` is the signal read in channel :math:`i`, in channel MEF
    - :math:`a^i_0` is the autofluorescence in channel :math:`i`
    - :math:`a^i_j` is the signal in channel :math:`i` due to fluorophore
      :math:`j`
    - :math:`f_j` is the amount of fluorophore :math:`j`, in some units to
      be determined.
        
    Or in matrix notation:

    .. math:: s = a_0 + A \\cdot f

    The compensation procedure consists on finding numbers that are
    proportional to the amount of a single fluorophore. In other words,
    compensation is the process of finding the vector :math:`f` starting
    from signals :math:`s` for an arbitrary sample. This is performed by
    applying the following equation:

    .. math:: f = A^{-1} \\cdot (s - a_0)

    This requires knowledge of the matrix :math:`A` and the vector
    :math:`a_0`.

    We assume that :math:`A` and :math:`a_0` stay constant with different
    samples, provided that the fluorophores involved are identical and the
    instrument stays the same. This makes it possible for us to use
    *control samples* that facilitate calculating :math:`a_0` and
    :math:`A`.

    1. No-fluorophore control. In this case, :math:`f_j = 0` for all
    :math:`j`. Therefore,

    .. math:: a_0^i = s^i_{NFC}

    Where :math:`s^i_{NFC}` is the signal in channel :math:`i` due to the
    no fluorophore control sample.

    2. Single-fluorophore control. For fluorophore :math:`k`,
    :math:`f_j = 0` for all :math:`j \\neq k`. Therefore,

    .. math:: s^i_{SFCk} = a_0^i +  a^i_k \\cdot f_k

    Where :math:`s^i_{SFCk}` is the signal in channel :math:`i` due to
    the single fluorophore control sample with fluorophore :math:`k`.

    Two unknowns remain: :math:`a^i_k` and :math:`f_k`. In fact,
    :math:`f_k` is undetermined up to a proportionality constant.
    Therefore, we add the following two restrictions. First, we choose
    to assign one channel to each fluorophore. Practically, this would be
    the channel whose emission window has the greatest overlap with the
    emission spectrum of the fluorophore. We therefore assign channel
    :math:`k` to fluorophore :math:`k`. Next, we choose to express the
    amount of fluorophore :math:`k` the same units as channel :math:`k`.
    These restrictions can be expressed mathematically as:

    .. math:: a^k_k = 1

    And therefore:

    .. math::

        s^k_{SFCk} = a_0^k + f_k

        f_k = s^k_{SFCk} - a_0^k

    And we can calculate the remaining coefficients of :math:`A` as
    follows:

    .. math:: a^i_k = (s^i_{SFCk} - a_0^i)/(s^k_{SFCk} - a_0^k)

    for all :math:`i \\neq k`.

    """

    # Check for appropriate number of single fluorophore controls
    if len(sfc_samples) != len(comp_channels):
        ValueError('number of single fluorophore controls should match'
            ' the number of channels specified')

    # Calculate autofluorescence vector
    if nfc_sample is None:
        a0 = np.zeros(len(comp_channels))
    else:
        a0 = np.array(statistic_fxn(nfc_sample[:,comp_channels]))
    # Signal on the single-fluorophore controls
    # s_sfc[i,j] = s_sfc^j_i (fluorophore i, channel j)
    s_sfc = [np.array(statistic_fxn(s[:,comp_channels])) for s in sfc_samples]
    # Get signal minus autofluorescence
    # s_bs[i,j] = s_sfc^j_i - a_0^j (fluorophore i, channel j)
    s_bs = np.array([s - a0 for s in s_sfc])
    # Calculate matrix A
    # A[i,j] = (s_sfc^i_j - a_0^i)/(s_sfc^j_j - a_0^j)
    A = s_bs.T / np.diag(s_bs)

    # Make output transformation function
    transform_fxn = functools.partial(FlowCal.transform.to_compensated,
                                      comp_channels=comp_channels,
                                      a0=a0,
                                      A=A)

    return transform_fxn
