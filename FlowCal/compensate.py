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
        Data corresponding to the no-fluorophore control sample (NFC). If
        None, no autofluorescence correction will be made.
    sfc_samples : list of FCSData objects
        Data corresponding to the single-fluorophore control samples
        (SFCs).
    comp_channels : list of str or int
        Channels to compensate. Each channel should correspond to an
        element of `sfc_samples`.

    Returns
    -------
    transform_fxn : function
        Transformation function to compensate flow cytometry data.
        This function has the following signature::

            data_compensated = transform_fxn(data, channels)

    Other parameters
    ----------------
    statistic_fxn : function, optional
        Statistical function from ``FlowCal.stats`` used to calculate the
        representative fluorescence of each control sample.

    Notes
    -----
    If using MEF calibration, we recommend using calibrated NFC and SFCs,
    and calibrating all samples before applying compensation. Calibration
    can correct for some small nonlinearities in the instrument's
    fluorescence detectors, especially in older instruments. On the other
    hand, compensation requires fluorescence units proportional to the
    fluorescent signal. Thus, performing compensation followed by
    calibration may give slightly different results than running
    calibration followed by compensation as we recommend.

    The compensation method used here is based on the following analysis.

    We assume we have an instrument with :math:`n` fluorescence channels
    and a sample with :math:`n` fluorophores, where we expect the signal
    in channel :math:`i` to correspond to fluorophore :math:`i` only.
    In reality, the signal from each channel will additionally contain
    some (hopefully small) signal from all other fluorophores. In
    mathematical terms:

    .. math:: s^i = a_0^i + f_i + \\sum_{j=1,j\\neq i}^{n} a^i_j \\cdot f_j

    where:

    - :math:`s^i` is the total signal observed in channel :math:`i`.
    - :math:`a^i_0` is the autofluorescence  signal in channel :math:`i`.
    - :math:`f_i` is the signal from fluorophore :math:`i`.
    - :math:`a^i_j` is a bleedthrough coefficient, which quantifies how
      much signal from fluorophore :math:`j` appears in channel :math:`i`.
        
    In matrix notation:

    .. math:: \\mathbf{s} = \\mathbf{a_0} + \\mathbf{A} \\cdot \\mathbf{f}

    Where :math:`\\mathbf{a_0}` is the autofluorescence vector and
    :math:`\\mathbf{A}` is the bleedthrough matrix, with all diagonal terms
    equal to one. For an arbitrary sample, the compensation procedure
    consists on solving for :math:`\\mathbf{f}` starting from the measured
    signals :math:`\\mathbf{s}`:

    .. math:: \\mathbf{f} = \\mathbf{A}^{-1} (\\mathbf{s} - \\mathbf{a_0})

    This requires knowledge of :math:`\\mathbf{A}` and
    :math:`\\mathbf{a_0}`. To find these out, we use the following control
    samples:

    1. No-fluorophore control (NFC). In this case,
    :math:`\\mathbf{f}_{NFC} = 0`. Therefore,

    .. math:: \\mathbf{a_0} = \\mathbf{s}_{NFC}

    Where :math:`\\mathbf{s}_{NFC}` is the vector containing the signals in
    all channels when measuring the NFC.

    2. Single-fluorophore controls (SFCs), one for each fluorophore. For
    fluorophore :math:`i`, all elements in :math:`\\mathbf{f}_{SFCi}` are 
    zero, except for the one at position :math:`i` (:math:`f_{SFCi}`).
    Therefore,

    .. math:: \\mathbf{s}_{SFCi} = \\mathbf{a_0} + \
    \\mathbf{a}_i \\cdot f_{SFCi}

    Where :math:`\\mathbf{s}_{SFCi}` is the vector containing the signals
    in all channels when measuring the SFC with fluorophore :math:`i`, and
    :math:`\\mathbf{a}_i` is the ith column of :math:`A`. Solving for
    :math:`\\mathbf{a}_i`:

    .. math:: \\mathbf{a}_i = (\\mathbf{s}_{SFCi} - \\mathbf{a_0})/f_{SFCi}

    Finally, using the additional restriction that :math:`a_i^i=1`, we
    have:

    .. math:: f_{SFCi} = s^i_{SFCi} - a^i_0

    Therefore

    .. math:: \\mathbf{a}_i = (\\mathbf{s}_{SFCi} - \\mathbf{a_0})/ \
    (s^i_{SFCi} - a^i_0)

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
    # Signals from the single-fluorophore controls
    # Matrix S_sfc contains the signal from an SFC in each row
    # S_sfc[:, i] <= s_SFCi
    S_sfc = np.array(
        [np.array(statistic_fxn(s[:,comp_channels])) for s in sfc_samples]).T
    # Get signal minus autofluorescence
    # Each column in S_sfc_noauto contains the signal from an SFC minus the
    # autofluorescence vector
    # S_sfc_noauto[:, i] <= s_SFCi - a_0
    S_sfc_noauto = S_sfc - a0[:, np.newaxis]
    # Calculate matrix A
    # The following uses broadcasting to divide column i by the (i,i) element
    # of S_sfc_noauto
    # A[:, i] <= (s_SFCi - a0)/(s^i_SFCi - a^i_0)
    A = S_sfc_noauto / np.diag(S_sfc_noauto)

    # Make output transformation function
    transform_fxn = functools.partial(FlowCal.transform.to_compensated,
                                      comp_channels=comp_channels,
                                      a0=a0,
                                      A=A)

    return transform_fxn
