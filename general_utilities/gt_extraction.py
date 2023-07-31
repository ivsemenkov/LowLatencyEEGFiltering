import numpy as np
import scipy.signal as sn


def apply_hfir(signal: np.ndarray, band: tuple[float, float], window, numtaps: int, srate: float, delay: int):
    """
    Function which filters signal with HFIR filter. We were finding central frequency with find_central_freq and
    taking band as (central_freq - 2, central_freq + 2)

    Parameters
    ----------
    signal : ndarray
                Signal to filter
    band : tuple (band_low, band_high)
                Cutoff for bandpass filter
    window :
                Window or window name for a filter
    numtaps : int
                Length of a filter
    srate : float
                Sampling rate of signal
    delay : int
                Filter delay to align indices

    Returns
    -------
    hfir_filtered : ndarray
                Complex-valued filtered analytic signal
    """
    b = sn.firwin(numtaps, cutoff=band, window=window, pass_zero=False, fs=srate)
    b_hfir = sn.hilbert(b)
    hfir_filtered = sn.lfilter(b_hfir, 1.0, signal)
    hfir_filtered = hfir_filtered[delay:]
    return hfir_filtered


def find_central_freq(data: np.ndarray, srate: float):
    """
    Find central frequency in the data

    Parameters
    ----------
    data : ndarray
                Data where to look for central frequency
    srate : float
                Sampling rate of data

    Returns
    -------
    alpha_freq : float
                Central frequency in band (8, 13)
    """
    f, pxx = sn.welch(data, fs=srate, nperseg=500, nfft=1000, noverlap=250)

    mask = np.logical_and((f > 8.0), (f < 13.0))
    max_idx = np.argmax(pxx[mask])
    alpha_freq = f[mask][max_idx]
    return alpha_freq
