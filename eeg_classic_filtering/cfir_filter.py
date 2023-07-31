import typing as tp
import numpy as np
from scipy.signal import lfilter


class CFIRBandDetector:
    def __init__(self,
                 delay: int,
                 band: tuple[float, float] = (8, 12),
                 sampling_rate: float = 250,
                 n_taps: int = 500,
                 n_fft: int = 2000,
                 weights: tp.Optional[np.ndarray] = None
                 ):
        """
        Class for the cFIR filter. It approximates an ideal FIR filter with specified n_taps, n_fft and band while
        trying to stay within the requested delay. It is a complex-valued filter, so it outputs approximation of an
        analytic signal

        Parameters
        ----------
        delay : int
                    Requested delay of the filter in samples. Can be 0. Note, filter's true delay will usually not
                    match the requested delay perfectly for low delay values. However, it typically helps to
                    reduce delay to decently small value
        band : tuple (band_low, band_high)
                    Cutoff band of an ideal bandpass FIR filter
        sampling_rate : float
                    Sampling rate of the data
        n_taps : int
                    Number of taps of the ideal FIR filter
        n_fft : int
                    Length of an ideal FIR filter
        weights : ndarray
                    Weights for cFIR filter. If None uses default ones
        """
        sampling_rate = float(sampling_rate)
        band = (float(band[0]), float(band[1]))
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * sampling_rate < band[0]) | (w / n_fft * sampling_rate > band[1])] = 0
        F = np.array(
            [np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)]
        )
        if weights is None:
            self.b = F.T.conj().dot(H) / n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.0])
        self.zi = np.zeros(len(self.b) - 1)
        self.delay = delay

    def filter_data(self, data: np.ndarray):
        """
        Filter data with cFIR filter

        Parameters
        ----------
        data : ndarray
                    Data to be filtered

        Returns
        -------
        y : ndarray
                    Approximated filtered analytic signal from the data
        """
        y, self.zi = lfilter(self.b, self.a, data, zi=self.zi)
        return y

