"""
Code adapted and rewritten in Python from https://github.com/Eden-Kramer-Lab/SSPE-paper/tree/main
Anirudh Wodeyar, Mark Schatza, Alik S Widge, Uri T Eden, Mark A Kramer (2021)
A state space modeling approach to real-time phase estimation eLife 10:e68803
https://doi.org/10.7554/eLife.68803
"""
from typing import NamedTuple, Optional
import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt, firwin, hilbert


def make_pink_noise(alpha: float, n_samp: int, dt: float, rng: np.random.Generator) -> npt.NDArray[np.floating]:
    """
    Given an alpha value for the 1/f^alpha produce data of length n_samp and at Fs = 1/dt

    Parameters
    ----------
    alpha : float
                Alpha value for 1/f^alpha noise
    n_samp : int
                Amount of samples to generate
    dt : float
                Inverse of a sampling rate (Fs = 1/dt)
    rng : np.random.Generator
                Random generator

    Returns
    -------
    pink_noise : ndarray
                Pink noise samples
    """
    x1 = rng.standard_normal(n_samp)
    xf1 = np.fft.fft(x1)
    A = np.abs(xf1)
    phase = np.angle(xf1)

    df = 1 / (dt * len(x1))
    faxis = np.arange(len(x1) // 2 + 1) * df
    faxis = np.concatenate([faxis, faxis[-2:0:-1]])
    faxis[0] = np.inf

    oneOverf = 1 / faxis**alpha

    Anew = np.sqrt((A**2) * oneOverf.T)
    xf1new = Anew * np.exp(1j * phase)
    return np.real(np.fft.ifft(xf1new)).T


class SimulationResults(NamedTuple):
    data: npt.NDArray[np.floating]
    gt_states: npt.NDArray[np.complex_]
    true_phase: npt.NDArray[np.floating]


def gen_sine_w_white(duration_sec: float, amount_of_steps: int, Fs: float, rng: np.random.Generator,
                     FREQ_HZ: float = 6, noise_coeff: float = 1., n_samp: Optional[int] = None) -> SimulationResults:
    """
    Generate sines with white noise

    Parameters
    ----------
    duration_sec : float
                Amount of seconds per single time-series in a test dataset
    amount_of_steps : int
                Forecasting horizon
    Fs : float
                Sampling rate for the data
    rng : np.random.Generator or None
                Random generator. Can be passed with a fixed seed for reproducibility. If None will use unpredictable
                seed
    FREQ_HZ : float
                Central frequency for dataset generation
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    n_samp : int or None
                If int a function will generate n_samp samples. If None will use duration_sec, amount_of_steps and Fs

    Returns
    -------
    data : ndarray
                Array of unfiltered, noisy observations
    gt_states : ndarray
                Array of noise-free ground truth analytic signal from which data was created
    true_phase : ndarray
                Array of ground truth phases
    """
    A = 10

    if n_samp is None:
        n_samp = int(duration_sec * Fs) + amount_of_steps
    times = np.arange(1, n_samp + 1) / Fs
    true_phase = 2 * np.pi * FREQ_HZ * times
    Vlo = A * np.cos(true_phase)
    data = Vlo + noise_coeff * rng.standard_normal(n_samp)
    true_phase = _wrapToPi(true_phase)
    gt_states = Vlo + 1j * A * np.sin(true_phase)
    return SimulationResults(data, gt_states, true_phase)


def gen_sine_w_pink(duration_sec: float, amount_of_steps: int, Fs: float, rng: np.random.Generator, FREQ_HZ: float = 6,
                    noise_coeff: float = 1., n_samp: Optional[int] = None) -> SimulationResults:
    """
    Generate sines with pink noise

    Parameters
    ----------
    duration_sec : float
                Amount of seconds per single time-series in a test dataset
    amount_of_steps : int
                Forecasting horizon
    Fs : float
                Sampling rate for the data
    rng : np.random.Generator or None
                Random generator. Can be passed with a fixed seed for reproducibility. If None will use unpredictable
                seed
    FREQ_HZ : float
                Central frequency for dataset generation
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    n_samp : int or None
                If int a function will generate n_samp samples. If None will use duration_sec, amount_of_steps and Fs

    Returns
    -------
    data : ndarray
                Array of unfiltered, noisy observations
    gt_states : ndarray
                Array of noise-free ground truth analytic signal from which data was created
    true_phase : ndarray
                Array of ground truth phases
    """
    A = 10
    PINK_NOISE_SNR = 10
    PINK_NOISE_ALPHA = 1.5

    if n_samp is None:
        n_samp = int(duration_sec * Fs) + amount_of_steps
    noise = make_pink_noise(alpha=PINK_NOISE_ALPHA, n_samp=n_samp, dt=1 / Fs, rng=rng)
    times = np.arange(1, n_samp + 1) / Fs
    true_phase = 2 * np.pi * FREQ_HZ * times
    Vlo = A * np.cos(true_phase)
    data = Vlo + PINK_NOISE_SNR * noise_coeff * noise
    true_phase = _wrapToPi(true_phase)
    gt_states = Vlo + 1j * A * np.sin(true_phase)
    return SimulationResults(data, gt_states, true_phase)


def gen_filt_pink_noise_w_added_pink_noise(duration_sec: float, amount_of_steps: int, Fs: float,
                                           rng: np.random.Generator,
                                           FIR_BAND_HZ: tuple[float, float] = (8, 12), noise_coeff: float = 1.,
                                           n_samp: Optional[int] = None) -> SimulationResults:
    """
    Generate broadband oscillation in pink noise

    To get the broadband oscillation, filtfilt pink noise with FIR bandpass filter

    Parameters
    ----------
    duration_sec : float
                Amount of seconds per single time-series in a test dataset
    amount_of_steps : int
                Forecasting horizon
    Fs : float
                Sampling rate for the data
    rng : np.random.Generator or None
                Random generator. Can be passed with a fixed seed for reproducibility. If None will use unpredictable
                seed
    FIR_BAND_HZ : tuple (band_low, band_high)
                Band to filter ground truth in
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    n_samp : int or None
                If int a function will generate n_samp samples. If None will use duration_sec, amount_of_steps and Fs

    Returns
    -------
    data : ndarray
                Array of unfiltered, noisy observations
    gt_states : ndarray
                Array of noise-free ground truth analytic signal from which data was created
    true_phase : ndarray
                Array of ground truth phases
    """
    A = 10
    FIR_ORDER = 750
    FIR_BAND_HZ = list(FIR_BAND_HZ)
    PINK_NOISE_SNR = 10
    PINK_NOISE_ALPHA = 1.5

    if n_samp is None:
        n_samp = int(duration_sec * Fs) + amount_of_steps
    b = firwin(numtaps=FIR_ORDER + 1, cutoff=FIR_BAND_HZ, fs=Fs, pass_zero=False)

    pn_signal = filtfilt(b=b, a=[1], x=make_pink_noise(alpha=PINK_NOISE_ALPHA, n_samp=n_samp, dt=1 / Fs, rng=rng))
    pn_noise = make_pink_noise(alpha=PINK_NOISE_ALPHA, n_samp=n_samp, dt=1 / Fs, rng=rng)

    Vlo = A * (pn_signal / pn_signal.std())
    data = Vlo + PINK_NOISE_SNR * noise_coeff * pn_noise
    gt_states: npt.NDArray[np.complex_] = hilbert(Vlo)  # pyright: ignore
    true_phase = np.angle(gt_states)
    return SimulationResults(data, gt_states, true_phase)


def _wrapToPi(phase: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Emulate MATLAB's wrapToPi, https://stackoverflow.com/a/71914752

    Parameters
    ----------
    phase : ndarray
                Array of phases

    Returns
    -------
    xwrap : ndarray
                New array of phases
    """
    xwrap = np.remainder(phase, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = phase < 0
    mask2 = np.remainder(phase, np.pi) == 0
    mask3 = np.remainder(phase, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap
