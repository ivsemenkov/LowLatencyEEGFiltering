import typing as tp
import numpy as np
from .models import collect, MatsudaParams, SingleRhythmModel
from .simulations import gen_filt_pink_noise_w_added_pink_noise, gen_sine_w_pink, gen_sine_w_white, make_pink_noise


def generate_datapoint(duration_sec: float, amount_of_steps: int, fs: float, dataset_name: str,
                       rng: tp.Optional[np.random.Generator],
                       frequency: tp.Optional[float] = 10, filter_band: tp.Optional[tuple[float, float]] = (8, 12),
                       noise_coeff: float = 1., n_samp: tp.Optional[int] = None):
    """
    Generate synthetic data

    Parameters
    ----------
    duration_sec : float
                Amount of seconds per single time-series in a test dataset
    amount_of_steps : int
                Forecasting horizon
    fs : float
                Sampling rate for the data
    dataset_name : str
                Synthetic dataset name to generate. Should be in ['sines_white', 'sines_pink', 'filtered_pink',
                'state_space_white', 'state_space_pink']
    rng : np.random.Generator or None
                Random generator. Can be passed with a fixed seed for reproducibility. If None will use unpredictable
                seed
    frequency : float
                Central frequency for dataset generation. Use None for filtered_pink as it relies on filter_band
    filter_band : tuple (band_low, band_high)
                Band to filter ground truth in for the filtered_pink dataset. Use None for others
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    n_samp : int or None
                If int a function will generate n_samp samples. If None will use duration_sec, amount_of_steps and fs

    Returns
    -------
    data : ndarray
                Array of unfiltered, noisy observations
    gt : ndarray
                Array of noise-free ground truth analytic signal from which data was created
    """
    assert dataset_name in ['sines_white', 'sines_pink', 'filtered_pink', 'state_space_white', 'state_space_pink']

    if rng is None:
        rng = np.random.default_rng(None)
    else:
        assert isinstance(rng, np.random.Generator)

    if dataset_name == 'sines_white':
        data, gt, _ = gen_sine_w_white(duration_sec=duration_sec,
                                       amount_of_steps=amount_of_steps,
                                       Fs=fs, FREQ_HZ=frequency,
                                       noise_coeff=noise_coeff,
                                       n_samp=n_samp, rng=rng)
    elif dataset_name == 'sines_pink':
        data, gt, _ = gen_sine_w_pink(duration_sec=duration_sec,
                                      amount_of_steps=amount_of_steps,
                                      Fs=fs, FREQ_HZ=frequency,
                                      noise_coeff=noise_coeff,
                                      n_samp=n_samp, rng=rng)
    elif dataset_name == 'filtered_pink':
        data, gt, _ = gen_filt_pink_noise_w_added_pink_noise(duration_sec=duration_sec,
                                                             amount_of_steps=amount_of_steps,
                                                             Fs=fs, FIR_BAND_HZ=filter_band,
                                                             noise_coeff=noise_coeff,
                                                             n_samp=n_samp, rng=rng)
    elif dataset_name in ['state_space_white', 'state_space_pink']:
        data, gt = generate_data_state_space(duration_sec=duration_sec,
                                             amount_of_steps=amount_of_steps,
                                             fs=fs, dataset_name=dataset_name,
                                             frequency=frequency, noise_coeff=noise_coeff,
                                             n_samp=n_samp, rng=rng)
    else:
        raise ValueError

    return data, gt


def generate_data_state_space(duration_sec: float, amount_of_steps: int, fs: float, dataset_name: str, frequency: float,
                              rng: np.random.Generator, noise_coeff: float = 1., n_samp: tp.Optional[int] = None):
    """
    Generate state space data

    Parameters
    ----------
    duration_sec : float
                Amount of seconds per single time-series in a test dataset
    amount_of_steps : int
                Forecasting horizon
    fs : float
                Sampling rate for the data
    dataset_name : str
                Synthetic dataset name to generate. Should be in ['sines_white', 'sines_pink', 'filtered_pink',
                'state_space_white', 'state_space_pink']
    frequency : float
                Central frequency for dataset generation. Use None for filtered_pink as it relies on filter_band
    rng : np.random.Generator or None
                Random generator. Can be passed with a fixed seed for reproducibility. If None will use unpredictable
                seed
    noise_coeff : float
                Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
    n_samp : int or None
                If int a function will generate n_samp samples. If None will use duration_sec, amount_of_steps and fs

    Returns
    -------
    data : ndarray
                Array of unfiltered, noisy observations
    gt : ndarray
                Array of noise-free ground truth analytic signal from which data was created
    """
    if n_samp is None:
        n_samp = int(duration_sec * fs) + amount_of_steps

    # Setup oscillatioins model and generate oscillatory signal
    a_gt = 0.99  # as in x_next = A*exp(2*pi*OSCILLATION_FREQ / sr)
    signal_sigma_gt = np.sqrt(10)  # std of the model-driving white noise in the Matsuda model

    mp = MatsudaParams(A=a_gt, freq=frequency, sr=fs)
    oscillation_model = SingleRhythmModel(mp=mp, sigma=signal_sigma_gt)
    gt = collect(oscillation_model, n_samp)

    # Setup simulated noise and measurements
    if dataset_name == 'state_space_white':
        noise_synthetic = rng.standard_normal(len(gt))
        data = np.real(gt) + noise_coeff * noise_synthetic
    elif dataset_name == 'state_space_pink':
        PINK_NOISE_SNR = 10
        PINK_NOISE_ALPHA = 1.5
        noise_synthetic = make_pink_noise(alpha=PINK_NOISE_ALPHA, n_samp=n_samp, dt=1 / fs, rng=rng)
        data = np.real(gt) + PINK_NOISE_SNR * noise_coeff * noise_synthetic
    else:
        raise ValueError

    return data, gt
