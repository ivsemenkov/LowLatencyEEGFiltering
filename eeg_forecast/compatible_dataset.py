import typing as tp
import numpy as np
from darts import TimeSeries
from general_utilities.data_generator import generate_datapoint


class DartsCompatibleDataset:
    def __init__(self,
                 amount_of_observations: int,
                 gt_lag: int):
        """
        Base class for the dataset compatible with darts library

        Parameters
        ----------
        amount_of_observations : int
                    Amount of time-series (observations) in a resultant dataset
        gt_lag : int
                    A lag of ground truth relative to inputs which should be compensated
        """
        self.amount_of_observations = amount_of_observations
        self.gt_lag = gt_lag

    def extract_two_components(self, ts: np.ndarray):
        """
        Transform ts from complex series to two stacked real (with real and complex components as stacked series)

        Parameters
        ----------
        ts : ndarray
                    Time-series

        Returns
        -------
        final_ts : ndarray
                    Initial time-series where real and imaginary parts are stacked
        """
        ds_real = np.real(ts)
        ds_imag = np.imag(ts)
        final_ts = np.stack((ds_real, ds_imag), axis=-1)
        return final_ts

    def get_target_and_cov(self):
        """
        Get target time-series and covariates time-series

        Returns
        -------
        target_ts : ndarray
                    Target time-series
        covariates : ndarray
                    Time-series of covariates
        """
        return self.target_ts, self.covariates

    def split_test_target_and_cov(self, input_chunk_length: int, amount_of_splits: int):
        """
        Get target time-series and covariates time-series which are split to fit model's input size for testing
        per sample

        Parameters
        ----------
        input_chunk_length : int
                    Amount of lags which a model will require as an input to predict final point
        amount_of_splits : int
                    Number of splits per each target time-series

        Returns
        -------
        test_target_ts : ndarray
                    Split target time-series
        test_covariates : ndarray
                    Split time-series of covariates
        """
        test_target_ts = [[ts[idx:idx+input_chunk_length] for idx in range(amount_of_splits)]
                          for ts in self.target_ts]
        test_covariates = [[ts[idx:idx+input_chunk_length] for idx in range(amount_of_splits)]
                           for ts in self.covariates]
        return test_target_ts, test_covariates


class DartsCompatibleSyntheticDataset(DartsCompatibleDataset):
    def __init__(self,
                 target_ts_size: int,
                 dataset_path: str,
                 amount_of_observations: int,
                 first_observation_idx: int,
                 gt_lag: int,
                 gt_key: str):
        """
        Class for the pre-split synthetic dataset compatible with darts library

        Parameters
        ----------
        target_ts_size : int
                    How long (in samples) should be each time-series of data (there will be multiple of them).
                    Each such time-series is processed separately one sample at a time to imitate real-time framework
        dataset_path : str
                    Path to the respective dataset
        first_observation_idx : int
                    Index of a first observation (time-series) to consider. For example, if first_observation_idx == 5
                    it will ignore time-series with indices 0, 1, 2, 3, 4
        amount_of_observations : int
                    Amount of observations (time-series) to take from the pre-split dataset. For example, with
                    first_observation_idx == 5 and amount_of_observations == 10 it will take observations 5 to 5+10
                    from the pre-split dataset. Important: first_observation_idx + amount_of_observations cannot exceed
                    the amount of observations in the pre-split dataset
        gt_lag : int
                    A lag of ground truth relative to inputs which should be compensated
        gt_key : str
                    Key to the ground truth for the synthetic data. Note, in pre-split synthetic datasets we have keys
                    True_GT and HFIR_GT. True_GT will give an actual noiseless analytic signal from which noisy
                    samples were generated. HFIR_GT gives an approximation of True_GT by filtering noisy samples with
                    HFIR filter
        """
        super().__init__(amount_of_observations=amount_of_observations, gt_lag=gt_lag)

        self.target_ts_size = target_ts_size
        self.dataset_path = dataset_path
        self.start_ds_idx = first_observation_idx
        self.finish_ds_idx = first_observation_idx + amount_of_observations
        self.gt_key = gt_key
        self.load_data()

    def load_data(self):
        """
        Load pre-split dataset and transform into darts TimeSeries
        """
        data = np.load(self.dataset_path)
        assert data[self.gt_key].shape[1] >= self.target_ts_size + self.gt_lag, \
            f"{data[self.gt_key].shape} < {self.target_ts_size} + {self.gt_lag}"
        ts_t_x = [TimeSeries.from_values(data['Observations'][ds_idx, self.gt_lag:self.target_ts_size + self.gt_lag])
                  for ds_idx in range(self.start_ds_idx, self.finish_ds_idx)]
        ts_t_y = [TimeSeries.from_values(self.extract_two_components(data[self.gt_key])[ds_idx, :self.target_ts_size, :])
                  for ds_idx in range(self.start_ds_idx, self.finish_ds_idx)]
        self.target_ts = ts_t_y
        self.covariates = ts_t_x

    def get_test_target_and_cov(self, input_chunk_length: int):
        """
        Get target time-series and covariates time-series which are split to fit model's input size for testing
        per sample

        Parameters
        ----------
        input_chunk_length : int
                    Amount of lags which a model will require as an input to predict final point

        Returns
        -------
        test_target_ts : ndarray
                    Split target time-series
        test_covariates : ndarray
                    Split time-series of covariates
        """
        amount_of_splits = len(self.target_ts[0]) - input_chunk_length + 1

        self.test_target_ts, self.test_covariates = self.split_test_target_and_cov(
            input_chunk_length=input_chunk_length,
            amount_of_splits=amount_of_splits)

        return self.test_target_ts, self.test_covariates


class DartsCompatibleGeneratedDataset(DartsCompatibleDataset):
    def __init__(self,
                 input_chuck_length: int,
                 duration: float,
                 fs: float,
                 dataset_name: str,
                 frequency: tp.Optional[float],
                 filter_band: tp.Optional[tuple[float, float]],
                 amount_of_observations: int,
                 gt_lag: int,
                 noise_coeff: float,
                 rng_seed: tp.Optional[int] = 0):
        """
        Dataset class which generates darts compatible dataset for specified parameters

        Parameters
        ----------
        input_chuck_length : int
                    Amount of lags which a model will require as an input to predict final point
        duration : float
                    Amount of seconds per single time-series in a dataset
        fs : float
                    Sampling rate for the data
        dataset_name : str
                    Synthetic dataset name to generate. Should be in ['sines_white', 'sines_pink', 'filtered_pink',
                    'state_space_white', 'state_space_pink']
        frequency : float
                    Central frequency for dataset generation. Use None for filtered_pink as it relies on filter_band
        filter_band : tuple (band_low, band_high)
                    Band to filter ground truth in for the filtered_pink dataset. Use None for others
        amount_of_observations : int
                    Amount of time-series (observations) in a resultant dataset. Total dataset size is
                    duration * fs * amount_of_observations samples. Each time-series will be generated independently
        gt_lag : int
                    A lag of ground truth relative to inputs which should be compensated
        noise_coeff : float
                    Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                    noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
        rng_seed : int
                    A seed for the random number generator used to generated dataset. Useful for reproducibility.
                    If None, then unpredictable seed will be pulled from the OS
        """
        super().__init__(amount_of_observations=amount_of_observations, gt_lag=gt_lag)

        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.input_chuck_length = input_chuck_length
        self.duration = duration
        self.fs = fs
        self.dataset_name = dataset_name
        self.frequency = frequency
        self.filter_band = filter_band
        self.noise_coeff = noise_coeff
        self.n_samp = int(self.duration * self.fs) + self.gt_lag + self.input_chuck_length - 1
        self.target_ts_size = int(self.duration * self.fs)
        self.load_data()

    def get_initial_data(self):
        """
        Generate a single time-series (observation)

        Returns
        -------
        data : dict
                    Dictionary with keys 'full_X' for unfiltered, noisy inputs and 'full_y' for noise-free
                    ground truth analytic signal from which input_data was created
        """
        assert self.gt_lag >= 0
        full_X = []
        full_y = []
        for _ in range(self.amount_of_observations):

            input_data, gt_states = generate_datapoint(duration_sec=self.duration,
                                                       amount_of_steps=self.gt_lag,
                                                       fs=self.fs,
                                                       dataset_name=self.dataset_name,
                                                       frequency=self.frequency,
                                                       filter_band=self.filter_band,
                                                       noise_coeff=self.noise_coeff,
                                                       n_samp=self.n_samp,
                                                       rng=self.rng)

            assert np.iscomplexobj(gt_states)
            gt_states = np.stack((np.real(gt_states), np.imag(gt_states)), axis=1)

            full_X.append(input_data)
            full_y.append(gt_states)

        full_X = np.array(full_X)
        full_y = np.array(full_y)
        assert full_X.shape == (self.amount_of_observations, self.n_samp)
        assert full_y.shape == (self.amount_of_observations, self.n_samp, 2)
        data = {'full_X': full_X, 'full_y': full_y}
        return data

    def load_data(self):
        """
        Generates a whole dataset
        """
        added_one = False
        if self.n_samp % 2 == 1:
            self.n_samp += 1  # must be even for pink noise creation, otherwise creates offset of 1
            added_one = True
        data = self.get_initial_data()
        if added_one:
            self.n_samp -= 1  # back to target
        ts_t_x = [TimeSeries.from_values(data['full_X'][ds_idx, self.gt_lag:self.n_samp])
                  for ds_idx in range(self.amount_of_observations)]
        ts_t_y = [TimeSeries.from_values(data['full_y'][ds_idx, :self.n_samp - self.gt_lag, :]) for ds_idx in
                  range(self.amount_of_observations)]
        self.target_ts = ts_t_y
        self.covariates = ts_t_x

    def get_test_target_and_cov(self, input_chunk_length: int):
        """
        Get target time-series and covariates time-series which are split to fit model's input size for testing
        per sample

        Parameters
        ----------
        input_chunk_length : int
                    Amount of lags which a model will require as an input to predict final point

        Returns
        -------
        test_target_ts : ndarray
                    Split target time-series
        test_covariates : ndarray
                    Split time-series of covariates
        """
        self.test_target_ts, self.test_covariates = self.split_test_target_and_cov(
            input_chunk_length=input_chunk_length,
            amount_of_splits=self.target_ts_size)

        return self.test_target_ts, self.test_covariates
