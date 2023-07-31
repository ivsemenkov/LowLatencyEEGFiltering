import typing as tp
import numpy as np
import torch
from .base_datasets import BaseTorchDataset
from general_utilities.data_generator import generate_datapoint


class GeneratedSyntheticDataset(BaseTorchDataset):
    def __init__(self,
                 duration: float,
                 fs: float,
                 dataset_name: str,
                 frequency: tp.Optional[float],
                 filter_band: tp.Optional[tuple[float, float]],
                 input_size: int,
                 output_size: int,
                 amount_of_observations: int,
                 noise_coeff: float,
                 rng_seed: tp.Optional[int] = 0
                 ):
        """
        Dataset class which generates PyTorch dataset for specified parameters

        Parameters
        ----------
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
        input_size : int
                    How long (in samples) should be each time-series of EEG data (there will be multiple of them).
                    Each such time-series is processed separately one sample at a time to imitate real-time framework
        output_size : int
                    How long (in samples) should be each time-series of true ground truth
                    (there will be multiple of them). output_size should be <= input_size. If output_size < input_size
                    then a ground truth series will consist of filtered states for output_size latest samples from
                    input_size. For example, if input_size == 5 and output_size == 3 then each ground truth series will
                    have filtered ground truth for indices 2, 3, 4 of input series. That might be useful if
                    algorithm cannot filter every single sample (algorithm might need 999 more inputs as it uses 1000
                    lagged samples to filter the most fresh one)
        amount_of_observations : int
                    Amount of time-series (observations) in a resultant dataset. Total dataset size is
                    duration * fs * amount_of_observations samples. Each time-series will be generated independently
        noise_coeff : float
                    Noise coefficient by which a sample of noise is amplified. Used to calculate Noise level. For white
                    noise Noise level (NL) = noise_coeff. For pink noise Noise level = 10 * noise_coeff
        rng_seed : int
                    A seed for the random number generator used to generated dataset. Useful for reproducibility.
                    If None, then unpredictable seed will be pulled from the OS
        """
        super().__init__()
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.duration = duration
        self.fs = fs
        self.dataset_name = dataset_name
        self.frequency = frequency
        self.input_size = input_size
        self.output_size = output_size
        self.amount_of_observations = amount_of_observations
        self.noise_coeff = noise_coeff
        self.filter_band = filter_band
        self.load_torch_dataset()

    def generate_data(self):
        """
        Generate a single time-series (observation)

        Returns
        -------
        input_data : ndarray
                    Time-series of unfiltered, noisy inputs
        gt_states : ndarray
                    Time-series of noise-free ground truth analytic signal from which input_data was created
        """
        input_data, gt_states = generate_datapoint(duration_sec=self.duration,
                                                   amount_of_steps=0,
                                                   fs=self.fs, dataset_name=self.dataset_name,
                                                   frequency=self.frequency,
                                                   filter_band=self.filter_band,
                                                   noise_coeff=self.noise_coeff,
                                                   n_samp=None, rng=self.rng)
        assert np.iscomplexobj(gt_states)
        gt_states = np.stack((np.real(gt_states), np.imag(gt_states)), axis=1)
        input_data = input_data[..., None]
        return input_data, gt_states

    def load_data(self):
        """
        Generates a whole dataset and writes inputs into self.X, ground truth into self.y
        """
        full_X = []
        full_y = []
        for _ in range(self.amount_of_observations):

            input_data, gt_states = self.generate_data()
            full_X.append(input_data)
            full_y.append(gt_states)

        self.X = np.array(full_X)
        self.y = np.array(full_y)
        assert self.X.shape == (self.amount_of_observations, self.duration * self.fs, 1)
        assert self.y.shape == (self.amount_of_observations, self.duration * self.fs, 2)

        # trim_dataset

        size_diff = self.input_size - self.output_size
        assert size_diff >= 0
        self.X = self.X[:, :self.input_size, :]
        self.y = self.y[:, size_diff:size_diff + self.output_size, :]
        self.calculate_ds_mean()
        self.calculate_ds_std()

    def to_torch(self):
        """
        Move dataset from numpy arrays to torch tensors
        """
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        """
        Get dataset size

        Returns
        -------
        amount_of_observations : int
                    Amount of generated time-series
        """
        return self.X.size(0)

    def __getitem__(self, idx: int):
        """
        Return input datapoint and respective ground truth

        Parameters
        ----------
        idx : int
                    Index of time-series or a chunk

        Returns
        -------
        input_tensor : torch.Tensor
                    Time-series or chunk of unfiltered, noisy inputs
        gt_tensor : torch.Tensor
                    Time-series or chunk of ground truth analytic signal
        """
        input_tensor = self.X[idx]
        gt_tensor = self.y[idx]
        return input_tensor, gt_tensor

    def load_torch_dataset(self):
        """
        Load whole dataset, check consistency and move numpy arrays to torch tensors
        """
        self.load_data()
        assert self.X.shape == (self.amount_of_observations, self.input_size, 1)
        assert self.y.shape == (self.amount_of_observations, self.output_size, 2)
        self.to_torch()
