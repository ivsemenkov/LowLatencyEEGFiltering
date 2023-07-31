from abc import ABC
import numpy as np
import torch
import torch.utils.data.dataset as dataset


class BaseTorchDataset(dataset.Dataset, ABC):
    def __init__(self):
        """
        Base class for the dataset
        """
        super().__init__()
        self.input_mean = None
        self.input_std = None
        self.X = None

    def scale_inputs_by_std(self):
        """
        Scales inputs by self.input_std
        """
        self.X = self.X / self.input_std

    def set_ds_std(self, deviation: float):
        """
        Sets self.input_str

        Parameters
        ----------
        deviation : float
                    Standard deviation to use for potential scaling
        """
        self.input_std = deviation

    def set_ds_mean(self, mean: float):
        """
        Sets self.input_mean

        Parameters
        ----------
        mean : float
                    Mean to use
        """
        self.input_mean = mean

    def calculate_ds_std(self):
        """
        Calculates and records standard deviation of inputs
        """
        self.input_std = np.std(self.X)

    def calculate_ds_mean(self):
        """
        Calculates and records mean of inputs
        """
        self.input_mean = np.mean(self.X)

    def get_ds_std(self):
        """
        Returns inputs standard deviation
        """
        return self.input_std

    def get_ds_mean(self):
        """
        Returns inputs mean
        """
        return self.input_mean


class BaseChunkDataset(BaseTorchDataset):
    def __init__(self,
                 dataset_path: str,
                 first_observation_idx: int,
                 amount_of_observations: int,
                 input_size: int,
                 output_size: int,
                 split_ds_into_chunks: bool,
                 input_chunk_length: int,
                 output_chunk_length: int
                 ):
        """
        Base class of a prepared dataset which can be split in chunks if needed

        Parameters
        ----------
        dataset_path : str
                    Path to the dataset
        first_observation_idx : int
                    Index of a first observation (time-series) to consider. For example, if first_observation_idx == 5
                    it will ignore time-series with indices 0, 1, 2, 3, 4
        amount_of_observations : int
                    Amount of observations (time-series) to take from the pre-split dataset. For example, with
                    first_observation_idx == 5 and amount_of_observations == 10 it will take observations 5 to 5+10
                    from the pre-split dataset. Important: first_observation_idx + amount_of_observations cannot exceed
                    the amount of observations in the pre-split dataset
        input_size : int
                    How long (in samples) should be each time-series of EEG data (there will be multiple of them).
                    Each such time-series is processed separately one sample at a time to imitate real-time framework
        output_size : int
                    How long (in samples) should be each time-series of HFIR filtered ground truth EEG data
                    (there will be multiple of them). output_size should be <= input_size. If output_size < input_size
                    then a ground truth series will consist of filtered states for output_size latest samples from
                    input_size. For example, if input_size == 5 and output_size == 3 then each ground truth series will
                    have filtered ground truth for indices 2, 3, 4 of input series. That might be useful if
                    algorithm cannot filter every single sample (algorithm might need 999 more inputs as it uses 1000
                    lagged samples to filter the most fresh one)
        split_ds_into_chunks : bool
                    If True datasets will return specific chunks of length equal to the models input size instead of
                    larger time-series which are supposed to be filtered one sample at a time in correct order. This
                    is useful for training as the whole data is available from the beginning, and it helps to use less
                    GPU space by holding fewer data in each batch at the same time. Dataset does not actually split
                    everything to chunks to avoid holding in memory same lags multiple times. Dataset calculates
                    indices to use them to accesses chunks from the whole dataset
        input_chunk_length : int
                    Size of a single input chunk if split_ds_into_chunks is True. Preferably should be equal to the
                    input size of the model
        output_chunk_length : int
                    Size of a single output chunk if split_ds_into_chunks is True. Here, the filtering is not done one
                    sample at a time. This size refers to how many final elements of the input_chunk_length do we want
                    to use to calculate loss values
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.first_observation_idx = first_observation_idx
        self.final_observation_idx = self.first_observation_idx + amount_of_observations
        self.amount_of_observations = amount_of_observations
        self.input_size = input_size
        self.output_size = output_size
        self.original_input_size = input_size
        self.original_output_size = output_size
        self.original_amount_of_observations = amount_of_observations
        self.split_ds_into_chunks = split_ds_into_chunks
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.X = None
        self.y = None

    def load_data(self):
        """
        Should be overwritten
        """
        raise NotImplementedError

    def split_into_chunks(self):
        """
        Calculates sizes and amount of chunks
        """
        # amount of chunks per observation = output size
        assert self.input_size - self.output_size - self.input_chunk_length - self.output_chunk_length + 1 >= 0
        self.input_size = self.input_chunk_length
        self.output_size = self.output_chunk_length
        self.amount_of_observations = self.X.shape[0] * (self.original_output_size - self.output_chunk_length + 1)

    def get_chunk_data(self, idx: int):
        """
        Returns single chunk of inputs and ground truth

        Parameters
        ----------
        idx : int
                    Index of a required datapoint (chunk)

        Returns
        -------
        input_data : torch.Tensor
                    Chunk of unfiltered, noisy inputs
        gt_states : torch.Tensor
                    Chunk of ground truth analytic signal
        """
        obs_idx, output_idx = divmod(idx, self.original_output_size - self.output_chunk_length + 1)
        x_start_idx = self.X.shape[1] - self.input_chunk_length - output_idx
        x_fin_idx = self.X.shape[1] - output_idx
        y_start_idx = self.y.shape[1] - self.output_chunk_length - output_idx
        y_fin_idx = self.y.shape[1] - output_idx
        input_tensor = self.X[obs_idx, x_start_idx:x_fin_idx, :]
        gt_tensor = self.y[obs_idx, y_start_idx:y_fin_idx, :]
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_tensor, dtype=torch.float32)
        return input_tensor, gt_tensor

    def __len__(self):
        """
        Get dataset size

        Returns
        -------
        amount_of_observations : int
                    Amount of time-series or their chunks in dataset
        """
        return self.amount_of_observations

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
        if self.split_ds_into_chunks:
            input_tensor, gt_tensor = self.get_chunk_data(idx=idx)
        else:
            input_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
            gt_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        return input_tensor, gt_tensor

    def load_dataset(self):
        """
        Loads data into this class and checks consistency
        """
        self.load_data()
        assert self.X.shape == (self.original_amount_of_observations, self.original_input_size, 1)
        assert self.y.shape == (self.original_amount_of_observations, self.original_output_size, 2)
