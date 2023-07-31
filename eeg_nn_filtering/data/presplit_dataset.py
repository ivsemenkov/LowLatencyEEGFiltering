import typing as tp
import numpy as np
from .base_datasets import BaseChunkDataset


class PreSplitDataset(BaseChunkDataset):
    def __init__(self,
                 dataset_path: str,
                 first_observation_idx: int,
                 amount_of_observations: int,
                 input_size: int,
                 output_size: int,
                 gt_key: str,
                 split_ds_into_chunks: bool = False,
                 input_chunk_length: tp.Optional[int] = None,
                 output_chunk_length: tp.Optional[int] = None
                 ):
        """
        Class of a prepared and split dataset

        Parameters
        ----------
        dataset_path : str
                    Path to the multi-person real EEG dataset
        first_observation_idx : int
                    Index of a first observation (time-series) to consider. For example, if first_observation_idx == 5
                    it will ignore time-series with indices 0, 1, 2, 3, 4
        amount_of_observations : int
                    Amount of observations (time-series) to take from the pre-split dataset. For example, with
                    first_observation_idx == 5 and amount_of_observations == 10 it will take observations 5 to 5+10
                    from the pre-split dataset. Important: first_observation_idx + amount_of_observations cannot exceed
                    the amount of observations in the pre-split dataset
        input_size : int
                    How long (in samples) should be each time-series (there will be multiple of them).
                    Each such time-series is processed separately one sample at a time to imitate real-time framework
        output_size : int
                    How long (in samples) should be each time-series of ground truth (there will be multiple of them).
                    output_size should be <= input_size. If output_size < input_size then a ground truth series will
                    consist of filtered states for output_size latest samples from input_size. For example, if
                    input_size == 5 and output_size == 3 then each ground truth series will have filtered ground truth
                    for indices 2, 3, 4 of input series. That might be useful if algorithm cannot filter every single
                    sample (algorithm might need 999 more inputs as it uses 1000 lagged samples to filter the most
                    fresh one)
        gt_key : str
                    Key to the ground truth for the synthetic data. Note, in pre-split synthetic datasets we have keys
                    True_GT and HFIR_GT. True_GT will give an actual noiseless analytic signal from which noisy
                    samples were generated. HFIR_GT gives an approximation of True_GT by filtering noisy samples with
                    HFIR filter
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
        super().__init__(
            dataset_path=dataset_path,
            first_observation_idx=first_observation_idx,
            amount_of_observations=amount_of_observations,
            input_size=input_size,
            output_size=output_size,
            split_ds_into_chunks=split_ds_into_chunks,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
        )
        self.gt_key = gt_key
        self.load_dataset()

    def load_data(self):
        """
        Load whole dataset
        """
        data = np.load(self.dataset_path)
        full_X = data['Observations'][self.first_observation_idx:self.final_observation_idx, :, :]
        full_y = data[self.gt_key][self.first_observation_idx:self.final_observation_idx, :, :]
        assert full_X.shape[0] == self.final_observation_idx - self.first_observation_idx
        assert full_y.shape[0] == self.final_observation_idx - self.first_observation_idx

        # trim_dataset

        size_diff = self.input_size - self.output_size
        assert size_diff >= 0
        full_X = full_X[:, :self.input_size, :]
        self.X = full_X
        self.y = full_y[:, size_diff:size_diff + self.output_size, :]

        self.calculate_ds_mean()
        self.calculate_ds_std()
        if self.split_ds_into_chunks:
            self.split_into_chunks()
