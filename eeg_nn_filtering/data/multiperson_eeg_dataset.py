import typing as tp
import numpy as np
from .base_datasets import BaseChunkDataset


class RealEEGMultiPersonDataset(BaseChunkDataset):
    def __init__(self,
                 dataset_path: str,
                 first_observation_idx: int,
                 amount_of_observations: int,
                 ids_to_use: list,
                 input_size: int,
                 output_size: int,
                 gap: int,
                 shuffle_dataset: bool,
                 maximal_per_person_size: tp.Optional[int] = None,
                 split_ds_into_chunks: bool = False,
                 input_chunk_length: tp.Optional[int] = None,
                 output_chunk_length: tp.Optional[int] = None
                 ):
        """
        Class of a multi-person real EEG dataset

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
        ids_to_use : list
                    List of people's ids to use
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
        gap : int
                    Amount of samples to remove from the leftmost and rightmost parts of the dataset which are more
                    affected by transient response
        shuffle_dataset : bool
                    Initially, dataset is loaded in time order. If shuffle_dataset is True then it will shuffle the
                    data. Not recommended during testing
        maximal_per_person_size : int or None
                    If not None then it will take at most maximal_per_person_size time-series from a single person.
                    If None, takes the largest amount of available time-series. Can be used to avoid imbalance
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
            output_chunk_length=output_chunk_length
        )
        self.shuffle_dataset = shuffle_dataset
        self.ids_to_use = ids_to_use
        self.maximal_per_person_size = maximal_per_person_size
        self.gap = gap
        self.load_dataset()

    def split_segment(self, segment: tuple[np.ndarray, np.ndarray], full_X: list[np.ndarray], full_y: list[np.ndarray],
                      segment_states: np.ndarray, full_segment_states: list[np.ndarray]):
        """
        Split a single continuous segment of good samples into time-series

        Parameters
        ----------
        segment : tuple
                    Tuple with observations and ground truth of a good segment
        full_X : list
                    Current list of observations
        full_y : list
                    Current list of ground truth
        segment_states : ndarray
                    Array with markers of good and bad states for the segment
        full_segment_states : list
                    Current list of good and bad states

        Returns
        -------
        full_X : list
                    New current list of observations with new ones added
        full_y : list
                    New current list of ground truth with new ones added
        full_segment_states : list
                    New current list of good and bad states with new ones added
        """
        data_ch, gt = segment
        segment_size = len(data_ch)
        assert len(data_ch) == len(gt)
        n_obs = segment_size // self.min_observation_size
        for idx in range(n_obs):
            seg_X = data_ch[idx * self.min_observation_size: (idx + 1) * self.min_observation_size]
            seg_y = gt[idx * self.min_observation_size: (idx + 1) * self.min_observation_size]
            seg_states = segment_states[idx * self.min_observation_size: (idx + 1) * self.min_observation_size]
            seg_states = seg_states[self.size_diff:self.min_observation_size]
            X = seg_X[:self.input_size]
            y = seg_y[self.size_diff:self.min_observation_size]
            gt_states = np.stack((np.real(y), np.imag(y)), axis=1)
            full_X.append(X)
            full_y.append(gt_states)
            full_segment_states.append(seg_states)
        return full_X, full_y, full_segment_states

    def process_one_person_data(self, data: dict, idx: int):
        """
        Get all observations for single person id

        Parameters
        ----------
        data : dict
                    Loaded npz file with data
        idx : int
                    Person id

        Returns
        -------
        person_X : list
                    Per person list of observations
        person_y : list
                    Per person list of ground truth
        per_obs_states : list
                    Per person list of good and bad states
        """
        data_ch = data[f'subj_{idx}_P4'][self.gap:-self.gap]
        state = data[f'subj_{idx}_validity'][self.gap:-self.gap]
        gt = data[f'subj_{idx}_HFIR_GT'][self.gap:-self.gap]
        bitmask = state != -1
        indices2 = np.diff(bitmask)
        indices2 = np.where(indices2)[0] + 1
        segments = []
        start = 0
        per_segm_states = []
        for idx in indices2:
            if state[start] != -1:
                assert -1 not in state[start:idx]
                if idx - start >= self.min_observation_size:
                    segments.append((data_ch[start:idx], gt[start:idx]))
                    per_segm_states.append(state[start:idx])
            start = idx
        if state[start] != -1:
            assert -1 not in state[start:]
            idx = state.shape[0]
            if idx - start >= self.min_observation_size:
                segments.append((data_ch[start:idx], gt[start:idx]))
                per_segm_states.append(state[start:idx])
        person_X = []
        person_y = []
        per_obs_states = []
        for idx, segment in enumerate(segments):
            segment_states = per_segm_states[idx]
            person_X, person_y, per_obs_states = self.split_segment(segment, person_X, person_y, segment_states,
                                                                    per_obs_states)
        assert len(person_X) == len(person_y)
        if self.maximal_per_person_size is not None:
            person_X = person_X[:self.maximal_per_person_size]
            person_y = person_y[:self.maximal_per_person_size]
            per_obs_states = per_obs_states[:self.maximal_per_person_size]
        return person_X, person_y, per_obs_states

    def shuffle_ds(self, full_X: np.ndarray, full_y: np.ndarray, person_per_obs_states: np.ndarray):
        """
        Shuffle dataset

        Parameters
        ----------
        full_X : ndarray
                    Array of observations
        full_y : ndarray
                    Array of ground truth
        person_per_obs_states : ndarray
                    Array of good and bad states

        Returns
        -------
        full_X : ndarray
                    Shuffled array of observations
        full_y : ndarray
                    Shuffled array of ground truth
        person_per_obs_states : ndarray
                    Shuffled array of good and bad states
        """
        assert len(full_X) == len(full_y) == len(person_per_obs_states)
        np_rng = np.random.default_rng(0)
        shuffled_indices = np_rng.permutation(len(full_X))
        return full_X[shuffled_indices], full_y[shuffled_indices], person_per_obs_states[shuffled_indices]

    def load_data(self):
        """
        Load and preprocess whole dataset
        """
        assert self.input_size >= self.output_size
        data = np.load(self.dataset_path)
        self.min_observation_size = self.input_size
        self.size_diff = self.input_size - self.output_size
        full_X = []
        full_y = []
        per_obs_states = []
        for idx in self.ids_to_use:
            person_X, person_y, person_per_obs_states = self.process_one_person_data(data, idx)
            full_X.extend(person_X)
            full_y.extend(person_y)
            per_obs_states.extend(person_per_obs_states)
        full_X = np.array(full_X)
        full_y = np.array(full_y)
        per_obs_states = np.array(per_obs_states)
        if self.shuffle_dataset:
            full_X, full_y, per_obs_states = self.shuffle_ds(full_X, full_y, per_obs_states)
        print(f'With input_size={self.input_size} and output_size={self.output_size} '
              f'the largest amount of observations in dataset is {len(full_X)}')
        per_obs_states = per_obs_states[self.first_observation_idx:self.final_observation_idx]
        per_obs_dict = {state: np.sum(per_obs_states == state) for state in np.unique(per_obs_states)}
        total = sum(per_obs_dict.values())
        percentages = {state: value / total for state, value in per_obs_dict.items()}
        print(f'Validity distribution: {per_obs_dict}, percentages: {percentages}')
        full_X = full_X[self.first_observation_idx:self.final_observation_idx, :][..., None]
        full_y = full_y[self.first_observation_idx:self.final_observation_idx, :]
        assert full_X.shape == (self.final_observation_idx - self.first_observation_idx, self.input_size, 1)
        assert full_y.shape == (self.final_observation_idx - self.first_observation_idx, self.output_size, 2)
        self.X = full_X
        self.y = full_y
        self.calculate_ds_mean()
        self.calculate_ds_std()
        if self.split_ds_into_chunks:
            self.split_into_chunks()
